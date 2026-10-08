# Arquitetura — ChurnGuard AI

## 1. Visão geral

```mermaid
flowchart TB
    subgraph Apresentação
        FE["Frontend<br/>React + TypeScript + Tailwind"]
    end
    subgraph API["API REST — FastAPI (backend/app)"]
        R["Rotas / contratos<br/>api/routes · schemas (Pydantic)"]
        S["Camada de serviços<br/>services/scoring · services/business"]
        RP["Repositórios<br/>repositories/"]
    end
    subgraph ML["Machine Learning (ml/src)"]
        FEAT["Atributos point-in-time<br/>build_snapshot · add_features"]
        MOD["Pipeline serializado<br/>churn_model.joblib"]
        EXP["Explicabilidade<br/>SHAP → frases de negócio"]
    end
    subgraph Dados
        DB[("PostgreSQL<br/>operacional + previsões + ações")]
    end
    JOBS["Jobs agendados<br/>score_customers · update_outcomes"]
    MON["Monitoramento<br/>desempenho · drift · alertas"]

    FE -->|HTTPS + X-API-Key| R --> S
    S --> RP --> DB
    S --> FEAT --> MOD --> EXP
    JOBS --> S
    DB --> MON
```

| Camada | Pasta | Responsabilidade | Não faz |
|---|---|---|---|
| Apresentação | `frontend/` | Telas, filtros, gráficos | Regra de negócio, cálculo de risco |
| API | `backend/app/api`, `schemas` | HTTP, validação de entrada/saída, códigos de erro, autenticação | Consultas SQL, lógica de ML |
| Serviços | `backend/app/services` | Regras de negócio: threshold efetivo, recomendação, simulação coerente, auditoria | Detalhes de HTTP |
| Repositórios | `backend/app/repositories` | Consultas SQL e mapeamento ORM | Regras de negócio |
| ML | `ml/src` | Atributos, modelo, explicação, métricas | Conhecer o banco ou a API |
| Persistência | `database/migrations` | Schema versionado, restrições, views analíticas | — |
| Jobs | `backend/app/jobs` | Pontuação em lote e registro do resultado real | — |
| Monitoramento | `ml/src/monitoring`, `backend/app/services/monitoring.py` | Desempenho, calibração, drift (PSI), alertas | Re-treinar automaticamente (decisão humana) |

O pacote `ml` **não depende** do backend: pode ser usado em notebooks, jobs ou outra API. O backend usa o `ml`
como biblioteca.

## 2. Fluxo de uma previsão a partir do banco

```mermaid
sequenceDiagram
    autonumber
    participant U as Frontend / cliente HTTP
    participant A as FastAPI (rota /customers/{id}/risk)
    participant S as scoring service
    participant R as raw_data repository
    participant DB as PostgreSQL
    participant M as ChurnScorer (ml)

    U->>A: GET /api/v1/customers/CLI-07443/risk (X-API-Key)
    A->>A: valida chave → usuário
    A->>S: score_customer(id, data_referencia, threshold)
    S->>DB: cliente existe? está ativo?
    S->>R: load_raw_frames(ref, [id])
    R->>DB: uso, faturas, chamados, pesquisas ≤ data de referência
    R-->>S: DataFrames no formato do treino
    S->>M: build_snapshot → predict_proba → SHAP
    M-->>S: probabilidade, fatores, categoria principal
    S->>S: nível de risco (threshold efetivo) + recomendação
    S->>DB: INSERT predictions (entrada, fatores, recomendação)
    A->>DB: INSERT access_log (usuário, ação, cliente)
    A-->>U: 200 PredictionResult
```

**Consistência treino × produção.** O risco calculado on-line usa exatamente as mesmas funções do treino
(`build_snapshot` e o pipeline serializado). Um teste automatizado garante que, para uma data passada,
a probabilidade calculada a partir do banco é idêntica à obtida com a base analítica usada no treino.

## 3. Ciclo de vida do modelo

```mermaid
flowchart LR
    A["Dados brutos"] --> B["Base analítica<br/>(point-in-time)"]
    B --> C["Treino + CV temporal"]
    C --> D["Avaliação no teste<br/>(uma vez)"]
    D --> E["Re-treino com todos<br/>os dados rotulados"]
    E --> F["model_versions<br/>(versão ativa)"]
    F --> G["Job: score_customers<br/>(previsões 'batch')"]
    G --> H["Dashboard / ações<br/>de retenção"]
    G --> I["Job: update_outcomes<br/>(após 60 dias)"]
    I --> J["previsto × real<br/>monitoramento"]
    J -->|drift ou queda de performance| C
```

## 4. Banco de dados

PostgreSQL 16. Schema versionado em [`database/migrations`](../database/migrations), aplicado por
`python -m backend.app.db.migrate` (cada arquivo em uma transação, registrado em `schema_migrations`).

```mermaid
erDiagram
    plans ||--o{ customers : "plano atual"
    customers ||--o{ usage_monthly : "uso mensal"
    customers ||--o{ payments : "faturas"
    customers ||--o{ tickets : "chamados"
    customers ||--o{ surveys : "pesquisas"
    customers ||--o{ predictions : "previsões"
    model_versions ||--o{ predictions : "gerada por"
    customers ||--o{ retention_actions : "ações"
    predictions |o--o{ retention_actions : "motivada por"
    customers ||--o{ access_log : "acessos (LGPD)"

    plans {
        smallint plano_id PK
        varchar plano
        numeric preco_base
        smallint funcionalidades_disponiveis
    }
    customers {
        varchar customer_id PK
        varchar nome
        smallint idade "só auditoria"
        varchar regiao "só auditoria"
        varchar tipo_cliente
        smallint plano_id FK
        varchar ciclo_contrato
        varchar forma_pagamento
        bool renovacao_automatica
        date data_inicio
        date data_cancelamento
        varchar motivo_cancelamento "pós-evento"
        varchar status
    }
    usage_monthly {
        varchar customer_id PK
        date mes_referencia PK
        int acessos
        int minutos_uso
        smallint dias_ativos
        date ultimo_acesso
    }
    payments {
        varchar payment_id PK
        varchar customer_id FK
        numeric valor_cobrado
        date data_vencimento
        date data_pagamento
        varchar status
    }
    tickets {
        varchar ticket_id PK
        varchar customer_id FK
        date data_abertura
        varchar categoria
        numeric tempo_resolucao_horas
        smallint avaliacao
    }
    surveys {
        bigint survey_id PK
        varchar customer_id FK
        date data_resposta
        smallint nota_satisfacao
    }
    model_versions {
        varchar version PK
        varchar model_name
        numeric threshold
        jsonb risk_thresholds
        jsonb metrics
        bool is_active
    }
    predictions {
        bigint id PK
        varchar customer_id FK
        date data_referencia
        varchar model_version FK
        numeric probabilidade
        varchar nivel_risco
        jsonb fatores
        jsonb recomendacao
        jsonb entrada
        varchar origem
        smallint churn_real
        date resultado_em
    }
    retention_actions {
        bigint id PK
        varchar customer_id FK
        bigint prediction_id FK
        varchar tipo_acao
        varchar status
        varchar responsavel
        varchar resultado
    }
    access_log {
        bigint id PK
        varchar usuario
        varchar acao
        varchar customer_id
        timestamptz created_at
    }
```

### Decisões de modelagem

| Decisão | Motivo |
|---|---|
| Tabelas operacionais espelham os sistemas de origem (uso, faturas, chamados) | O banco guarda **fatos**; atributos do modelo são calculados na hora, com regras point-in-time |
| `predictions.entrada` (JSONB) guarda os atributos usados | Auditoria ("com quais dados o modelo decidiu?") e monitoramento de drift sem recalcular |
| `predictions.churn_real` + `resultado_em` | Responde **"o que o modelo previu?" × "o que realmente aconteceu?"** |
| `predictions.origem` (`batch`, `api`, `historico`) | Separa pontuação agendada, consultas on-line e previsões históricas do período de teste |
| `model_versions` com índice único parcial em `is_active` | Garante uma única versão ativa; previsões sempre rastreáveis à versão |
| `retention_actions.prediction_id` | Liga a decisão humana à previsão que a motivou (medir efetividade das ações) |
| `access_log` | Rastreabilidade de quem consultou dados de qual cliente (LGPD) |
| `CHECK` constraints (domínios, `idade 18–100`, `status` × `data_cancelamento`) | Qualidade garantida no banco, não só na aplicação |
| `motivo_cancelamento` documentado como pós-evento | Evita que alguém o use como atributo no futuro (vazamento) |

### Views

| View | Uso |
|---|---|
| `v_latest_predictions` | Previsão mais recente de cada cliente (tabela do dashboard) |
| `v_prediction_outcomes` | Cada previsão classificada como VP, FP, FN, VN ou "Aguardando" |
| `v_monthly_model_performance` | Previsto × real e matriz de confusão por data de referência |

Tabela `monitoring_runs` (migration 003): cada execução do monitoramento com status, alertas, desempenho e drift.
Migration 004: novo tipo de ação `Revisão humana: sem ação`, para registrar quando a equipe decide não agir.

## 5. Implantação com Docker

```mermaid
flowchart LR
    U(["Navegador"]) -->|":8080"| NG["frontend<br/>Nginx + React (build)"]
    NG -->|"/api + X-API-Key<br/>(injetada no servidor)"| API["api<br/>FastAPI · 2 workers"]
    DEV(["Swagger / integrações"]) -->|":8000 + X-API-Key"| API
    API --> DB[("db<br/>PostgreSQL 16<br/>volume pgdata")]
    BS["bootstrap<br/>(executa uma vez)"] -->|"migrations · seed ·<br/>pontuação · monitoramento"| DB
    BS -.-> VOL[("volume mldata<br/>dataset gerado")]
    JOBS["jobs (perfil opcional)<br/>resultado real → pontuação → monitoramento"] --> DB
```

| Serviço | Imagem | Papel | Depende de |
|---|---|---|---|
| `db` | `postgres:16-alpine` | Banco, volume persistente, porta só em `127.0.0.1` | — |
| `bootstrap` | `churnguard-api` | Gera dados (se faltarem), migrations, seed, 1º monitoramento; idempotente | `db` saudável |
| `api` | `churnguard-api` | FastAPI (usuário sem privilégios, healthcheck em `/health`) | `bootstrap` concluído |
| `frontend` | `churnguard-frontend` | Nginx: SPA, cache de assets, cabeçalhos de segurança, proxy da API com a chave | `api` saudável |
| `jobs` | `churnguard-api` | Rotina diária sob demanda (`--profile jobs`) | `db` saudável |

## 6. Segurança e privacidade (resumo)

- Autenticação por `X-API-Key` em toda a API versionada; `/health` público.
- Registro de acesso a dados de clientes (`access_log`) com o usuário da chave.
- Minimização: listagens e detalhes não expõem idade; região é usada apenas como filtro de visualização.
- CORS restrito às origens configuradas; métodos limitados a GET/POST/PATCH.
- Segredos via variáveis de ambiente (`.env`, nunca versionado — ver `.env.example`).
- Detalhes em [`docs/lgpd.md`](lgpd.md).

## 7. Estrutura do backend

```text
backend/
├── app/
│   ├── main.py              # FastAPI: middlewares, routers, OpenAPI, carga do modelo no startup
│   ├── core/                # config (pydantic-settings), segurança (API key)
│   ├── db/                  # sessão, executor de migrations, seed, bootstrap
│   ├── models/entities.py   # ORM (SQLAlchemy 2.0)
│   ├── schemas/schemas.py   # contratos Pydantic (entrada/saída)
│   ├── repositories/        # SQL: raw_data (formato do ML) e repositórios de domínio
│   ├── services/            # scoring (modelo + banco) e business (clientes, dashboard, ações)
│   ├── api/routes/          # predict, customers, operations (actions, dashboard), system
│   ├── services/monitoring.py  # desempenho, calibração, drift (PSI), alertas
│   └── jobs/                # score_customers, update_outcomes, monitor
├── tests/                   # testes de integração com PostgreSQL real
├── Dockerfile
└── requirements.txt
```
