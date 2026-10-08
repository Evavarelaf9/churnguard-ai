# Auditoria de requisitos — ChurnGuard AI

Conferência item a item da especificação do projeto contra o código entregue. Para cada requisito:
**onde está implementado** e o **status** após a auditoria.

Legenda: ✅ atendido desde a entrega · 🔧 lacuna encontrada na auditoria e corrigida · ⚠️ atendido com limitação documentada

## Resumo

| Resultado | Itens |
|---|---|
| ✅ Atendidos sem ajuste | 21 |
| 🔧 Corrigidos na auditoria | 8 |
| ⚠️ Com limitação documentada | 1 |

## Lacunas encontradas e correções

| # | Lacuna | Correção | Onde |
|---|---|---|---|
| A | A explicação individual listava fatores, mas não havia **uma frase-resumo em linguagem de negócio** ("O fator que mais contribuiu…") | `ChurnExplainer.narrative()` gera o `resumo`, que a API, o banco e a tela do cliente passam a mostrar | `ml/src/explainability/explainer.py`, `PredictionResult.resumo`, `CustomerDetail.tsx` |
| B | Só o limite de **alto** risco era configurável; o de **médio** era fixo | Parâmetro `limite_medio` em todas as rotas de risco + segundo controle deslizante na interface (médio ≤ alto) | `ChurnScorer.thresholds()`, rotas, `ThresholdControl.tsx`, `RiskSettings.tsx` |
| C | O filtro **período** listava datas, mas as previsões históricas não mostravam o que aconteceu depois | Datas passadas trazem as previsões daquele dia (inclusive de quem cancelou) com a coluna **Resultado em 60 dias** | `repositories.list_latest`, `Customers.tsx` |
| D | A simulação não permitia alterar o **volume de chamados** | Campo "Chamados nos últimos 3 meses" (`tickets_3m`), com ajuste coerente dos chamados de 6 meses | `services/scoring.apply_changes`, `SimulationPanel.tsx` |
| E | O monitoramento calculava o **volume de previsões**, mas a tela não o exibia | Gráfico de volume por data e origem | `MonitoringCharts.tsx`, `Monitoring.tsx` |
| F | A decisão humana de **não agir** não podia ser registrada | Novo tipo de ação "Revisão humana: sem ação" (migration 004) | `database/migrations/004_revisao_humana.sql`, `ActionModal.tsx` |
| G | Registrar/atualizar ações não dava **confirmação visual** | Aviso acessível (`aria-live`) de sucesso ou erro | `context/Toast.tsx` |
| H | A tabela de threshold mostrava precision/recall, mas não **quantos cancelamentos seriam encontrados** | Colunas "Abordados" e "Churns encontrados (x de y)" | `business.threshold_tradeoff`, `ModelPage.tsx` |

Também nesta rodada: as previsões históricas do seed passaram a ser geradas com explicação completa (principal
fator, recomendação e resumo), para que o período histórico tenha o mesmo nível de detalhe da previsão atual.

## Conferência dos requisitos

| Requisito | Status | Implementação |
|---|---|---|
| Problema de negócio e objetivo | ✅ | README §1–3; landing page |
| Dataset (realista, dicionário, limitações) | ✅ | `ml/src/data/generate_synthetic.py`, [`dataset.md`](dataset.md) |
| Análise exploratória | ✅ | `ml/src/eda/`, `ml/notebooks/01_eda.ipynb`, [`eda.md`](eda.md) |
| Limpeza e qualidade dos dados | ✅ | Problemas injetados e tratados (duplicatas, ausentes, domínio, outliers) |
| Engenharia de atributos | ✅ | `ml/src/features/build_features.py` (tendências, taxas, janelas point-in-time) |
| Pipeline reprodutível (sem vazamento) | ✅ | `FeatureEngineer` + `ColumnTransformer` em `ml/src/preprocessing/pipeline.py` |
| Divisão temporal | ✅ | Treino até set/2025, intervalo de 60 dias, teste dez/2025–jul/2026; CV `TemporalGroupSplit` |
| Modelos (Regressão Logística, Random Forest, Gradient Boosting) | ✅ | `ml/src/training/models.py`; comparação em [`model.md`](model.md) |
| Métricas (precision, recall, F1, ROC-AUC, PR-AUC, matriz de confusão) | ✅ | `ml/src/evaluation/metrics.py`; tela *Modelo* |
| Desbalanceamento | ✅ | Pesos, undersampling e SMOTE comparados; decisão: threshold (preserva a calibração) |
| Ajuste de hiperparâmetros | ✅ | Busca aleatória com CV temporal (`train.py`) |
| Threshold configurável | 🔧 | Alto **e médio** configuráveis (item B); trade-off com churns encontrados (item H) |
| Explicabilidade global (SHAP / permutation) | ✅ | `docs/images/model/`, telas *Visão geral* e *Modelo* |
| Explicação individual em linguagem de negócio | 🔧 | Fatores + recomendação + **resumo** (item A) |
| Níveis de risco configuráveis | 🔧 | `RISK_THRESHOLDS` + `limite_medio` (item B) |
| Dashboard com KPIs e gráficos | ✅ | `Dashboard.tsx` |
| Tabela de clientes com filtros (risco, plano, região, valor, período) | 🔧 | Período histórico com resultado real (item C) |
| Página de detalhe do cliente | ✅ | `CustomerDetail.tsx` |
| Simulação de retenção | 🔧 | Uso, atrasos, reclamações, **chamados** (item D), plano, pagamento, renovação |
| API (predict, batch, risk, metrics, health, OpenAPI) | ✅ | `backend/app/api/routes`, [`api.md`](api.md), [`openapi.json`](openapi.json) |
| Arquitetura em camadas | ✅ | [`architecture.md`](architecture.md) |
| Banco PostgreSQL (clientes, planos, pagamentos, uso, chamados, previsões, ações; previsto × real) | 🔧 | Migrations 001–004; `v_prediction_outcomes` |
| Monitoramento (desempenho, churn real × previsto, drift, volume, FP/FN) | 🔧 | Volume agora visível na tela (item E); [`monitoring.md`](monitoring.md) |
| Ações de retenção com decisão humana | 🔧 | Revisão humana sem ação (item F) + confirmação (item G) |
| Ética e LGPD | ✅ | [`lgpd.md`](lgpd.md), auditoria de viés (`ml/src/evaluation/fairness.py`), `access_log` |
| Stack especificada | ✅ | Python, scikit-learn, SHAP, FastAPI, PostgreSQL, React + TS, Tailwind, Docker |
| Interface moderna, responsiva e acessível (tooltips, carregamento, vazio, erro) | ✅ | `components/ui`, testes de acessibilidade |
| Landing page (título, texto, CTA, 4 indicadores) | ✅ | `Landing.tsx` |
| README (21 seções, imagens, GIF) | ✅ | `README.md`, `docs/images/` |
| Execução com Docker | ⚠️ | `docker-compose.yml` validado (`docker compose config`, bootstrap limpo, Nginx); a construção das imagens não pôde ser executada no ambiente de desenvolvimento (Docker Hub indisponível) |

## Verificação

| Suíte | Resultado |
|---|---|
| ML (`pytest ml/tests`) | 20 testes passando |
| API com PostgreSQL (`pytest backend/tests`) | 33 testes passando (6 novos cobrindo os itens A–D, F, H) |
| Frontend (`vitest`) | 10 testes passando (2 novos: aviso de confirmação e limites alto/médio) |
| `ruff`, `tsc`, `vite build` | Sem erros |
