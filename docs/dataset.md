# Dataset — origem, dicionário de dados e limitações

## 1. Origem

O ChurnGuard AI usa uma **base sintética realista**, gerada por `ml/src/data/generate_synthetic.py`
(semente fixa `42`, totalmente reprodutível).

### Por que não uma base pública?

| Critério | Base pública típica (ex.: Telco/IBM) | Base sintética do projeto |
|---|---|---|
| Histórico de uso mês a mês | ❌ | ✅ 57 meses |
| Pagamentos com data de vencimento/pagamento | ❌ | ✅ |
| Chamados de suporte com categoria, tempo e avaliação | ❌ | ✅ |
| Atributos de *mudança de comportamento* | ❌ impossível | ✅ |
| Divisão temporal honesta (treino no passado, teste no futuro) | ❌ | ✅ |
| Simulação de *data drift* para monitoramento | ❌ | ✅ (degradação do suporte em 2026) |
| Dados pessoais reais | ⚠️ depende | ✅ nenhum (nomes fictícios) |

### Como os dados são gerados

A simulação roda **mês a mês** de jan/2022 a set/2026 para 9.000 clientes. Cada cliente tem **traços latentes**
(engajamento, fragilidade financeira, sensibilidade a preço, satisfação, afinidade com concorrentes) que evoluem no
tempo com choques aleatórios — por exemplo, um "episódio de desengajamento" (fim de projeto, troca de ferramenta)
ou um "episódio de problema técnico".

Os traços latentes **não são exportados**. O que chega às tabelas é só o que uma empresa real registraria: acessos,
faturas, chamados, pesquisas. O cancelamento é sorteado a partir de um risco mensal (*hazard*) que combina:

- engajamento baixo e **queda recente** de engajamento;
- atrasos de pagamento e faturas em aberto;
- reclamações e baixa satisfação;
- problemas técnicos;
- **reajuste anual de janeiro** × sensibilidade a preço;
- canal de aquisição, renovação automática, forma de pagamento e tempo de casa;
- contratos anuais só saem (sem multa) no aniversário de 12 meses.

**Região e idade não influenciam o risco por construção.** Isso permite verificar, depois, se o modelo está
aprendendo correlações espúrias com atributos sensíveis.

### Problemas de qualidade injetados de propósito

| Problema | Proporção | Objetivo didático |
|---|---|---|
| Duplicatas exatas em `customers` | 25 linhas | Deduplicação na ingestão |
| `idade` ausente | ~3% | Tratamento de ausentes |
| `idade` impossível (0, 1, 112, 120, 150) | 12 linhas | Validação de domínio |
| `regiao` ausente | ~0,5% | Categoria "não informado" |
| `minutos_uso` extremos (integrações/bots) | ~0,2% | Winsorização |
| *Drift*: tempo de resolução de chamados +35% a partir de mar/2026 | — | Monitoramento |

## 2. Tabelas brutas (`ml/data/raw/`)

Espelham o modelo relacional do PostgreSQL (`database/migrations`).

### `plans.csv`
| Coluna | Tipo | Descrição |
|---|---|---|
| `plano_id` | int | Identificador do plano |
| `plano` | texto | Básico, Essencial, Profissional, Empresarial |
| `preco_base` | decimal | Preço de tabela mensal (R$) |
| `funcionalidades_disponiveis` | int | Quantidade de módulos liberados |
| `usuarios_incluidos` | int | Usuários incluídos no plano |

### `customers.csv`
| Coluna | Tipo | Descrição | Uso no modelo |
|---|---|---|---|
| `customer_id` | texto | Identificador pseudonimizado (`CLI-00001`) | chave |
| `nome` | texto | Nome **fictício** | ❌ nunca |
| `idade` | int | Idade do titular | ❌ só auditoria |
| `regiao` | texto | Região do Brasil | ❌ só auditoria |
| `tipo_cliente` | texto | Pessoa Física, Pequena Empresa, Média Empresa | ✅ |
| `plano_id` | int | Plano atual | ✅ |
| `ciclo_contrato` | texto | Mensal ou Anual (fidelidade 12 meses, 15% de desconto) | ✅ |
| `forma_pagamento` | texto | Cartão de crédito, Boleto, Pix, Débito automático | ✅ |
| `renovacao_automatica` | bool | Renovação automática ativada | ✅ |
| `canal_aquisicao` | texto | Orgânico, Indicação, Anúncios pagos, Parceiros, Vendas diretas | ✅ |
| `data_inicio` | data | Data de contratação | ✅ (tempo de casa) |
| `data_cancelamento` | data | Data do cancelamento | 🎯 só para o rótulo |
| `motivo_cancelamento` | texto | Motivo informado **após** o cancelamento | ⛔ vazamento |
| `status` | texto | Ativo / Cancelado (situação **hoje**) | ⛔ vazamento |

### `usage_monthly.csv`
| Coluna | Descrição |
|---|---|
| `customer_id`, `mes_referencia` | Chave (1ª dia do mês) |
| `acessos` | Logins no mês |
| `sessoes` | Sessões de uso |
| `minutos_uso` | Tempo total de uso |
| `funcionalidades_usadas` | Módulos distintos utilizados |
| `dias_ativos` | Dias com pelo menos um acesso |
| `ultimo_acesso` | Data do último acesso no mês (vazio se não acessou) |

### `payments.csv`
| Coluna | Descrição |
|---|---|
| `payment_id` | Identificador da fatura |
| `customer_id`, `mes_referencia` | Cliente e competência |
| `valor_cobrado` | Valor da fatura (com desconto e reajustes) |
| `data_vencimento` | Sempre dia 10 |
| `data_pagamento` | Vazio se em aberto |
| `status` | Pago em dia, Pago com atraso, Em aberto |

### `tickets.csv`
| Coluna | Descrição |
|---|---|
| `ticket_id`, `customer_id` | Chaves |
| `data_abertura`, `data_resolucao` | Abertura e fechamento |
| `categoria` | Técnico, Cobrança, Dúvida, Reclamação |
| `tempo_resolucao_horas` | Horas até a resolução |
| `avaliacao` | Nota 1–5 (≈55% respondem) |

### `surveys.csv`
Pesquisa de satisfação trimestral (0–10), respondida por ~⅓ dos clientes.

### `price_adjustments.csv`
Reajustes anuais aplicados em janeiro para clientes com 6+ meses: 2023 (6%), 2024 (5%), 2025 (7%), 2026 (9%).

## 3. Base analítica (`ml/data/processed/analytical_base.parquet`)

Construída por `ml/src/data/build_analytical_base.py`.

- **Grão:** cliente ativo × data de referência (fim de mês, jun/2024 → jul/2026 — 26 snapshots).
- **Rótulo:** `churn = 1` se `data_referencia < data_cancelamento ≤ data_referencia + 60 dias`.
- **Scoring:** `scoring_base.parquet` contém os clientes ativos em 30/09/2026, sem rótulo — é a base que o
  dashboard pontua.

### Regras *point-in-time* (anti-vazamento)

| Fonte | Regra |
|---|---|
| Uso | Só meses ≤ mês da referência |
| Faturas | Só vencimentos ≤ referência. Atraso medido **como era visível naquele dia**: fatura paga depois da referência conta como "em aberto" |
| Chamados | Só abertos até a referência; tempo de resolução e avaliação só se o chamado já estava fechado |
| Pesquisas | Última resposta dos 12 meses anteriores à referência |
| Cadastro | `data_cancelamento`, `motivo_cancelamento` e `status` nunca viram atributos |

### Variáveis

| Grupo | Variáveis |
|---|---|
| Identificação | `customer_id`, `data_referencia`, `nome` (exibição) |
| Cliente | `idade`\*, `regiao`\*, `tipo_cliente`, `canal_aquisicao`, `tempo_cliente_meses` |
| Plano | `plano`, `funcionalidades_disponiveis`, `ciclo_contrato`, `forma_pagamento`, `renovacao_automatica`, `valor_mensal`, `reajuste_pct_6m` |
| Uso | `acessos_m1`…`acessos_m6` (m1 = mês da referência), `sessoes_m1`, `minutos_uso_m1`, `minutos_uso_m2`, `funcionalidades_usadas_m1`, `dias_ativos_m1`, `dias_desde_ultimo_acesso` |
| Financeiro | `faturas_6m`, `faturas_atrasadas_6m`, `dias_atraso_medio_6m`, `faturas_em_aberto`, `valor_total_pago`, `qtd_cobrancas_total` |
| Suporte | `tickets_3m`, `tickets_6m`, `tickets_tecnicos_6m`, `reclamacoes_6m`, `tempo_medio_resolucao_h_6m`, `avaliacao_media_6m`, `dias_desde_ultimo_chamado` |
| Satisfação | `ultima_nota_satisfacao` |
| Alvo | `churn` |

\* mantidas apenas para auditoria de viés; não entram no modelo.

## 4. Limitações

1. **Dados sintéticos.** As relações foram desenhadas a partir de padrões conhecidos de retenção; os resultados
   demonstram o *método*, não descobertas sobre um mercado real. Métricas em dados reais tendem a ser diferentes.
2. **Causalidade embutida é simplificada.** O gerador tem poucas causas de churn; empresas reais têm muitas
   (mudança de orçamento, fusões, sazonalidade setorial).
3. **Granularidade mensal.** Sem eventos diários (cliques, uso por funcionalidade), sinais muito rápidos se perdem.
4. **Sem texto livre.** Chamados reais têm descrição textual, uma fonte rica de sinal (NLP) não simulada aqui.
5. **Observações correlacionadas.** O mesmo cliente aparece em vários snapshots; a validação precisa considerar isso
   (divisão temporal, e não aleatória).
6. **Rótulo com horizonte fixo (60 dias).** Mudar o horizonte muda o problema e exige retreino.
