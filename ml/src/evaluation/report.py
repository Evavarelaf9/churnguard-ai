"""Gera as figuras do modelo (docs/images/model) e o relatório docs/model.md.

Lê apenas artefatos produzidos por `ml.src.training.train` — nenhum número é
digitado à mão.

Uso:
    python -m ml.src.evaluation.report
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from ml.src import config
from ml.src.evaluation import plots as P
from ml.src.features.build_features import FEATURE_LABELS
from ml.src.viz import style as S


def br(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def money(v: float) -> str:
    return f"R$ {v:,.0f}".replace(",", ".")


def main() -> None:
    S.apply()
    out = config.MODEL_FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    r = json.loads((config.REPORTS_DIR / "training_results.json").read_text(encoding="utf-8"))
    meta = json.loads(config.MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    scores = pd.read_parquet(config.REPORTS_DIR / "test_scores_models.parquet")
    pred = pd.read_parquet(config.TEST_PREDICTIONS_FILE)
    shap_df = pd.read_parquet(config.REPORTS_DIR / "shap_sample.parquet")
    shap_feat = pd.read_parquet(config.REPORTS_DIR / "shap_sample_features.parquet")
    scored = pd.read_parquet(config.SCORED_CUSTOMERS_FILE)
    sel = r["selected"]
    t_op, t_med = r["risk_thresholds"]["alto"], r["risk_thresholds"]["medio"]
    figs = {}

    def save(fig, name):
        fig.savefig(out / f"{name}.png")
        plt.close(fig)
        figs[name] = f"images/model/{name}.png"

    save(P.model_comparison(r), "01_comparacao_modelos")
    save(P.roc_pr_curves(scores, sel), "02_curvas_roc_pr")
    save(P.imbalance_plot(r["imbalance"], r["tuned"][sel]["name"]), "03_desbalanceamento")
    save(P.confusion_matrices(pred.churn.to_numpy(), pred.probabilidade.to_numpy(),
                              [(0.5, "Threshold padrão (0,50)"), (t_op, f"Threshold operacional ({br(t_op, 2)})")]),
         "04_matriz_confusao")
    oof_table = pd.DataFrame(r["threshold_oof_table"])
    save(P.threshold_plot(oof_table, t_op, t_med), "05_threshold")
    save(P.business_value_plot(oof_table, t_op, meta["business_assumptions"]), "06_valor_negocio")
    save(P.calibration_plot(pred.churn.to_numpy(), pred.probabilidade.to_numpy()), "07_calibracao")
    save(P.gains_plot(r["lift_test"]), "08_ganho_lift")
    save(P.risk_groups_plot(r["risk_summary_test"], r["scoring"], r["risk_thresholds"]), "09_niveis_risco")
    save(P.importance_plot(r["permutation_importance"]), "10_importancia_permutacao")
    save(P.shap_summary_plot(shap_df, shap_feat, FEATURE_LABELS), "11_shap_resumo")
    save(P.lr_coefficients_plot(r["logreg_coefficients"], FEATURE_LABELS), "12_coeficientes_logistica")
    top = scored.sort_values("probabilidade", ascending=False).iloc[3]
    example = json.loads(top.explicacao_json)
    save(P.individual_explanation_plot(example, top.customer_id), "13_explicacao_individual")
    save(P.predicted_vs_actual_plot(pred), "14_previsto_vs_real")

    write_report(r, meta, figs, example, top, scored)
    print(f"Relatório do modelo: docs/model.md ({len(figs)} figuras em docs/images/model)")


def _table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def write_report(r, meta, figs, example, top, scored) -> None:
    img = lambda k, alt: f"![{alt}]({figs[k]})"
    sel = r["selected"]
    best = r["tuned"][sel]
    t_op, t_med = r["risk_thresholds"]["alto"], r["risk_thresholds"]["medio"]
    tm, t05 = r["test_metrics_operational"], r["test_metrics_05"]
    ev = r["test_expected_value"]
    sp = r["split"]
    biz = meta["business_assumptions"]

    comp = pd.DataFrame([{
        "Modelo": v["name"],
        "CV PR-AUC (inicial → ajustado)": f"{br(r['baseline'][k]['cv']['pr_auc'])} → **{br(v['cv']['pr_auc'])}** ± {br(v['cv']['pr_auc_std'])}",
        "CV ROC-AUC": br(v["cv"]["roc_auc"]),
        "Teste PR-AUC": br(v["test"]["pr_auc"]),
        "Teste ROC-AUC": br(v["test"]["roc_auc"]),
        "Brier (teste)": br(v["test"]["brier"], 4),
    } for k, v in r["tuned"].items()])

    params = pd.DataFrame([{"Modelo": v["name"], "Hiperparâmetros escolhidos": ", ".join(
        f"`{k}={br(x, 4) if isinstance(x, float) else x}`" for k, x in v["best_params"].items())} for v in r["tuned"].values()])

    imb = pd.DataFrame(r["imbalance"])
    imb_t = imb.assign(**{
        "PR-AUC": imb.pr_auc.map(br), "ROC-AUC": imb.roc_auc.map(br),
        "Recall @0,5": imb.recall.map(lambda v: S.pct(v)), "Precision @0,5": imb.precision.map(lambda v: S.pct(v)),
        "Prob. média": imb.prob_media.map(lambda v: S.pct(v)), "Brier": imb.brier.map(lambda v: br(v, 4)),
    })[["modelo", "estrategia", "PR-AUC", "ROC-AUC", "Recall @0,5", "Precision @0,5", "Prob. média", "Brier"]]
    imb_t.columns = ["Modelo", "Estratégia", "PR-AUC", "ROC-AUC", "Recall @0,5", "Precision @0,5", "Prob. média", "Brier"]
    imb_tuned = pd.DataFrame(r["imbalance_tuned"])
    imb_tuned_t = pd.DataFrame({
        "Estratégia (modelo ajustado)": imb_tuned.estrategia,
        "PR-AUC": imb_tuned.pr_auc.map(br), "Brier": imb_tuned.brier.map(lambda v: br(v, 4)),
        "Prob. média": imb_tuned.prob_media.map(lambda v: S.pct(v)),
    })

    thr = pd.DataFrame(r["threshold_test_table"]).sort_values("threshold")
    thr_t = pd.DataFrame({
        "Threshold": thr.threshold.map(lambda v: f"**{br(v, 2)}**" if abs(v - t_op) < 1e-9 else br(v, 2)),
        "Precision": thr.precision.map(lambda v: S.pct(v)), "Recall": thr.recall.map(lambda v: S.pct(v)),
        "F1": thr.f1.map(lambda v: br(v)), "Clientes abordados": thr.abordados.map(S.num),
        "% da base": thr.percentual_abordados.map(lambda v: S.pct(v)),
        "Churns identificados": [f"{S.num(a)} de {S.num(b)}" for a, b in zip(thr.churns_identificados, thr.churns_totais)],
        "Falsos positivos": thr.fp.map(S.num), "Resultado esperado": thr.resultado_liquido.map(money),
    })

    risk = pd.DataFrame(r["risk_summary_test"])
    risk_t = pd.DataFrame({"Nível": risk.risco, "Clientes": risk.clientes.map(S.num), "Churns": risk.churns.map(S.num),
                           "Taxa real de churn": risk.taxa_real.map(lambda v: S.pct(v))})

    perm = pd.DataFrame(r["permutation_importance"]).head(10)
    shap_g = {d["feature"]: d for d in r["shap_global"]}
    lift = pd.DataFrame(r["lift_test"])
    top_dec = lift.iloc[0]
    capture20 = lift.captura_acumulada.iloc[1]

    def direction(f):
        d = shap_g.get(f, {}).get("direction")
        if d is None:
            return "depende da categoria"
        return "↑ valor alto aumenta o risco" if d > 0.2 else ("↓ valor alto reduz o risco" if d < -0.2 else "efeito não linear")

    perm_t = pd.DataFrame({"#": range(1, len(perm) + 1), "Fator": perm.label,
                           "Importância (queda na PR-AUC)": perm.importance.map(lambda v: br(v, 4)),
                           "Direção (SHAP)": perm.feature.map(direction)})

    ex_factors = "\n".join(f"- {f['text']}" for f in example["fatores_risco"])
    ex_protect = "\n".join(f"- {f['text']}" for f in example["fatores_protecao"]) or "- —"
    sc = r["scoring"]
    cat_counts = scored[scored.nivel_risco == "Alto"].categoria_principal.value_counts()
    cat_names = {"uso": "utilização", "financeiro": "financeiro", "suporte": "suporte e satisfação",
                 "contrato": "contrato", "preco": "preço", "perfil": "perfil"}
    cat_line = ", ".join(f"{cat_names.get(k, k)} ({v})" for k, v in cat_counts.items())

    md = f"""# Documentação do Modelo — ChurnGuard AI

> Gerado automaticamente por `python -m ml.src.evaluation.report` a partir dos artefatos de
> `python -m ml.src.training.train`. Versão do modelo: **{meta['model_version']}** · treinado em {meta['trained_at']}.

## Resumo executivo

| Item | Valor |
|---|---|
| Pergunta | Probabilidade de o cliente **cancelar nos próximos {meta['horizon_days']} dias** |
| Modelo escolhido | **{best['name']}** |
| Estratégia de desbalanceamento | **{r['imbalance_strategy']}** (probabilidades calibradas + threshold ajustado) |
| ROC-AUC / PR-AUC no teste | **{br(tm['roc_auc'])}** / **{br(tm['pr_auc'])}** (taxa base {S.pct(tm['base_rate'])}) |
| Threshold operacional | **{br(t_op, 2)}** (escolhido no treino, maximizando o resultado esperado) |
| No teste, com esse threshold | precision **{S.pct(tm['precision'])}** · recall **{S.pct(tm['recall'])}** · F1 {br(tm['f1'])} |
| 10% de maior risco | churn real de **{S.pct(top_dec.taxa_real)}** (**{br(top_dec.lift, 1)}×** a média) |
| Abordando os 20% de maior risco | encontra **{S.pct(capture20, 0)}** dos cancelamentos |
| Níveis de risco | Baixo < {br(t_med, 2)} ≤ Médio < {br(t_op, 2)} ≤ Alto |
| Clientes ativos hoje | {S.num(sc['clientes'])} · alto risco: **{S.num(sc['distribuicao_risco']['Alto'])}** · médio: {S.num(sc['distribuicao_risco']['Médio'])} |
| Receita mensal em alto risco | **{money(sc['receita_mensal_alto_risco'])}** |

---

## 1. Engenharia de atributos

Código: `ml/src/features/build_features.py`. Todas as transformações são **por linha e sem estado**
(usam apenas informações do próprio cliente naquela data de referência); por isso não há como vazar
informação entre treino e teste. Tudo o que depende da distribuição dos dados (limites de outlier,
medianas, padronização) é aprendido no pipeline, **só no treino**.

| Atributo | Como é calculado | Por que ajuda |
|---|---|---|
| `media_acessos_semana` | acessos do mês ÷ 4,33 | frequência em unidade intuitiva para o negócio |
| `taxa_utilizacao_funcionalidades` | funcionalidades usadas ÷ disponíveis no plano | normaliza o uso pela capacidade do plano (Empresarial tem 20 módulos, Básico 5) |
| `taxa_dias_ativos` | dias ativos ÷ 30 | hábito de uso, mais estável que contagem de acessos |
| `minutos_por_sessao` | minutos ÷ sessões | profundidade do uso |
| **`variacao_uso`** | média de acessos (3 últimos meses) ÷ média (3 anteriores) − 1 | **mudança de comportamento** — sinal central da EDA |
| **`variacao_uso_ultimo_mes`** | acessos do mês ÷ mês anterior − 1 | queda abrupta recente |
| **`tendencia_uso`** | inclinação da reta de acessos em 6 meses ÷ média | tendência de médio prazo, robusta a um mês atípico |
| **`meses_consecutivos_queda`** | meses seguidos com uso menor que o anterior | persistência da queda |
| `variacao_minutos` | minutos do mês ÷ mês anterior − 1 | queda de intensidade mesmo com acessos estáveis |
| `percentual_pagamentos_atrasados` | faturas atrasadas ÷ faturas (6 meses) | atrito financeiro relativo, comparável entre clientes novos e antigos |
| `tem_fatura_em_aberto` | fatura vencida há mais de 5 dias sem pagamento | risco de churn involuntário |
| `teve_reajuste_recente` / `reajuste_pct_6m` | variação da mensalidade em 6 meses | sensibilidade a preço (picos de churn pós-janeiro) |
| `relacao_tickets_por_mes` | chamados (6m) ÷ meses observados | volume de suporte normalizado pelo tempo de casa |
| **`aceleracao_tickets`** | chamados (3 últimos meses) − chamados (3 anteriores) | **aumento** de atrito com o suporte |
| `proporcao_reclamacoes` | reclamações ÷ chamados | qualidade da relação, não só volume |
| `tempo_desde_ultimo_chamado` | dias desde o último chamado (365 se nunca abriu) | recência do atrito |
| `nunca_abriu_chamado`, `sem_avaliacao_suporte`, `sem_pesquisa_satisfacao`, `historico_curto` | *flags* de ausência | ausência carrega informação (pouco relacionamento) |
| `cliente_novo` | até 3 meses de casa | janela de onboarding |
| **`meses_ate_renovacao`** / `renovacao_proxima` | meses até o aniversário do contrato anual (mensal = 1) | captura o "penhasco" de renovação visto na EDA |

**Atributos excluídos** e motivo:

{_table(pd.DataFrame({"Coluna": list(meta['features']['excluded']), "Motivo": list(meta['features']['excluded'].values())}))}
| `data_cancelamento`, `motivo_cancelamento`, `status` | informação posterior ao evento — **vazamento** (não chegam à base analítica) |

## 2. Preparação dos dados

Código: `ml/src/preprocessing/pipeline.py`. Um único objeto `Pipeline` do scikit-learn, salvo com `joblib`,
é usado no treino, na avaliação e na API:

```text
FeatureEngineer (sem estado, seleciona só as colunas do modelo)
  → ColumnTransformer
       numéricas  : Winsorizer (p0,5–p99,5 do treino) → SimpleImputer(mediana) → StandardScaler*
       binárias   : SimpleImputer(moda)
       categóricas: SimpleImputer("Não informado") → OneHotEncoder(handle_unknown="ignore")
  → Modelo
* padronização só para a Regressão Logística (árvores não precisam)
```

### Divisão temporal

Os dados têm dinâmica temporal (picos pós-reajuste) e o mesmo cliente aparece em vários meses. Uma divisão
aleatória colocaria o "futuro" de um cliente no treino e o "passado" no teste. Por isso:

| Conjunto | Datas de referência | Linhas | Churn |
|---|---|---:|---:|
| Treino | {sp['train_start']} → {sp['train_end']} | {S.num(sp['train_rows'])} | {S.pct(sp['train_churn_rate'])} |
| *Intervalo* | out/2025 e nov/2025 descartados | — | — |
| Teste | {sp['test_start']} → {sp['test_end']} | {S.num(sp['test_rows'])} | {S.pct(sp['test_churn_rate'])} |

O intervalo de 2 meses existe porque o rótulo olha {meta['horizon_days']} dias à frente: o rótulo do último
mês de treino depende de eventos até o fim de novembro/2025.

**Validação cruzada temporal** (`TemporalGroupSplit`), usada no ajuste de hiperparâmetros, no estudo de
desbalanceamento e na escolha do threshold — sempre **dentro do treino**:

{_table(pd.DataFrame(sp['cv_folds']).rename(columns={'treino_ate': 'Treina até', 'validacao': 'Valida em'}).assign(Dobra=[1, 2, 3])[['Dobra', 'Treina até', 'Valida em']])}

## 3. Métricas: por que não acurácia?

Com {S.pct(sp['test_churn_rate'])} de churn no teste, um "modelo" que responde sempre *"não cancela"* teria
**{S.pct(1 - sp['test_churn_rate'])} de acurácia** — e não encontraria nenhum cliente em risco. Com o threshold
padrão de 0,50, o nosso modelo tem acurácia de {S.pct(t05['accuracy'])}, mas recall de apenas {S.pct(t05['recall'])}.

| Métrica | O que responde | Uso no projeto |
|---|---|---|
| **PR-AUC** | Quão bem o modelo acha churns sem gerar alarmes falsos, em todos os thresholds | **Métrica principal** de seleção |
| ROC-AUC | Quão bem o modelo ordena clientes por risco | Complementar |
| **Recall (churn)** | Dos clientes que cancelaram, quantos o modelo sinalizou? | Churn não detectado = retenção perdida |
| **Precision (churn)** | Dos clientes sinalizados, quantos cancelaram de fato? | Falso positivo = recurso de retenção desperdiçado |
| F1 | Equilíbrio entre os dois | Referência estatística do threshold |
| Brier score | Quão realistas são as probabilidades | Essencial, pois o produto **mostra** a probabilidade |

## 4. Modelos testados

{_table(comp)}

{img('01_comparacao_modelos', 'Comparação de modelos')}
{img('02_curvas_roc_pr', 'Curvas ROC e PR')}

### Hiperparâmetros (RandomizedSearchCV, CV temporal, otimizando PR-AUC)

{_table(params)}

O que mudou e por quê:
- **Gradient Boosting**: árvores mais rasas (`max_leaf_nodes` pequeno), `min_samples_leaf` e regularização L2
  maiores e amostragem de atributos (`max_features`). Com sinal fraco e ruidoso (churn), modelos mais simples por
  árvore e mais regularizados generalizam melhor para o período seguinte.
- **Random Forest**: `min_samples_leaf` alto suaviza as probabilidades das folhas (menos sobreajuste a clientes
  específicos que aparecem em vários snapshots).
- **Regressão Logística**: regularização (L1/L2) quase não muda o resultado — o limite é a própria forma
  linear, que não captura interações (ex.: *contrato anual × meses até a renovação*).

### Modelo escolhido: {best['name']}

- Maior PR-AUC média na validação cruzada ({br(best['cv']['pr_auc'])} ± {br(best['cv']['pr_auc_std'])}) e também no teste.
- Melhor calibração (menor Brier), importante porque o dashboard exibe a probabilidade.
- Captura interações e não-linearidades (o efeito da renovação só existe para contratos anuais).
- Rápido para treinar e pontuar; nativo do scikit-learn (sem dependência extra em produção).
- A explicabilidade é garantida com SHAP (`TreeExplainer`, exato para árvores).
- A Regressão Logística fica como **baseline interpretável** e confirma a direção dos efeitos (seção 8).

## 5. Desbalanceamento de classes

Proporção de ~{round((1 - sp['train_churn_rate']) / sp['train_churn_rate'])}:1 no treino. As estratégias foram
comparadas **dentro de cada dobra** (o `Pipeline` do imbalanced-learn só reamostra no `fit`; validação e teste
nunca são reamostrados).

{_table(imb_t)}

{img('03_desbalanceamento', 'Estratégias de desbalanceamento')}

Verificação no modelo já ajustado:

{_table(imb_tuned_t)}

**Decisão: {r['imbalance_strategy']}.** O que os resultados mostram:
1. Nenhuma estratégia melhora a **ordenação** (PR-AUC/ROC-AUC) de forma consistente. O ganho do SMOTE no modelo
   com parâmetros padrão desaparece depois do ajuste de hiperparâmetros.
2. `class_weight` e reamostragem apenas **deslocam** as probabilidades para cima (probabilidade média de ~8% vai a
   25–40%), o que "melhora" o recall em 0,50 — mas o mesmo efeito se obtém baixando o threshold.
3. O custo é perder a **calibração** (Brier dobra): um cliente mostrado com "40%" teria, na realidade, ~10% de
   chance. Como o produto exibe a probabilidade para pessoas tomarem decisões, isso é inaceitável.

→ Tratamos o desbalanceamento na **decisão** (threshold), não nos dados.

## 6. Escolha do threshold

O threshold foi escolhido com previsões *out-of-fold* do **treino** (o teste não participa da escolha), pelo
critério de **maior resultado líquido esperado** da campanha de retenção:

> resultado = (churns abordados × mensalidade × {biz['meses_receita_preservada']} meses × {S.pct(biz['taxa_sucesso_retencao'], 0)} de sucesso) − (clientes abordados × R$ {biz['custo_acao']:.0f})

As premissas estão em `ml/src/config.py` (`BUSINESS`) e devem ser ajustadas à realidade de cada empresa.
Threshold escolhido: **{br(t_op, 2)}** (o que maximiza F1 seria {br(meta['threshold']['f1_otimo'], 2)}).

{img('05_threshold', 'Análise de threshold')}
{img('06_valor_negocio', 'Resultado esperado por threshold')}

### Efeito do threshold no período de teste

{_table(thr_t)}

Leitura:
- **0,30** → aborda {S.pct(thr[thr.threshold.round(2) == 0.3].percentual_abordados.iloc[0])} da base e encontra {S.pct(thr[thr.threshold.round(2) == 0.3].recall.iloc[0])} dos churns.
- **0,50** → quase ninguém é abordado: precision alta, mas {S.pct(1 - thr[thr.threshold.round(2) == 0.5].recall.iloc[0])} dos churns passam despercebidos.
- **0,70** → conservador demais para este problema.
- **{br(t_op, 2)}** (operacional) → equilíbrio entre esforço da equipe e cancelamentos evitados. O threshold é
  configurável na API e no dashboard.

{img('04_matriz_confusao', 'Matriz de confusão')}

## 7. Níveis de risco

| Nível | Regra | Uso sugerido |
|---|---|---|
| 🟢 **Baixo** | probabilidade < {br(t_med, 2)} | sem ação; monitoramento |
| 🟡 **Médio** | {br(t_med, 2)} ≤ probabilidade < {br(t_op, 2)} | ações de engajamento em escala (e-mail, conteúdo) |
| 🔴 **Alto** | probabilidade ≥ {br(t_op, 2)} (threshold operacional) | contato humano prioritário |

O limite de **médio** é 1,5× a taxa média de churn do treino (arredondado): acima dele, o cliente já tem risco
claramente acima da média, mas ainda não justifica o custo de contato individual. Ambos os limites estão
salvos em `ml/models/model_metadata.json` e são configuráveis.

{_table(risk_t)}

{img('09_niveis_risco', 'Níveis de risco')}
{img('08_ganho_lift', 'Lift e ganho acumulado')}
{img('07_calibracao', 'Calibração')}

## 8. Explicabilidade

### Global — o que mais influencia o risco

{_table(perm_t)}

{img('10_importancia_permutacao', 'Importância por permutação')}
{img('11_shap_resumo', 'Resumo SHAP')}

Leitura para o negócio:
- **Janela de renovação** é o fator mais forte: clientes mensais podem sair a qualquer momento, e clientes
  anuais concentram a saída perto do aniversário do contrato. *Ação: conversa de renovação antecipada.*
- **Hábito de uso** (dias ativos, funcionalidades usadas, frequência) vem logo depois, e **queda recente de uso**
  aparece entre os principais sinais. *Clientes não cancelam do nada; o uso cai antes.*
- **Atrito** — reclamações, atraso de pagamento, boleto sem renovação automática, nota de satisfação baixa — soma risco.
- **Correlação não é causalidade:** o modelo aprende associações. "Boleto" não *causa* churn; ele pode indicar
  menor compromisso ou mais fricção de pagamento. Recomendações devem ser testadas (ex.: teste A/B).

{img('12_coeficientes_logistica', 'Coeficientes da regressão logística')}

### Individual — por cliente

Cada previsão traz os principais fatores **em linguagem de negócio**, calculados com SHAP e convertidos por
templates (`ml/src/explainability/explainer.py`). Fatores estruturais (perfil, canal, contrato mensal) continuam
visíveis, mas o **principal fator** e a **recomendação** priorizam o que a empresa pode influenciar.

**Exemplo real da base atual — cliente `{top.customer_id}`**

| | |
|---|---|
| Probabilidade estimada de cancelamento (60 dias) | **{S.pct(example['probabilidade'], 0)}** |
| Classificação | **{example['nivel_risco']} risco** |
| Principal fator | {example['principal_fator']} |

Fatores que aumentam o risco:
{ex_factors}

Fatores que reduzem o risco:
{ex_protect}

**Sugestão:** *{example['recomendacao']['acao_sugerida']}*

{example['recomendacao']['prioridade']}

> {example['recomendacao']['aviso']}

{img('13_explicacao_individual', 'Explicação individual')}

Categoria principal entre os clientes em **alto risco** hoje: {cat_line}.

## 9. Resultado esperado no período de teste

Com o threshold operacional, a campanha teria abordado **{S.num(ev['abordados'])}** clientes-mês
({S.pct(ev['percentual_abordados'])} das observações), encontrado **{S.num(ev['churns_identificados'])}** dos
{S.num(ev['churns_totais'])} cancelamentos e gerado resultado líquido esperado de **{money(ev['resultado_liquido'])}**
(receita preservada {money(ev['receita_preservada_esperada'])} − custo {money(ev['custo_campanha'])}), sob as
premissas acima.

{img('14_previsto_vs_real', 'Previsto versus real')}

## 10. Modelo de produção

Depois da avaliação, o mesmo pipeline (mesmos hiperparâmetros e thresholds) é **re-treinado com todos os
snapshots rotulados** ({meta['training_data']['production_fit']}, {S.num(meta['training_data']['production_rows'])} linhas)
para aproveitar os dados mais recentes. Os números de desempenho reportados são os do teste (modelo treinado só
até set/2025), que é a estimativa honesta do comportamento no futuro.

| Artefato | Conteúdo |
|---|---|
| `ml/models/churn_model.joblib` | pipeline de produção + metadados + amostra de referência |
| `ml/models/churn_model_eval.joblib` | pipeline avaliado no teste (auditoria) |
| `ml/models/model_metadata.json` | versão, parâmetros, thresholds, métricas, premissas |
| `ml/data/processed/scored_customers.parquet` | clientes ativos com probabilidade, risco, fatores e sugestão |
| `ml/data/processed/test_predictions.parquet` | previsto × real no teste (base do monitoramento) |

## 11. Limitações

- Dados **sintéticos**: os números demonstram o método; em dados reais o desempenho será diferente.
- PR-AUC de {br(tm['pr_auc'])} significa que **a maioria dos sinalizados não cancelaria** — o modelo prioriza, não decide.
- Premissas financeiras (custo, sucesso, meses preservados) são hipóteses; o threshold deve ser revisto com dados reais de campanha.
- O modelo aprende **correlações**; efeito causal das ações deve ser medido com grupo de controle.
- Relações mudam com o tempo (novo reajuste, mudança no suporte) → monitoramento e re-treino (ver `docs/monitoring.md`).
"""
    (config.DOCS_DIR / "model.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
