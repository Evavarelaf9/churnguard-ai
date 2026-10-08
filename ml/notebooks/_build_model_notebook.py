"""Gera e executa o notebook 02_modelagem.ipynb.

O treino completo roda em `python -m ml.src.training.train` (≈8 min). O notebook
lê os artefatos produzidos, mostra os resultados e demonstra a inferência,
a explicação individual e a simulação de retenção.

Uso (na raiz do projeto):
    python ml/notebooks/_build_model_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
    md("""# ChurnGuard AI · 02 — Modelagem, avaliação e explicabilidade

Este notebook apresenta os resultados do pipeline de treino (`ml/src/training/train.py`) e demonstra o uso do
modelo. Relatório completo: `docs/model.md`."""),
    code("""import sys, json
from pathlib import Path
ROOT = Path.cwd().resolve().parents[1] if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

from ml.src import config
from ml.src.scoring import ChurnScorer, simulate
from ml.src.features.build_features import add_features, MODEL_FEATURES
from ml.src.viz import style as S
S.apply()
pd.set_option("display.max_columns", 50)
results = json.loads((config.REPORTS_DIR / "training_results.json").read_text(encoding="utf-8"))
meta = json.loads(config.MODEL_METADATA_FILE.read_text(encoding="utf-8"))
FIG = config.MODEL_FIG_DIR"""),
    md("## 1. Engenharia de atributos\nExemplo dos atributos derivados para alguns clientes (todos calculados por linha, sem estado):"),
    code("""base = pd.read_parquet(config.ANALYTICAL_BASE_FILE)
cols = ["media_acessos_semana", "variacao_uso", "tendencia_uso", "meses_consecutivos_queda",
        "taxa_utilizacao_funcionalidades", "percentual_pagamentos_atrasados", "aceleracao_tickets",
        "meses_ate_renovacao", "churn"]
add_features(base.sample(6, random_state=3))[cols].round(3)"""),
    md("## 2. Divisão temporal e validação cruzada"),
    code("""print(json.dumps(results["split"], indent=2, ensure_ascii=False))"""),
    md("## 3. Modelos: inicial × ajustado"),
    code("""rows = []
for k, v in results["tuned"].items():
    rows.append({"modelo": v["name"],
                 "CV PR-AUC inicial": results["baseline"][k]["cv"]["pr_auc"], "CV PR-AUC ajustado": v["cv"]["pr_auc"],
                 "teste ROC-AUC": v["test"]["roc_auc"], "teste PR-AUC": v["test"]["pr_auc"], "teste Brier": v["test"]["brier"]})
display(pd.DataFrame(rows).round(4))
display(Image(FIG / "01_comparacao_modelos.png"))
display(Image(FIG / "02_curvas_roc_pr.png"))"""),
    code("""pd.DataFrame([{"modelo": v["name"], **v["best_params"]} for v in results["tuned"].values()])"""),
    md("## 4. Desbalanceamento\nReamostragem e pesos deslocam as probabilidades sem melhorar a ordenação — e destroem a calibração."),
    code("""imb = pd.DataFrame(results["imbalance"])[["modelo", "estrategia", "pr_auc", "roc_auc", "recall", "precision", "prob_media", "brier"]]
display(imb.round(3))
display(pd.DataFrame(results["imbalance_tuned"])[["estrategia", "pr_auc", "brier", "prob_media"]].round(4))
print("Estratégia adotada:", results["imbalance_strategy"])
display(Image(FIG / "03_desbalanceamento.png"))"""),
    md("## 5. Threshold\nEscolhido com previsões *out-of-fold* do treino, maximizando o resultado esperado da campanha."),
    code("""print("Threshold operacional:", meta["threshold"], "\\nFaixas de risco:", meta["risk_thresholds"])
pd.DataFrame(results["threshold_test_table"]).round(3)"""),
    code("""display(Image(FIG / "05_threshold.png")); display(Image(FIG / "06_valor_negocio.png")); display(Image(FIG / "04_matriz_confusao.png"))"""),
    md("## 6. Níveis de risco, lift e calibração"),
    code("""display(pd.DataFrame(results["risk_summary_test"]))
display(Image(FIG / "09_niveis_risco.png")); display(Image(FIG / "08_ganho_lift.png")); display(Image(FIG / "07_calibracao.png"))"""),
    md("## 7. Explicabilidade global"),
    code("""display(pd.DataFrame(results["permutation_importance"]).head(12)[["label", "importance", "std"]].round(4))
display(Image(FIG / "10_importancia_permutacao.png")); display(Image(FIG / "11_shap_resumo.png"))"""),
    md("## 8. Usando o modelo: previsão, explicação individual e threshold configurável"),
    code("""scorer = ChurnScorer.load()
clientes = pd.read_parquet(config.SCORING_BASE_FILE)
pred = scorer.predict(clientes)
clientes = clientes.join(pred)
clientes.sort_values("probabilidade", ascending=False)[["customer_id", "nome", "plano", "valor_mensal", "probabilidade", "nivel_risco"]].head(10)"""),
    code("""alvo = clientes.sort_values("probabilidade", ascending=False).iloc[[3]]
exp = scorer.explain(alvo)[0]
print(f"Cliente {alvo.customer_id.iloc[0]} — probabilidade estimada de cancelamento: {exp['probabilidade']:.0%} ({exp['nivel_risco']} risco)")
print("\\nFatores que aumentam o risco:")
for f in exp["fatores_risco"]: print("  •", f["text"])
print("\\nFatores que reduzem o risco:")
for f in exp["fatores_protecao"]: print("  •", f["text"])
print("\\nSugestão:", exp["recomendacao"]["acao_sugerida"])
print(exp["recomendacao"]["prioridade"])
print("⚠", exp["recomendacao"]["aviso"])"""),
    code("""for t in (0.3, 0.5, 0.7, meta["threshold"]["operacional"]):
    n = scorer.predict(clientes, threshold=t).acima_do_threshold.sum()
    print(f"threshold {t:.2f} → {n:>4} clientes sinalizados para abordagem")"""),
    md("## 9. Simulação de retenção\n> **Simulação hipotética**: mostra como o modelo reagiria se os sinais fossem diferentes. Não é uma previsão garantida do comportamento do cliente."),
    code("""c = alvo.drop(columns=pred.columns)
cenarios = {
    "Uso volta ao nível de 3 meses atrás": {"acessos_m1": float(c.acessos_m4.iloc[0]), "acessos_m2": float(c.acessos_m5.iloc[0]),
                                            "dias_ativos_m1": 22},
    "Sem atrasos de pagamento": {"faturas_atrasadas_6m": 0, "dias_atraso_medio_6m": 0, "faturas_em_aberto": 0},
    "Sem reclamações": {"reclamacoes_6m": 0},
    "Migra para débito automático": {"forma_pagamento": "Débito automático", "renovacao_automatica": 1},
}
pd.DataFrame([{"cenário": k, **simulate(scorer, c, v)} for k, v in cenarios.items()]).drop(columns="aviso").round(3)"""),
    md("""## 10. Conclusões

- O **Gradient Boosting ajustado** foi o melhor em PR-AUC na validação cruzada e no teste, com probabilidades calibradas.
- Desbalanceamento tratado **no threshold**, não nos dados, preservando probabilidades interpretáveis.
- Threshold operacional escolhido por valor de negócio, com premissas explícitas e configuráveis.
- Explicações individuais em linguagem de negócio + sugestão de ação, sempre como **apoio à decisão humana**."""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, HERE / "02_modelagem.ipynb")
print("Notebook executado e salvo em", HERE / "02_modelagem.ipynb")
