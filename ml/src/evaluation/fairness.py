"""Auditoria de viés: o modelo trata grupos de forma parecida?

Idade e região NÃO são entradas do modelo (ver docs/lgpd.md). Mesmo assim, outras
variáveis podem funcionar como "proxies". Esta auditoria compara, no período de
teste, por grupo:
  - churn real × probabilidade média prevista (calibração por grupo);
  - % de clientes sinalizados como alto risco;
  - recall (quantos que cancelaram foram sinalizados) e precision.

Uso:
    python -m ml.src.evaluation.fairness
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.src import config
from ml.src.preprocessing.pipeline import DEFAULT_SPLIT, temporal_train_test_split
from ml.src.viz import style as S


def group_metrics(df: pd.DataFrame, col: str, order: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(col, observed=True):
        y, flag = d.churn.to_numpy(), d.alto.to_numpy()
        tp = int((flag & (y == 1)).sum())
        rows.append({
            "grupo": str(g), "clientes_mes": len(d), "churn_real": y.mean(), "prob_media": d.proba.mean(),
            "pct_alto_risco": flag.mean(),
            "recall": tp / max(int(y.sum()), 1), "precision": tp / max(int(flag.sum()), 1),
        })
    out = pd.DataFrame(rows).set_index("grupo")
    return out.reindex(order) if order else out


def main() -> dict:
    S.apply()
    bundle = joblib.load(config.MODELS_DIR / "churn_model_eval.joblib")
    th = bundle["metadata"]["risk_thresholds"]["alto"]
    base = pd.read_parquet(config.ANALYTICAL_BASE_FILE)
    _, test = temporal_train_test_split(base, DEFAULT_SPLIT)
    test = test.assign(proba=bundle["pipeline"].predict_proba(test)[:, 1])
    test["alto"] = test.proba >= th
    idade = test.idade.where(test.idade.between(18, 100))
    test["faixa_idade"] = pd.cut(idade, [17, 25, 35, 45, 60, 100], labels=["18–25", "26–35", "36–45", "46–60", "60+"])
    test["regiao_"] = test.regiao.fillna("Não informada")

    tables = {
        "regiao": group_metrics(test, "regiao_", ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]),
        "faixa_idade": group_metrics(test, "faixa_idade", ["18–25", "26–35", "36–45", "46–60", "60+"]),
        "tipo_cliente": group_metrics(test, "tipo_cliente", ["Pessoa Física", "Pequena Empresa", "Média Empresa"]),
    }
    summary = {}
    for k, t in tables.items():
        summary[k] = {
            "tabela": t.round(4).reset_index().to_dict("records"),
            "amplitude_recall": float(t.recall.max() - t.recall.min()),
            "amplitude_gap_calibracao": float((t.prob_media - t.churn_real).abs().max()),
            "razao_sinalizacao": float(t.pct_alto_risco.min() / t.pct_alto_risco.max()),
        }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "fairness.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, (k, title) in zip(axes, [("regiao", "Região"), ("faixa_idade", "Faixa etária"), ("tipo_cliente", "Tipo de cliente")]):
        t = tables[k]
        x = np.arange(len(t))
        ax.bar(x - 0.2, t.churn_real, width=0.38, color=S.ORANGE, label="Churn real", zorder=2)
        ax.bar(x + 0.2, t.prob_media, width=0.38, color=S.BLUE, label="Probabilidade média prevista", zorder=2)
        for xi, r in zip(x, t.recall):
            ax.text(xi, max(t.churn_real.max(), t.prob_media.max()) * 1.08, f"recall\n{S.pct(r, 0)}", ha="center", fontsize=8, color=S.INK_2)
        ax.set_xticks(x, t.index, fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.set_ylim(0, max(t.churn_real.max(), t.prob_media.max()) * 1.35)
        S.title(ax, title, "Período de teste · limite de alto risco " + f"{th:.2f}".replace(".", ","))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(1, 1.07))
    fig.tight_layout(w_pad=2.5)
    out = config.DOCS_DIR / "images" / "fairness"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "auditoria_vies.png")
    plt.close(fig)

    for k, t in tables.items():
        print(f"\n{k}\n", t.round(3).to_string())
    return summary


if __name__ == "__main__":
    main()
