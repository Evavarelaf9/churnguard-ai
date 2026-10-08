"""Métricas de avaliação, análise de threshold e impacto de negócio."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.src import config


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def expected_value(y_true, proba, monthly_value, threshold: float, business: dict | None = None) -> dict:
    """Resultado financeiro esperado de abordar os clientes acima do threshold.

    Premissas (configuráveis em config.BUSINESS):
      - cada cliente abordado custa `custo_acao` (contato + incentivo);
      - uma abordagem a quem iria cancelar retém o cliente com probabilidade
        `taxa_sucesso_retencao`;
      - cliente retido preserva `meses_receita_preservada` meses de mensalidade.
    Falsos positivos geram só custo; falsos negativos são receita perdida sem tentativa.
    """
    b = business or config.BUSINESS
    y = np.asarray(y_true)
    v = np.asarray(monthly_value, dtype=float)
    act = np.asarray(proba) >= threshold
    saved = (y == 1) & act
    revenue_saved = (v[saved] * b["meses_receita_preservada"] * b["taxa_sucesso_retencao"]).sum()
    cost = act.sum() * b["custo_acao"]
    return {
        "abordados": int(act.sum()),
        "percentual_abordados": float(act.mean()),
        "churns_identificados": int(saved.sum()),
        "churns_totais": int(y.sum()),
        "receita_preservada_esperada": float(revenue_saved),
        "custo_campanha": float(cost),
        "resultado_liquido": float(revenue_saved - cost),
    }


def threshold_table(y_true, proba, monthly_value, thresholds=None) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.05, 0.81, 0.05), 2)
    rows = []
    for t in thresholds:
        m = classification_metrics(y_true, proba, t)
        ev = expected_value(y_true, proba, monthly_value, t)
        rows.append({
            "threshold": t, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "abordados": ev["abordados"], "percentual_abordados": ev["percentual_abordados"],
            "churns_identificados": ev["churns_identificados"], "churns_totais": ev["churns_totais"],
            "fp": m["fp"], "fn": m["fn"], "resultado_liquido": ev["resultado_liquido"],
        })
    return pd.DataFrame(rows)


def choose_threshold(y_true, proba, monthly_value) -> dict:
    """Escolhe o threshold operacional em uma grade fina.

    Critério principal: maior resultado líquido esperado (visão de negócio).
    Também reporta o threshold que maximiza F1 (visão estatística) para comparação.
    """
    grid = np.round(np.arange(0.03, 0.80, 0.01), 2)
    t = threshold_table(y_true, proba, monthly_value, grid)
    best_value = t.loc[t.resultado_liquido.idxmax()]
    best_f1 = t.loc[t.f1.idxmax()]
    return {
        "threshold_negocio": float(best_value.threshold),
        "threshold_f1": float(best_f1.threshold),
        "tabela": t,
    }


def classify_risk(proba, thresholds: dict | None = None) -> np.ndarray:
    th = thresholds or config.RISK_THRESHOLDS
    p = np.asarray(proba)
    return np.select([p >= th["alto"], p >= th["medio"]], ["Alto", "Médio"], "Baixo")


def lift_table(y_true, proba, bins: int = 10) -> pd.DataFrame:
    """Taxa real de churn por decil de risco previsto (decil 1 = maior risco)."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    d["decil"] = pd.qcut(d.p.rank(method="first", ascending=False), bins, labels=range(1, bins + 1))
    g = d.groupby("decil", observed=True).agg(clientes=("y", "size"), churns=("y", "sum"),
                                              prob_media=("p", "mean"), taxa_real=("y", "mean"))
    g["lift"] = g.taxa_real / d.y.mean()
    g["captura_acumulada"] = g.churns.cumsum() / d.y.sum()
    return g
