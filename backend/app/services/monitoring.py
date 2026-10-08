"""Monitoramento do modelo em produção.

Um modelo de ML não é treinado uma vez e esquecido: o comportamento dos clientes,
os preços e a operação mudam. Este serviço responde, a cada execução:

1. **Desempenho** — nas previsões cujo resultado real (60 dias) já é conhecido:
   precision, recall, PR-AUC, falsos positivos/negativos e calibração (previsto × real).
2. **Data drift** — a distribuição dos atributos atuais mudou em relação aos dados
   com os quais o modelo foi validado? (PSI por atributo, ponderado pela importância).
3. **Volume e distribuição das previsões** — quantas previsões, quanto risco.
4. **Alertas e recomendações** — status geral (ok / atenção / crítico) e o que fazer.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.services.scoring import get_scorer
from ml.src.features.build_features import FEATURE_LABELS, add_features
from ml.src.monitoring.drift import (
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    REFERENCE_FILE,
    compare_to_reference,
)

# Política de alertas (documentada em docs/monitoring.md)
POLICY = {
    "janela_desempenho_meses": 3,
    "queda_relativa_recall_critica": 0.20,     # recall 20% abaixo do validado
    "queda_relativa_pr_auc_critica": 0.20,
    "gap_calibracao_atencao": 0.03,             # |churn previsto − real| > 3 p.p.
    "psi_moderado": PSI_MODERATE,
    "psi_significativo": PSI_SIGNIFICANT,
    "top_atributos_importantes": 10,
}
def _pct(v: float) -> str:
    return f"{v * 100:.1f}%".replace(".", ",")


def _dec(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


SEVERITY = {"ok": 0, "info": 0, "atenção": 1, "crítico": 2}


@lru_cache
def reference_profile() -> dict:
    return json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))


def _load_predictions(db: Session) -> pd.DataFrame:
    rows = db.execute(text("""
        SELECT data_referencia, origem, probabilidade::float AS probabilidade, threshold::float AS threshold,
               churn_real, entrada
        FROM predictions WHERE origem IN ('historico', 'batch')""")).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


def _performance(df: pd.DataFrame) -> list[dict]:
    out = []
    for ref, g in df[df.churn_real.notna()].groupby("data_referencia"):
        y = g.churn_real.astype(int).to_numpy()
        p = g.probabilidade.to_numpy()
        flag = p >= g.threshold.to_numpy()
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        out.append({
            "data_referencia": ref, "previsoes": int(len(g)),
            "churn_real": float(y.mean()), "churn_previsto": float(p.mean()),
            "gap_calibracao": float(p.mean() - y.mean()),
            "precision": prec, "recall": rec,
            "f1": (2 * prec * rec / (prec + rec)) if prec and rec else None,
            "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
            "vp": tp, "fp": fp, "fn": fn, "vn": tn,
        })
    return out


def _drift(df: pd.DataFrame, important: set[str]) -> tuple[list[dict], list[dict], date | None]:
    profile = reference_profile()
    has_input = df[df.entrada.notna()]
    timeline, current, current_date = [], [], None
    for ref, g in has_input.groupby("data_referencia"):
        feats = add_features(pd.json_normalize(list(g.entrada)))
        comp = compare_to_reference(profile, feats)
        timeline.append({"data_referencia": ref, "origem": g.origem.iloc[0],
                         "psi": {c["feature"]: round(c["psi"], 4) for c in comp}})
        current, current_date = comp, ref  # grupos ordenados por data → o último é o atual
    for c in current:
        c["importante"] = c["feature"] in important
    return current, timeline, current_date


def compute(db: Session) -> dict:
    scorer = get_scorer()
    meta = scorer.metadata
    baseline = meta["metrics"]["test_at_operational_threshold"]
    importance = meta.get("global_importance", [])
    important = {f["feature"] for f in importance[: POLICY["top_atributos_importantes"]]}
    rank = {f["feature"]: i + 1 for i, f in enumerate(importance)}

    df = _load_predictions(db)
    perf = _performance(df)
    drift_now, drift_timeline, drift_date = _drift(df, important)
    for d in drift_now:
        d["ranking_importancia"] = rank.get(d["feature"])

    alerts: list[dict] = []
    # 1) Desempenho recente
    recent = perf[-POLICY["janela_desempenho_meses"]:]
    if recent:
        rec = np.mean([r["recall"] for r in recent if r["recall"] is not None])
        pra = np.mean([r["pr_auc"] for r in recent if r["pr_auc"] is not None])
        if rec < baseline["recall"] * (1 - POLICY["queda_relativa_recall_critica"]):
            alerts.append({"nivel": "crítico", "tipo": "desempenho", "titulo": "Queda de recall",
                           "detalhe": f"Recall médio dos últimos {len(recent)} meses: {_pct(rec)} (validado: {_pct(baseline['recall'])}).",
                           "recomendacao": "Investigar mudanças recentes e avaliar re-treino com dados atualizados."})
        if pra < baseline["pr_auc"] * (1 - POLICY["queda_relativa_pr_auc_critica"]):
            alerts.append({"nivel": "crítico", "tipo": "desempenho", "titulo": "Queda de PR-AUC",
                           "detalhe": f"PR-AUC médio recente: {_dec(pra, 3)} (validado: {_dec(baseline['pr_auc'], 3)}).",
                           "recomendacao": "Avaliar re-treino e revisão de atributos."})
    miscal = [r for r in perf if abs(r["gap_calibracao"]) > POLICY["gap_calibracao_atencao"]]
    if miscal:
        worst = max(miscal, key=lambda r: abs(r["gap_calibracao"]))
        meses = ", ".join(pd.Timestamp(r["data_referencia"]).strftime("%m/%Y") for r in miscal)
        alerts.append({"nivel": "atenção", "tipo": "calibração", "titulo": "Churn real diferente do previsto",
                       "detalhe": (f"Em {meses}, o churn real ficou mais de {POLICY['gap_calibracao_atencao'] * 100:.0f} p.p. distante do previsto "
                                   f"(pior caso: previsto {_pct(worst['churn_previsto'])} × real {_pct(worst['churn_real'])}). "
                                   "Os meses coincidem com o reajuste anual de janeiro."),
                       "recomendacao": "Incluir o efeito do reajuste no próximo treino (ex.: atributo de sensibilidade a preço) "
                                       "e reforçar ações de retenção antes de reajustes."})
    # 2) Drift dos dados atuais
    sig_imp = [d for d in drift_now if d["status"] == "significativo" and d["importante"]]
    sig_other = [d for d in drift_now if d["status"] == "significativo" and not d["importante"]]
    mod_imp = [d for d in drift_now if d["status"] == "moderado" and d["importante"]]
    if sig_imp:
        alerts.append({"nivel": "crítico", "tipo": "drift", "titulo": "Mudança significativa em atributos importantes",
                       "detalhe": ", ".join(f"{d['label']} (PSI {_dec(d['psi'])})" for d in sig_imp),
                       "recomendacao": "Re-treinar o modelo e validar novamente antes de usar as previsões."})
    if sig_other:
        alerts.append({"nivel": "atenção", "tipo": "drift", "titulo": "Mudança significativa em atributos de baixa influência",
                       "detalhe": ", ".join(f"{d['label']} (PSI {_dec(d['psi'])})" for d in sig_other[:5]),
                       "recomendacao": "Acompanhar. Preços e reajustes mudaram desde a validação; como esses atributos pesam "
                                       "pouco no modelo, o impacto esperado é limitado — confirmar no desempenho."})
    if mod_imp:
        alerts.append({"nivel": "atenção", "tipo": "drift", "titulo": "Mudança moderada em atributos importantes",
                       "detalhe": ", ".join(f"{d['label']} (PSI {_dec(d['psi'])})" for d in mod_imp),
                       "recomendacao": "Investigar a causa (operação, produto, cadastro) e acompanhar a tendência."})
    trend = _rising(drift_timeline, "tempo_medio_resolucao_h_6m")
    if trend:
        alerts.append({"nivel": "info", "tipo": "tendência", "titulo": "Tempo de resolução de chamados em alta",
                       "detalhe": trend, "recomendacao": "Sinal operacional: verificar a capacidade do time de suporte."})

    status = max((a["nivel"] for a in alerts), key=lambda n: SEVERITY[n], default="ok")
    status = "ok" if status == "info" else status
    volume = (df.groupby(["data_referencia", "origem"]).size().rename("previsoes").reset_index()
              .to_dict("records"))
    dist = (df.assign(alto=df.probabilidade >= df.threshold).groupby("data_referencia")
            .agg(probabilidade_media=("probabilidade", "mean"), percentual_alto=("alto", "mean")).reset_index()
            .to_dict("records"))
    pending = int(df.churn_real.isna().sum())
    labeled = [r for r in perf]
    tot = {k: sum(r[k] for r in labeled) for k in ("vp", "fp", "fn", "vn")}

    return {
        "data_referencia": get_settings().today,
        "model_version": scorer.version,
        "status": status,
        "alertas": sorted(alerts, key=lambda a: -SEVERITY[a["nivel"]]),
        "politica": POLICY,
        "referencia_drift": reference_profile()["description"],
        "linha_de_base": {k: baseline[k] for k in ("precision", "recall", "f1", "roc_auc", "pr_auc", "base_rate")},
        "desempenho_mensal": perf,
        "matriz_acumulada": tot,
        "previsoes_aguardando_resultado": pending,
        "drift_atual": {"data_referencia": drift_date, "atributos": drift_now},
        "drift_temporal": drift_timeline,
        "volume": volume,
        "distribuicao_previsoes": dist,
        "rotulos": FEATURE_LABELS,
    }


def _rising(timeline: list[dict], feature: str) -> str | None:
    vals = [(t["data_referencia"], t["psi"].get(feature)) for t in timeline if t["psi"].get(feature) is not None]
    if len(vals) < 4:
        return None
    first, last = vals[0][1], vals[-1][1]
    if last > first * 2 and last > 0.05:
        return (f"PSI de {FEATURE_LABELS[feature].lower()} subiu de {_dec(first, 3)} para {_dec(last, 3)} "
                f"entre {pd.Timestamp(vals[0][0]):%m/%Y} e {pd.Timestamp(vals[-1][0]):%m/%Y}, ainda abaixo do limite de alerta.")
    return None


def run_and_store(db: Session) -> dict:
    """Executa o monitoramento e grava um retrato em monitoring_runs."""
    r = compute(db)
    row = db.execute(text("""
        INSERT INTO monitoring_runs (data_referencia, model_version, status, alertas, desempenho, drift, resumo)
        VALUES (:ref, :mv, :status, CAST(:alertas AS jsonb), CAST(:desempenho AS jsonb), CAST(:drift AS jsonb),
                CAST(:resumo AS jsonb))
        RETURNING id, executado_em"""), {
        "ref": r["data_referencia"], "mv": r["model_version"], "status": r["status"],
        "alertas": json.dumps(r["alertas"], ensure_ascii=False, default=str),
        "desempenho": json.dumps(r["desempenho_mensal"], ensure_ascii=False, default=str),
        "drift": json.dumps(r["drift_atual"], ensure_ascii=False, default=str),
        "resumo": json.dumps({"matriz_acumulada": r["matriz_acumulada"], "linha_de_base": r["linha_de_base"],
                              "aguardando_resultado": r["previsoes_aguardando_resultado"]}, default=str),
    }).one()
    db.commit()
    r["execucao"] = {"id": row.id, "executado_em": row.executado_em}
    return r


def list_runs(db: Session, limit: int = 20) -> list[dict]:
    rows = db.execute(text("""
        SELECT id, executado_em, data_referencia, model_version, status,
               jsonb_array_length(alertas) AS alertas, alertas AS detalhes
        FROM monitoring_runs ORDER BY executado_em DESC LIMIT :l"""), {"l": limit}).mappings().all()
    return [dict(r) for r in rows]
