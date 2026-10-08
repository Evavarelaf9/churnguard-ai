"""Serviços de negócio: clientes, dashboard, modelo e ações de retenção."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.models.entities import Prediction, RetentionAction
from backend.app.repositories.repositories import (
    AccessLogRepository,
    ActionRepository,
    CustomerRepository,
    ModelRepository,
    PredictionRepository,
)
from backend.app.services.scoring import (
    DISCLAIMER,
    CustomerNotFound,
    effective_thresholds,
    get_scorer,
)
from ml.src.evaluation.metrics import classify_risk
from ml.src.explainability.explainer import ChurnExplainer

WEEKS = 4.33


def prediction_to_result(p: Prediction) -> dict:
    fatores = p.fatores or {}
    scorer = get_scorer()
    th = float(p.threshold)
    faixas = dict(scorer.risk_thresholds)
    faixas["alto"] = th
    return {
        "customer_id": p.customer_id, "data_referencia": p.data_referencia,
        "probabilidade": float(p.probabilidade), "nivel_risco": p.nivel_risco,
        "acima_do_threshold": float(p.probabilidade) >= th, "threshold": th, "faixas_risco": faixas,
        "model_version": p.model_version, "principal_fator": p.principal_fator,
        "resumo": fatores.get("resumo"),
        "categoria_principal": p.categoria_principal, "fatores_risco": fatores.get("risco", []),
        "fatores_protecao": fatores.get("protecao", []), "recomendacao": p.recomendacao,
        "prediction_id": p.id, "aviso": DISCLAIMER,
    }


# --------------------------------------------------------------------------- #
# Clientes
# --------------------------------------------------------------------------- #
def list_customers(db: Session, threshold: float | None = None, medio: float | None = None, **filters) -> dict:
    page, size = filters.pop("pagina"), filters.pop("tamanho_pagina")
    th = effective_thresholds(get_scorer(), threshold, medio)
    rows, total = PredictionRepository(db).list_latest(limit=size, offset=(page - 1) * size, thresholds=th, **filters)
    for r in rows:  # a sugestão acompanha o nível efetivo
        rec = ChurnExplainer.recommend(r["nivel_risco"], r["categoria_principal"], r["ciclo_contrato"])
        r["acao_sugerida"] = rec["acao_sugerida"]
    return {"total": total, "pagina": page, "tamanho_pagina": size, "faixas_risco": th, "itens": rows}


def threshold_tradeoff(db: Session) -> dict:
    """Precision/recall por threshold com base em previsões cujo resultado real já é conhecido,
    e quantos clientes ativos seriam sinalizados hoje em cada threshold."""
    repo = PredictionRepository(db)
    outcomes = np.array(repo.labeled_outcomes())
    current = np.array([r["probabilidade"] for r in repo.latest_frame_rows()])
    grid = np.round(np.arange(0.05, 0.71, 0.01), 2)
    rows = []
    p, y = (outcomes[:, 0], outcomes[:, 1]) if len(outcomes) else (np.array([]), np.array([]))
    for t in grid:
        flag = p >= t
        tp = int(((y == 1) & flag).sum())
        fp = int(((y == 0) & flag).sum())
        fn = int(((y == 1) & ~flag).sum())
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        rows.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1,
                     "abordados_historico": tp + fp, "churns_identificados": tp, "churns_totais": tp + fn,
                     "percentual_sinalizado": float(flag.mean()) if len(p) else None,
                     "clientes_sinalizados_hoje": int((current >= t).sum())})
    scorer = get_scorer()
    return {"threshold_modelo": scorer.risk_thresholds["alto"], "faixas_risco": scorer.risk_thresholds,
            "base_observacoes": int(len(outcomes)), "taxa_real": float(y.mean()) if len(y) else None,
            "clientes_ativos": int(len(current)), "pontos": rows,
            "nota": "Calculado com previsões cujo resultado real (60 dias) já foi observado."}


def filter_options(db: Session) -> dict:
    return {"planos": [p.plano for p in CustomerRepository(db).plans()],
            "regioes": ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
            "niveis_risco": ["Alto", "Médio", "Baixo"],
            "datas_referencia": PredictionRepository(db).reference_dates(),
            "faixas_valor": [{"rotulo": "até R$ 60", "min": 0, "max": 60},
                             {"rotulo": "R$ 60–120", "min": 60, "max": 120},
                             {"rotulo": "R$ 120–250", "min": 120, "max": 250},
                             {"rotulo": "acima de R$ 250", "min": 250, "max": None}]}


def customer_detail(db: Session, customer_id: str, usuario: str) -> dict:
    repo = CustomerRepository(db)
    c = repo.get(customer_id)
    if c is None:
        raise CustomerNotFound(customer_id)
    preds = PredictionRepository(db)
    latest = preds.latest_for_customer(customer_id)
    payments = []
    for p in repo.payment_history(customer_id):
        paid = p.data_pagamento
        delay = (paid - p.data_vencimento).days if paid else None
        payments.append({**{k: getattr(p, k) for k in ("mes_referencia", "data_vencimento", "data_pagamento", "status")},
                         "valor_cobrado": float(p.valor_cobrado), "dias_atraso": max(delay, 0) if delay is not None else None})
    ref = get_settings().today
    end = c.data_cancelamento or ref
    AccessLogRepository(db).log(usuario, "visualizar_cliente", customer_id)
    db.commit()
    return {
        "perfil": {
            "customer_id": c.customer_id, "nome": c.nome, "tipo_cliente": c.tipo_cliente, "regiao": c.regiao,
            "plano": c.plan.plano, "ciclo_contrato": c.ciclo_contrato, "forma_pagamento": c.forma_pagamento,
            "renovacao_automatica": c.renovacao_automatica, "canal_aquisicao": c.canal_aquisicao,
            "data_inicio": c.data_inicio, "status": c.status,
            "tempo_cliente_meses": round((end - c.data_inicio).days / 30.44, 1),
            "valor_mensal": payments[-1]["valor_cobrado"] if payments else None,
        },
        "previsao_atual": prediction_to_result(latest) if latest else None,
        "historico_uso": repo.usage_history(customer_id),
        "historico_pagamentos": payments,
        "chamados": repo.tickets(customer_id),
        "pesquisas": repo.surveys(customer_id),
        "historico_previsoes": preds.history_for_customer(customer_id),
        "acoes_retencao": ActionRepository(db).list(customer_id=customer_id),
    }


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
def _band(values: pd.Series, bins, labels) -> pd.Series:
    return pd.cut(values, bins, labels=labels, include_lowest=True)


def dashboard_summary(db: Session, threshold: float | None = None, medio: float | None = None) -> dict:
    repo = PredictionRepository(db)
    df = pd.DataFrame(repo.latest_frame_rows())
    scorer = get_scorer()
    if df.empty:
        return {"vazio": True}
    th = effective_thresholds(scorer, threshold, medio)
    df["nivel_risco"] = classify_risk(df.probabilidade.to_numpy(), th)
    ent = pd.json_normalize(df.entrada)
    df["valor_mensal"] = ent.valor_mensal.astype(float)
    df["acessos_semana"] = ent.acessos_m1.astype(float) / WEEKS
    df["receita_esperada_perdida"] = df.probabilidade * df.valor_mensal
    levels = ["Alto", "Médio", "Baixo"]
    counts = df.nivel_risco.value_counts().reindex(levels, fill_value=0)

    def group(col, order=None):
        g = df.groupby(col, observed=True).agg(
            clientes=("customer_id", "size"), probabilidade_media=("probabilidade", "mean"),
            alto_risco=("nivel_risco", lambda s: int((s == "Alto").sum())),
            receita_mensal=("valor_mensal", "sum"),
            receita_em_risco=("valor_mensal", lambda s: float(s[df.loc[s.index, "nivel_risco"] == "Alto"].sum())),
        )
        if order:
            g = g.reindex([o for o in order if o in g.index])
        ints = {"clientes", "alto_risco"}
        return [{"grupo": str(k), **{kk: (int(vv) if kk in ints else float(vv)) for kk, vv in v.items()}}
                for k, v in g.iterrows()]

    df["faixa_valor"] = _band(df.valor_mensal, [0, 60, 120, 250, 1e9],
                              ["≤ R$ 60", "R$ 60–120", "R$ 120–250", "> R$ 250"])
    df["faixa_uso"] = _band(df.acessos_semana, [-0.1, 2, 5, 10, 20, 1e9],
                            ["≤2", "2–5", "5–10", "10–20", ">20"])
    df["faixa_prob"] = _band(df.probabilidade, [0, .05, .1, .15, .2, .3, .4, .5, 1],
                             ["0–5%", "5–10%", "10–15%", "15–20%", "20–30%", "30–40%", "40–50%", "50–100%"])

    # Sinais ACIONÁVEIS mais frequentes entre clientes que precisam de atenção
    # (fatores estruturais, como contrato mensal ou canal de aquisição, ficam fora desta contagem).
    attention = df[df.nivel_risco.isin(["Alto", "Médio"])]
    factor_counter, factor_cat = Counter(), {}
    for f in attention.fatores:
        acionaveis = [i for i in (f or {}).get("risco", []) if i.get("acionavel", True)]
        for item in acionaveis[:3]:
            factor_counter[item["label"]] += 1
            factor_cat[item["label"]] = item["category_label"]
    top_factors = [{"fator": k, "categoria": factor_cat[k], "clientes": v,
                    "percentual": v / max(len(attention), 1)} for k, v in factor_counter.most_common(8)]

    monthly = db.execute(text("""
        WITH meses AS (
            SELECT generate_series(date_trunc('month', CAST(:ref AS date)) - interval '23 months',
                                   date_trunc('month', CAST(:ref AS date)), interval '1 month')::date AS mes)
        SELECT m.mes,
               (SELECT COUNT(*) FROM customers c WHERE c.data_inicio < m.mes
                   AND (c.data_cancelamento IS NULL OR c.data_cancelamento >= m.mes)) AS ativos,
               (SELECT COUNT(*) FROM customers c WHERE c.data_cancelamento >= m.mes
                   AND c.data_cancelamento < (m.mes + interval '1 month')) AS cancelamentos
        FROM meses m ORDER BY m.mes"""), {"ref": get_settings().today}).mappings().all()
    evol = [{"mes": r["mes"], "clientes_ativos": r["ativos"], "cancelamentos": r["cancelamentos"],
             "taxa_churn": r["cancelamentos"] / r["ativos"] if r["ativos"] else 0} for r in monthly]
    perf = [{"data_referencia": r["data_referencia"], "churn_previsto": float(r["churn_previsto"]),
             "churn_real": float(r["churn_real"])} for r in repo.monthly_performance()]

    mv = ModelRepository(db).active()
    m = scorer.metadata["metrics"]["test_at_operational_threshold"]
    return {
        "data_referencia": df.data_referencia.max(),
        "model_version": scorer.version,
        "faixas_risco": th,
        "faixas_risco_modelo": scorer.risk_thresholds,
        "kpis": {
            "clientes_monitorados": int(len(df)),
            "alto_risco": int(counts["Alto"]), "medio_risco": int(counts["Médio"]), "baixo_risco": int(counts["Baixo"]),
            "clientes_atencao": int(counts["Alto"] + counts["Médio"]),
            "taxa_churn_estimada": float(df.probabilidade.mean()),
            "receita_mensal_total": float(df.valor_mensal.sum()),
            "receita_mensal_em_risco": float(df.loc[df.nivel_risco == "Alto", "valor_mensal"].sum()),
            "receita_esperada_perdida": float(df.receita_esperada_perdida.sum()),
            "previsoes_realizadas": repo.count(),
        },
        "distribuicao_risco": [{"nivel": lv, "clientes": int(counts[lv]), "percentual": float(counts[lv] / len(df)),
                                "receita_mensal": float(df.loc[df.nivel_risco == lv, "valor_mensal"].sum())}
                               for lv in levels],
        "histograma_probabilidade": [{"faixa": str(k), "clientes": int(v)}
                                     for k, v in df.faixa_prob.value_counts(sort=False).items()],
        "principais_fatores": top_factors,
        "importancia_global": scorer.metadata.get("global_importance", []),
        "categorias_alto_risco": [{"categoria": k, "clientes": int(v)} for k, v in
                                  df[df.nivel_risco == "Alto"].categoria_principal.value_counts().items()],
        "por_plano": group("plano", ["Básico", "Essencial", "Profissional", "Empresarial"]),
        "por_faixa_valor": group("faixa_valor", ["≤ R$ 60", "R$ 60–120", "R$ 120–250", "> R$ 250"]),
        "por_comportamento": group("faixa_uso", ["≤2", "2–5", "5–10", "10–20", ">20"]),
        "por_regiao": group("regiao"),
        "evolucao_churn": evol,
        "previsto_vs_real": perf,
        "metricas_modelo": {
            "nome": mv.model_name if mv else scorer.metadata["model_name"],
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "threshold": m["threshold"],
            "matriz_confusao": {"verdadeiro_positivo": m["tp"], "falso_positivo": m["fp"],
                                "verdadeiro_negativo": m["tn"], "falso_negativo": m["fn"]},
        },
    }


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def model_info() -> dict:
    meta = get_scorer().metadata
    return {
        "version": meta["model_version"], "model_name": meta["model_name"], "trained_at": meta["trained_at"],
        "target": meta["target"], "horizon_days": meta["horizon_days"], "threshold": meta["threshold"]["operacional"],
        "risk_thresholds": meta["risk_thresholds"], "best_params": meta["best_params"],
        "imbalance_strategy": meta.get("imbalance_strategy"), "business_assumptions": meta["business_assumptions"],
        "features": meta["features"], "training_data": meta["training_data"],
    }


def model_metrics() -> dict:
    meta = get_scorer().metadata
    m = meta["metrics"]["test_at_operational_threshold"]
    td = meta["training_data"]
    return {
        "version": meta["model_version"], "threshold": m["threshold"],
        **{k: m[k] for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "brier", "base_rate")},
        "matriz_confusao": {"verdadeiro_positivo": m["tp"], "falso_positivo": m["fp"],
                            "verdadeiro_negativo": m["tn"], "falso_negativo": m["fn"]},
        "validacao_cruzada": {k: v for k, v in meta["metrics"]["cv"].items() if not isinstance(v, list)},
        "periodo_teste": f"{td['test_start']} → {td['test_end']}",
        "resultado_esperado": meta["metrics"]["test_expected_value"],
        "observacao": ("Métricas medidas no período de teste (dados nunca vistos no treino). A acurácia isolada não é "
                       "indicada para churn: com ~9% de cancelamentos, prever 'ninguém cancela' teria ~91% de acurácia."),
    }


# --------------------------------------------------------------------------- #
# Ações de retenção
# --------------------------------------------------------------------------- #
def create_action(db: Session, customer_id: str, data: dict, usuario: str) -> RetentionAction:
    if CustomerRepository(db).get(customer_id) is None:
        raise CustomerNotFound(customer_id)
    if data.get("prediction_id") is None:
        latest = PredictionRepository(db).latest_for_customer(customer_id)
        data["prediction_id"] = latest.id if latest else None
    action = ActionRepository(db).add(RetentionAction(customer_id=customer_id, **data))
    AccessLogRepository(db).log(usuario, "registrar_acao", customer_id, {"acao_id": action.id, "tipo": action.tipo_acao})
    db.commit()
    return action


def update_action(db: Session, action_id: int, data: dict, usuario: str) -> RetentionAction | None:
    repo = ActionRepository(db)
    action = repo.get(action_id)
    if action is None:
        return None
    for k, v in data.items():
        setattr(action, k, v)
    AccessLogRepository(db).log(usuario, "atualizar_acao", action.customer_id, {"acao_id": action_id, **data})
    db.commit()
    db.refresh(action)
    return action
