"""Serviço de previsão: conecta o modelo de ML, o banco e as regras de negócio.

Responsabilidades
- carregar o modelo ativo uma única vez (singleton);
- montar os atributos de um cliente a partir do banco, com a MESMA função do treino
  (`build_snapshot`), respeitando a data de referência (sem olhar o futuro);
- aplicar threshold configurável e traduzir a previsão em nível de risco + sugestão;
- registrar cada previsão (auditoria e monitoramento "previsto × real").
"""
from __future__ import annotations

import math
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.repositories.raw_data import load_raw_frames
from backend.app.repositories.repositories import (
    CustomerRepository,
    PredictionRepository,
)
from ml.src.data.build_analytical_base import build_snapshot
from ml.src.explainability.explainer import ChurnExplainer
from ml.src.features.build_features import RAW_INPUT_COLUMNS
from ml.src.scoring import ChurnScorer

DISCLAIMER = ("Probabilidade estimada a partir de padrões históricos. Não é certeza de cancelamento; "
              "a decisão sobre qualquer ação cabe à equipe responsável.")


class CustomerNotFound(Exception):
    pass


class CustomerNotScorable(Exception):
    pass


@lru_cache
def get_scorer() -> ChurnScorer:
    return ChurnScorer.load(get_settings().model_path)


def _clean(v):
    """Converte tipos numpy/NaN para JSON."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    return v


def effective_thresholds(scorer: ChurnScorer, threshold: float | None, medio: float | None = None) -> dict:
    return scorer.thresholds(threshold, medio)


def score_frame(df: pd.DataFrame, threshold: float | None = None, explain: bool = True, top_k: int = 4,
                medio: float | None = None) -> list[dict]:
    """Pontua um DataFrame no formato de entrada do modelo. Não persiste nada."""
    scorer = get_scorer()
    th = effective_thresholds(scorer, threshold, medio)
    pred = scorer.predict(df, threshold=threshold, medio=medio)
    explanations = scorer.explain(df, top_k=top_k) if explain else [None] * len(df)
    out = []
    for (_, row), (_, p), e in zip(df.iterrows(), pred.iterrows(), explanations):
        level = str(p.nivel_risco)
        r = {
            "probabilidade": float(p.probabilidade),
            "nivel_risco": level,
            "acima_do_threshold": bool(p.acima_do_threshold),
            "threshold": th["alto"],
            "faixas_risco": th,
            "model_version": scorer.version,
            "aviso": DISCLAIMER,
        }
        if e is not None:
            # A recomendação acompanha o nível de risco efetivo (threshold pode ter sido alterado).
            rec = ChurnExplainer.recommend(level, e["categoria_principal"], row.get("ciclo_contrato"))
            r.update({
                "principal_fator": e["principal_fator"],
                "resumo": e["resumo"],
                "categoria_principal": e["categoria_principal"],
                "fatores_risco": e["fatores_risco"],
                "fatores_protecao": e["fatores_protecao"],
                "recomendacao": rec,
            })
        out.append(r)
    return out


def to_prediction_row(customer_id: str, ref: date, features: pd.Series, result: dict, origem: str) -> dict:
    return {
        "customer_id": customer_id,
        "data_referencia": ref,
        "model_version": result["model_version"],
        "probabilidade": round(result["probabilidade"], 5),
        "nivel_risco": result["nivel_risco"],
        "threshold": result["threshold"],
        "principal_fator": result.get("principal_fator"),
        "categoria_principal": result.get("categoria_principal"),
        "fatores": {"risco": result.get("fatores_risco", []), "protecao": result.get("fatores_protecao", []),
                    "resumo": result.get("resumo")},
        "recomendacao": result.get("recomendacao"),
        "entrada": {c: _clean(features.get(c)) for c in RAW_INPUT_COLUMNS},
        "origem": origem,
    }


def build_features_from_db(db: Session, ref: date, customer_ids: list[str] | None = None) -> pd.DataFrame:
    """Atributos point-in-time a partir do banco (mesma função usada no treino)."""
    raw = load_raw_frames(db, ref, customer_ids)
    if raw["customers"].empty:
        return pd.DataFrame()
    return build_snapshot(raw, pd.Timestamp(ref), labeled=False)


def customer_features(db: Session, customer_id: str, ref: date) -> pd.DataFrame:
    customer = CustomerRepository(db).get(customer_id)
    if customer is None:
        raise CustomerNotFound(customer_id)
    if customer.data_cancelamento is not None and customer.data_cancelamento <= ref:
        raise CustomerNotScorable(f"Cliente cancelou em {customer.data_cancelamento:%d/%m/%Y}; não há o que prever.")
    feats = build_features_from_db(db, ref, [customer_id])
    if feats.empty:
        raise CustomerNotScorable("Cliente com menos de 30 dias de histórico na data de referência.")
    return feats


def score_customer(db: Session, customer_id: str, ref: date | None = None, threshold: float | None = None,
                   persist: bool = True, medio: float | None = None) -> dict:
    ref = ref or get_settings().today
    feats = customer_features(db, customer_id, ref)
    result = score_frame(feats, threshold=threshold, medio=medio)[0]
    result.update({"customer_id": customer_id, "data_referencia": ref})
    if persist:
        row = to_prediction_row(customer_id, ref, feats.iloc[0], result, "api")
        pred = PredictionRepository(db).add_many([row])[0]
        db.commit()
        result["prediction_id"] = pred.id
    return result


# --------------------------------------------------------------------------- #
# Simulação de retenção
# --------------------------------------------------------------------------- #
FRIENDLY = {
    "reclamacoes_6m": "reclamações (6 meses)", "tickets_3m": "chamados (3 meses)", "tickets_6m": "chamados (6 meses)",
    "tickets_tecnicos_6m": "chamados técnicos (6 meses)", "forma_pagamento": "pagamento",
    "renovacao_automatica": "renovação automática", "ciclo_contrato": "contrato",
    "ultima_nota_satisfacao": "nota de satisfação", "dias_desde_ultimo_acesso": "dias desde o último acesso",
}
USAGE_SCALE_COLUMNS = ["acessos_m1", "acessos_m2", "sessoes_m1", "minutos_uso_m1", "minutos_uso_m2"]


def apply_changes(db: Session, feats: pd.DataFrame, changes: dict) -> tuple[pd.DataFrame, list[str]]:
    """Aplica mudanças hipotéticas de forma coerente (ex.: trocar plano muda preço e módulos)."""
    f = feats.copy()
    applied = []
    pct = changes.pop("aumento_uso_percentual", None)
    if pct is not None:
        factor = 1 + pct / 100
        for c in USAGE_SCALE_COLUMNS:
            f[c] = (f[c].fillna(0) * factor).round()
        f["dias_ativos_m1"] = (f["dias_ativos_m1"] * factor).clip(0, 30).round()
        f["funcionalidades_usadas_m1"] = np.minimum(
            (f["funcionalidades_usadas_m1"] * max(factor, 0)).round(), f["funcionalidades_disponiveis"])
        if factor > 1:
            f["dias_desde_ultimo_acesso"] = (f["dias_desde_ultimo_acesso"] / factor).round()
        applied.append(f"uso recente {'+' if pct >= 0 else ''}{pct:.0f}%")
    if changes.get("remover_atrasos"):
        f[["faturas_atrasadas_6m", "dias_atraso_medio_6m", "faturas_em_aberto"]] = 0
        applied.append("sem atrasos de pagamento")
    changes.pop("remover_atrasos", None)
    if changes.get("plano"):
        plans = {p.plano: p for p in CustomerRepository(db).plans()}
        plan = plans[changes["plano"]]
        discount = 0.85 if f["ciclo_contrato"].iloc[0] == "Anual" else 1.0
        f["plano"] = plan.plano
        f["funcionalidades_disponiveis"] = plan.funcionalidades_disponiveis
        f["valor_mensal"] = round(float(plan.preco_base) * discount, 2)
        f["funcionalidades_usadas_m1"] = np.minimum(f["funcionalidades_usadas_m1"], plan.funcionalidades_disponiveis)
        applied.append(f"plano {plan.plano}")
    changes.pop("plano", None)
    if changes.get("tickets_3m") is not None and changes.get("tickets_6m") is None:
        # Menos chamados nos últimos 3 meses também reduz o total de 6 meses (mantém a coerência).
        delta = f["tickets_3m"] - changes["tickets_3m"]
        f["tickets_6m"] = (f["tickets_6m"] - delta).clip(lower=changes["tickets_3m"])
        f["tickets_tecnicos_6m"] = np.minimum(f["tickets_tecnicos_6m"], f["tickets_6m"])
        f["reclamacoes_6m"] = np.minimum(f["reclamacoes_6m"], f["tickets_6m"])
    for k, v in changes.items():
        if v is None:
            continue
        if k not in f.columns:
            raise ValueError(f"Atributo não simulável: {k}")
        f[k] = int(v) if isinstance(v, bool) else v
        shown = ("sim" if v else "não") if isinstance(v, bool) else v
        applied.append(f"{FRIENDLY.get(k, k)}: {shown}")
    return f, applied


def simulate_customer(db: Session, customer_id: str, changes: dict, ref: date | None = None,
                      threshold: float | None = None, medio: float | None = None) -> dict:
    ref = ref or get_settings().today
    feats = customer_features(db, customer_id, ref)
    modified, applied = apply_changes(db, feats, dict(changes))
    before = score_frame(feats, threshold=threshold, medio=medio)[0]
    after = score_frame(modified, threshold=threshold, medio=medio)[0]
    return {
        "customer_id": customer_id,
        "data_referencia": ref,
        "alteracoes_aplicadas": applied,
        "atual": before,
        "simulado": after,
        "variacao_pontos_percentuais": round((after["probabilidade"] - before["probabilidade"]) * 100, 2),
        "aviso": ("Simulação hipotética: mostra como o modelo reagiria se esses sinais fossem diferentes. "
                  "Não é uma previsão garantida do comportamento futuro do cliente."),
    }
