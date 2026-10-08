"""Detecção de mudança na distribuição dos dados (data drift).

Métrica: PSI — Population Stability Index.

    PSI = Σ (atual_i − referência_i) × ln(atual_i / referência_i)

calculado sobre faixas (bins) definidas na distribuição de REFERÊNCIA (dados com os
quais o modelo foi validado). Interpretação usual no mercado:

    PSI < 0,10        estável
    0,10 ≤ PSI < 0,25 mudança moderada — investigar
    PSI ≥ 0,25        mudança significativa — avaliar re-treino

Uso:
    python -m ml.src.monitoring.drift      # gera ml/models/reference_profile.json
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ml.src import config
from ml.src.features.build_features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    NUMERIC_FEATURES,
    add_features,
)

REFERENCE_FILE = config.MODELS_DIR / "reference_profile.json"
PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25
EPS = 1e-4
N_BINS = 10


def psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    r = np.clip(np.asarray(ref_pct, float), EPS, None)
    c = np.clip(np.asarray(cur_pct, float), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_status(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return "significativo"
    if value >= PSI_MODERATE:
        return "moderado"
    return "estável"


def _numeric_edges(s: pd.Series) -> list[float]:
    q = np.unique(np.nanquantile(s.dropna(), np.linspace(0, 1, N_BINS + 1)[1:-1]))
    return [float(x) for x in q]


def _numeric_dist(s: pd.Series, edges: list[float]) -> list[float]:
    """Proporções por faixa: (-inf, e1], (e1, e2], ..., (ek, inf) + ausentes."""
    vals = s.to_numpy(float)
    miss = np.isnan(vals)
    idx = np.searchsorted(edges, vals[~miss], side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    total = len(vals)
    return list(np.append(counts, miss.sum()) / max(total, 1))


def _as_category(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.round(4).astype(str)
    return s.astype(str)


def build_reference_profile(features: pd.DataFrame, description: str) -> dict:
    profile = {"created_at": datetime.now().isoformat(timespec="seconds"), "description": description,
               "rows": int(len(features)), "numeric": {}, "categorical": {}}
    discrete = [f for f in NUMERIC_FEATURES if features[f].nunique() <= 12]
    for f in NUMERIC_FEATURES:
        if f in discrete:  # poucos valores distintos → compara frequências, não faixas
            continue
        edges = _numeric_edges(features[f])
        profile["numeric"][f] = {"edges": edges, "dist": _numeric_dist(features[f], edges),
                                 "mean": float(features[f].mean()), "missing": float(features[f].isna().mean())}
    for f in CATEGORICAL_FEATURES + BINARY_FEATURES + discrete:
        freq = _as_category(features[f]).value_counts(normalize=True)
        profile["categorical"][f] = {"freq": {k: float(v) for k, v in freq.items()},
                                     "missing": float(features[f].isna().mean())}
    return profile


def compare_to_reference(profile: dict, features: pd.DataFrame) -> list[dict]:
    """PSI de cada atributo do modelo em relação ao perfil de referência."""
    out = []
    for f, ref in profile["numeric"].items():
        if f not in features:
            continue
        cur = _numeric_dist(features[f], ref["edges"])
        v = psi(ref["dist"], cur)
        out.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "tipo": "numérico", "psi": v, "status": psi_status(v),
                    "media_referencia": ref["mean"], "media_atual": float(features[f].mean()),
                    "ausentes_referencia": ref["missing"], "ausentes_atual": float(features[f].isna().mean())})
    for f, ref in profile["categorical"].items():
        if f not in features:
            continue
        cur_freq = _as_category(features[f]).value_counts(normalize=True)
        cats = sorted(set(ref["freq"]) | set(cur_freq.index))
        v = psi(np.array([ref["freq"].get(c, 0.0) for c in cats]), np.array([cur_freq.get(c, 0.0) for c in cats]))
        top_ref = max(ref["freq"], key=ref["freq"].get)
        out.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "tipo": "categórico", "psi": v, "status": psi_status(v),
                    "media_referencia": ref["freq"].get(top_ref), "media_atual": float(cur_freq.get(top_ref, 0.0)),
                    "categoria_referencia": top_ref,
                    "ausentes_referencia": ref["missing"], "ausentes_atual": float(features[f].isna().mean())})
    return sorted(out, key=lambda d: d["psi"], reverse=True)


def main() -> None:
    """Perfil de referência = dados de treino do modelo validado (antes do período de teste)."""
    from ml.src.preprocessing.pipeline import DEFAULT_SPLIT, temporal_train_test_split

    base = pd.read_parquet(config.ANALYTICAL_BASE_FILE)
    train, _ = temporal_train_test_split(base, DEFAULT_SPLIT)
    feats = add_features(train)
    desc = (f"Dados de treino do modelo validado: snapshots {train.data_referencia.min().date()} → "
            f"{train.data_referencia.max().date()}")
    profile = build_reference_profile(feats, desc)
    REFERENCE_FILE.write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Perfil de referência salvo em {REFERENCE_FILE.relative_to(config.PROJECT_ROOT)} ({len(feats):,} linhas)")


if __name__ == "__main__":
    main()
