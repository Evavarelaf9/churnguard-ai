import numpy as np

from ml.src.features.build_features import add_features
from ml.src.monitoring.drift import (
    build_reference_profile,
    compare_to_reference,
    psi,
    psi_status,
)


def test_psi_identical_is_zero_and_shift_is_large():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(p, p) == 0
    assert psi(p, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


def test_psi_status_thresholds():
    assert psi_status(0.05) == "estável"
    assert psi_status(0.15) == "moderado"
    assert psi_status(0.30) == "significativo"


def test_drift_detects_shift_in_one_feature(base):
    ref = add_features(base[base.data_referencia <= "2025-09-30"].sample(4000, random_state=0))
    profile = build_reference_profile(ref, "teste")
    same = add_features(base[base.data_referencia <= "2025-09-30"].sample(4000, random_state=1))
    shifted = same.copy()
    shifted["tempo_medio_resolucao_h_6m"] = shifted["tempo_medio_resolucao_h_6m"] * 3
    by_feat = {d["feature"]: d for d in compare_to_reference(profile, same)}
    assert all(d["status"] == "estável" for d in by_feat.values())
    moved = {d["feature"]: d for d in compare_to_reference(profile, shifted)}
    assert moved["tempo_medio_resolucao_h_6m"]["status"] == "significativo"
