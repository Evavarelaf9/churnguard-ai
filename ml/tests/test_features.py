import numpy as np
import pandas as pd

from ml.src.features.build_features import (
    EXCLUDED_COLUMNS,
    MODEL_FEATURES,
    _consecutive_declines,
    add_features,
)


def test_add_features_does_not_mutate_input(sample):
    before = sample.copy()
    add_features(sample)
    pd.testing.assert_frame_equal(sample, before)


def test_features_are_row_wise(sample):
    """A mesma linha gera os mesmos atributos sozinha ou junto com outras (sem estado)."""
    full = add_features(sample)[MODEL_FEATURES]
    single = add_features(sample.iloc[[7]])[MODEL_FEATURES]
    pd.testing.assert_frame_equal(full.iloc[[7]], single)


def test_no_leakage_or_sensitive_columns_in_model():
    for col in ["churn", "customer_id", "idade", "regiao", "nome", "data_referencia",
                "data_cancelamento", "motivo_cancelamento", "status"]:
        assert col not in MODEL_FEATURES
    assert set(EXCLUDED_COLUMNS).isdisjoint(MODEL_FEATURES)


def test_engineered_features_have_no_infinite_values(sample):
    f = add_features(sample)[MODEL_FEATURES].select_dtypes("number")
    assert np.isfinite(f.fillna(0).to_numpy()).all()


def test_usage_variation_sign():
    row = pd.DataFrame([{**{f"acessos_m{i}": v for i, v in zip(range(1, 7), [10, 10, 10, 20, 20, 20])}}])
    base = {"funcionalidades_usadas_m1": 2, "funcionalidades_disponiveis": 5, "dias_ativos_m1": 10,
            "minutos_uso_m1": 100, "sessoes_m1": 10, "minutos_uso_m2": 200, "faturas_atrasadas_6m": 0,
            "faturas_6m": 6, "faturas_em_aberto": 0, "reajuste_pct_6m": np.nan, "valor_total_pago": 500,
            "tempo_cliente_meses": 14, "tickets_6m": 2, "tickets_3m": 2, "reclamacoes_6m": 0,
            "dias_desde_ultimo_chamado": 10, "avaliacao_media_6m": 4, "ultima_nota_satisfacao": 8,
            "ciclo_contrato": "Anual"}
    f = add_features(row.assign(**base)).iloc[0]
    assert f.variacao_uso == -0.5
    assert f.tendencia_uso < 0
    assert f.meses_ate_renovacao == 10  # 14 meses → 2º ano, 2 meses no ciclo → faltam 10
    assert f.aceleracao_tickets == 2


def test_consecutive_declines():
    lags = np.array([[50, 40, 30, 20, 10, 5], [10, 20, 30, 40, 50, 45], [10, 10, 10, 10, 10, 10]], dtype=float)
    assert _consecutive_declines(lags).tolist() == [5, 1, 0]
