import pandas as pd
import pytest

from ml.src import config
from ml.src.scoring import ChurnScorer, simulate


@pytest.fixture(scope="module")
def scorer():
    if not config.MODEL_FILE.exists():
        pytest.skip("modelo não treinado — rode ml.src.training.train")
    return ChurnScorer.load()


def test_probabilities_are_valid(scorer, sample):
    out = scorer.predict(sample)
    assert out.probabilidade.between(0, 1).all()
    assert set(out.nivel_risco) <= {"Baixo", "Médio", "Alto"}


def test_risk_levels_follow_thresholds(scorer, sample):
    out = scorer.predict(sample)
    th = scorer.risk_thresholds
    assert (out[out.nivel_risco == "Alto"].probabilidade >= th["alto"]).all()
    assert (out[out.nivel_risco == "Baixo"].probabilidade < th["medio"]).all()


def test_configurable_threshold(scorer, sample):
    strict = scorer.predict(sample, threshold=0.6).acima_do_threshold.sum()
    loose = scorer.predict(sample, threshold=0.1).acima_do_threshold.sum()
    assert loose >= strict


def test_missing_column_is_rejected(scorer, sample):
    with pytest.raises(ValueError, match="ausentes"):
        scorer.predict(sample.drop(columns=["acessos_m1"]))


def test_explanation_structure(scorer, sample):
    e = scorer.explain(sample.head(3))[0]
    assert {"probabilidade", "nivel_risco", "fatores_risco", "principal_fator", "recomendacao"} <= set(e)
    assert "Sugestão" in e["recomendacao"]["aviso"]
    for f in e["fatores_risco"]:
        assert f["impact"] > 0 and f["text"]


def test_simulation_improving_usage_does_not_increase_risk(scorer, base):
    current = pd.read_parquet(config.SCORING_BASE_FILE)
    probs = scorer.predict(current).probabilidade
    customer = current.loc[[probs.sort_values().index[-5]]]  # um dos clientes de maior risco
    r = simulate(scorer, customer, {"dias_ativos_m1": 28, "faturas_atrasadas_6m": 0, "reclamacoes_6m": 0})
    assert r["probabilidade_simulada"] <= r["probabilidade_atual"]
    assert "hipotética" in r["aviso"]
