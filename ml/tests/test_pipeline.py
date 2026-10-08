import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.src import config
from ml.src.preprocessing.pipeline import (
    DEFAULT_SPLIT,
    TemporalGroupSplit,
    Winsorizer,
    build_pipeline,
    temporal_train_test_split,
)


def test_temporal_split_has_gap(base):
    train, test = temporal_train_test_split(base, DEFAULT_SPLIT)
    gap = test.data_referencia.min() - train.data_referencia.max()
    assert gap > pd.Timedelta(days=config.CHURN_HORIZON_DAYS)
    assert train.data_referencia.max() < test.data_referencia.min()


def test_cv_never_trains_on_future(base):
    train, _ = temporal_train_test_split(base)
    for tr, va in TemporalGroupSplit().split(train, groups=train.data_referencia):
        last_train = train.data_referencia.iloc[tr].max()
        first_val = train.data_referencia.iloc[va].min()
        assert first_val - last_train > pd.Timedelta(days=config.CHURN_HORIZON_DAYS)


def test_winsorizer_learns_only_from_fit_data():
    w = Winsorizer(0.0, 0.9).fit(pd.DataFrame({"x": np.arange(100)}))
    out = w.transform(pd.DataFrame({"x": [1_000_000]}))
    assert out[0, 0] <= 100


def test_pipeline_ignores_extra_columns(sample):
    """Colunas extras (alvo, idade, região) não mudam a previsão."""
    pipe = build_pipeline(LogisticRegression(max_iter=500), scale=True).fit(sample, sample.churn)
    p1 = pipe.predict_proba(sample)[:, 1]
    p2 = pipe.predict_proba(sample.drop(columns=["churn", "idade", "regiao", "nome"]))[:, 1]
    np.testing.assert_allclose(p1, p2)


def test_pipeline_handles_unseen_category(sample):
    pipe = build_pipeline(LogisticRegression(max_iter=500), scale=True).fit(sample, sample.churn)
    odd = sample.head(2).copy()
    odd["forma_pagamento"] = "Criptomoeda"
    assert pipe.predict_proba(odd).shape == (2, 2)
