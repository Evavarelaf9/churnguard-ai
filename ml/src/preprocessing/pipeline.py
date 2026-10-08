"""Pipeline de preparação dos dados e divisão temporal.

Estrutura do pipeline completo (um único objeto scikit-learn serializado):

    ┌───────────────────┐   ┌────────────────────────────────────────────┐   ┌────────┐
    │ FeatureEngineer   │ → │ ColumnTransformer                          │ → │ Modelo │
    │ (sem estado)      │   │  numéricas : Winsorizer → Imputer → Scaler │   └────────┘
    └───────────────────┘   │  binárias  : Imputer                       │
                            │  categóricas: Imputer → OneHotEncoder      │
                            └────────────────────────────────────────────┘

Tudo que "aprende" com os dados (limites de outlier, medianas, médias/desvios,
categorias) é ajustado dentro de `fit`, ou seja, **apenas no treino** — e, na
validação cruzada, apenas no treino de cada dobra.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.src import config
from ml.src.features.build_features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    add_features,
)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita valores extremos aos percentis aprendidos no treino.

    Justificativa: a EDA mostrou outliers de minutos de uso gerados por
    integrações/bots. Remover linhas descartaria clientes legítimos; limitar
    ("capar") preserva a linha e reduz a influência do valor extremo — em
    especial na Regressão Logística, sensível à escala.
    """

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = X.quantile(self.lower).to_numpy()
        self.upper_ = X.quantile(self.upper).to_numpy()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    numeric_steps = [
        ("winsorizer", Winsorizer()),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Não informado")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Aplica `add_features` e seleciona apenas as colunas usadas pelo modelo.

    Sem estado (fit não aprende nada). Selecionar as colunas aqui garante que
    colunas extras na entrada — alvo, identificadores, idade, região — nunca
    cheguem ao modelo, mesmo que alguém as envie por engano.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(pd.DataFrame(X))[MODEL_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(MODEL_FEATURES, dtype=object)


def feature_engineer() -> FeatureEngineer:
    return FeatureEngineer()


def build_pipeline(model, scale: bool = True, sampler=None):
    """Pipeline completo: atributos → pré-processamento → (reamostragem) → modelo.

    Se `sampler` for informado (SMOTE, under/oversampling), usa o Pipeline do
    imbalanced-learn, que aplica a reamostragem **somente durante o fit** — nunca
    nos dados de validação/teste nem na predição.
    """
    steps = [("features", feature_engineer()), ("preprocess", build_preprocessor(scale))]
    if sampler is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(steps + [("sampler", sampler), ("model", model)])
    return Pipeline(steps + [("model", model)])


# --------------------------------------------------------------------------- #
# Divisão temporal
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TemporalSplit:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> str:
        return (
            f"Treino: snapshots ≤ {self.train_end.date()} | "
            f"intervalo de {config.CHURN_HORIZON_DAYS} dias sem dados | "
            f"Teste: {self.test_start.date()} → {self.test_end.date()}"
        )


DEFAULT_SPLIT = TemporalSplit(
    train_end=pd.Timestamp("2025-09-30"),
    test_start=pd.Timestamp("2025-12-31"),
    test_end=config.LAST_LABELED_SNAPSHOT,
)


def temporal_train_test_split(df: pd.DataFrame, split: TemporalSplit = DEFAULT_SPLIT):
    """Treino no passado, teste no futuro, com um intervalo entre eles.

    O intervalo existe porque o rótulo olha 60 dias à frente: o rótulo do último
    snapshot de treino (30/09/2025) depende de eventos até 29/11/2025. Se o teste
    começasse antes disso, os períodos se sobreporiam e informação do "futuro do
    treino" estaria no teste (e vice-versa).
    """
    ref = df["data_referencia"]
    horizon = pd.Timedelta(days=config.CHURN_HORIZON_DAYS)
    assert split.test_start > split.train_end + horizon, "teste precisa começar após o horizonte do treino"
    train = df[ref <= split.train_end].reset_index(drop=True)
    test = df[(ref >= split.test_start) & (ref <= split.test_end)].reset_index(drop=True)
    return train, test


class TemporalGroupSplit:
    """Validação cruzada com janela expansiva por data de referência.

    Cada dobra valida em `val_size` snapshots consecutivos e treina só com
    snapshots anteriores, deixando `gap` snapshots de intervalo (horizonte do
    rótulo). Nunca treina com o futuro.
    """

    def __init__(self, n_splits: int = 3, val_size: int = 3, gap: int = 2):
        self.n_splits = n_splits
        self.val_size = val_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        dates = pd.Series(pd.to_datetime(groups if groups is not None else X["data_referencia"])).reset_index(drop=True)
        uniq = np.sort(dates.unique())
        for k in range(self.n_splits, 0, -1):
            val_end = len(uniq) - (k - 1) * self.val_size
            val_start = val_end - self.val_size
            train_end = val_start - self.gap
            if train_end <= 0:
                continue
            train_dates, val_dates = uniq[:train_end], uniq[val_start:val_end]
            yield (
                np.where(dates.isin(train_dates))[0],
                np.where(dates.isin(val_dates))[0],
            )

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
