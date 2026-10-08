"""Catálogo de modelos candidatos e espaços de busca de hiperparâmetros."""
from __future__ import annotations

from dataclasses import dataclass, field

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.src import config


@dataclass
class ModelSpec:
    key: str
    name: str
    scale: bool  # precisa de padronização?
    estimator: object
    search_space: dict = field(default_factory=dict)
    n_iter: int = 15
    why: str = ""


def model_specs() -> list[ModelSpec]:
    seed = config.RANDOM_SEED
    return [
        ModelSpec(
            key="logreg",
            name="Regressão Logística",
            scale=True,
            estimator=LogisticRegression(max_iter=3000, random_state=seed),
            search_space={
                "model__C": loguniform(1e-3, 10),
                "model__l1_ratio": [0.0, 0.5, 1.0],   # 0 = L2 (ridge), 1 = L1 (lasso)
                "model__solver": ["saga"],
            },
            n_iter=12,
            why="Baseline interpretável: coeficientes mostram direção e força de cada fator.",
        ),
        ModelSpec(
            key="random_forest",
            name="Random Forest",
            scale=False,
            estimator=RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
            search_space={
                "model__n_estimators": [200, 300],
                "model__max_depth": [6, 10, 14, None],
                "model__min_samples_leaf": randint(5, 80),
                "model__max_features": ["sqrt", 0.3, 0.5],
            },
            n_iter=8,
            why="Captura interações e não-linearidades; robusto a outliers e escala.",
        ),
        ModelSpec(
            key="gradient_boosting",
            name="Gradient Boosting (HistGradientBoosting)",
            scale=False,
            estimator=HistGradientBoostingClassifier(random_state=seed),
            search_space={
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__max_iter": randint(150, 600),
                "model__max_leaf_nodes": randint(8, 48),
                "model__min_samples_leaf": randint(20, 300),
                "model__l2_regularization": loguniform(1e-3, 10),
                "model__max_features": uniform(0.4, 0.6),
                "model__early_stopping": [False],
            },
            n_iter=20,
            why="Estado da arte para dados tabulares; mesma família do LightGBM/XGBoost, nativo do scikit-learn.",
        ),
    ]
