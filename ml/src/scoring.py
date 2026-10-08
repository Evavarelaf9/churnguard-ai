"""Interface de inferência do modelo — usada pela API e pelos testes.

    scorer = ChurnScorer.load()
    result = scorer.predict(df_clientes)              # probabilidade + nível de risco
    result = scorer.explain(df_clientes, top_k=4)     # + fatores e recomendação
    scorer.predict(df, threshold=0.30)                # threshold configurável

O modelo devolve uma **probabilidade estimada** de cancelamento nos próximos
60 dias. O nível de risco e a recomendação são apoio à decisão humana.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from ml.src import config
from ml.src.evaluation.metrics import classify_risk
from ml.src.features.build_features import RAW_INPUT_COLUMNS


@dataclass
class ChurnScorer:
    pipeline: object
    metadata: dict
    background: pd.DataFrame | None = None
    _explainer: object = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path | str = config.MODEL_FILE) -> "ChurnScorer":
        bundle = joblib.load(path)
        return cls(bundle["pipeline"], bundle["metadata"], bundle.get("background"))

    @property
    def risk_thresholds(self) -> dict:
        return self.metadata.get("risk_thresholds", config.RISK_THRESHOLDS)

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @staticmethod
    def validate(df: pd.DataFrame) -> None:
        missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    def thresholds(self, threshold: float | None = None, medio: float | None = None) -> dict:
        """Faixas de risco efetivas: as do modelo, com os limites informados sobrepostos.

        `threshold` redefine o limite de alto risco; `medio`, o de médio risco.
        O limite médio nunca fica acima do alto.
        """
        th = dict(self.risk_thresholds)
        if threshold is not None:
            th["alto"] = float(threshold)
        if medio is not None:
            th["medio"] = float(medio)
        th["medio"] = min(th["medio"], th["alto"])
        return th

    def predict(self, df: pd.DataFrame, threshold: float | None = None, medio: float | None = None) -> pd.DataFrame:
        self.validate(df)
        proba = self.pipeline.predict_proba(df)[:, 1]
        thresholds = self.thresholds(threshold, medio)
        return pd.DataFrame(
            {
                "probabilidade": proba,
                "nivel_risco": classify_risk(proba, thresholds),
                "acima_do_threshold": proba >= thresholds["alto"],
            },
            index=df.index,
        )

    def explain(self, df: pd.DataFrame, top_k: int = 4) -> list[dict]:
        self.validate(df)
        if self._explainer is None:
            from ml.src.explainability.explainer import ChurnExplainer

            self._explainer = ChurnExplainer(self.pipeline, background=self.background,
                                             risk_thresholds=self.risk_thresholds)
        return self._explainer.explain(df, top_k=top_k)


def simulate(scorer: ChurnScorer, customer: pd.DataFrame, changes: dict) -> dict:
    """Simulação "e se": altera atributos hipotéticos e compara a probabilidade.

    Não é uma previsão garantida do comportamento futuro — mostra apenas como
    o modelo reagiria se esses sinais fossem diferentes.
    """
    before = scorer.predict(customer).iloc[0]
    modified = customer.copy()
    for k, v in changes.items():
        if k not in modified.columns:
            raise ValueError(f"Atributo desconhecido para simulação: {k}")
        modified[k] = v
    after = scorer.predict(modified).iloc[0]
    return {
        "probabilidade_atual": float(before.probabilidade),
        "probabilidade_simulada": float(after.probabilidade),
        "variacao_pontos_percentuais": float((after.probabilidade - before.probabilidade) * 100),
        "nivel_atual": before.nivel_risco,
        "nivel_simulado": after.nivel_risco,
        "aviso": "Simulação hipotética: indica como o modelo reagiria, não garante o comportamento do cliente.",
    }
