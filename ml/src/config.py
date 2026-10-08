"""Configuração central do pipeline de ML do ChurnGuard AI.

Todos os caminhos, datas de referência e parâmetros de negócio ficam aqui,
para que geração de dados, EDA, treino e API usem exatamente as mesmas regras.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent

DATA_DIR = ML_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ML_ROOT / "models"
REPORTS_DIR = ML_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
EDA_FIG_DIR = DOCS_DIR / "images" / "eda"

# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Simulação (dataset sintético)
# --------------------------------------------------------------------------- #
N_CUSTOMERS = 9_000
SIM_START = pd.Timestamp("2022-01-01")
SIM_END = pd.Timestamp("2026-09-30")  # "hoje" no universo do projeto

# --------------------------------------------------------------------------- #
# Definição do problema
# --------------------------------------------------------------------------- #
# Probabilidade de o cliente cancelar nos próximos N dias após a data de referência.
CHURN_HORIZON_DAYS = 60

# Snapshots mensais (fim de mês) usados para montar a base rotulada.
# O último snapshot rotulado precisa ter a janela de 60 dias inteira observada
# antes de SIM_END; por isso termina em 2026-07-31.
FIRST_SNAPSHOT = pd.Timestamp("2024-06-30")
LAST_LABELED_SNAPSHOT = pd.Timestamp("2026-07-31")

# Snapshot "atual": clientes ativos hoje, sem rótulo, que serão pontuados pelo modelo.
SCORING_SNAPSHOT = SIM_END

# Janelas de observação para agregação de comportamento.
USAGE_LAG_MONTHS = 6

# Arquivos gerados
ANALYTICAL_BASE_FILE = PROCESSED_DIR / "analytical_base.parquet"
SCORING_BASE_FILE = PROCESSED_DIR / "scoring_base.parquet"


def labeled_snapshots() -> list[pd.Timestamp]:
    """Datas de referência (fim de mês) com rótulo completo."""
    return list(pd.date_range(FIRST_SNAPSHOT, LAST_LABELED_SNAPSHOT, freq="ME"))

# --------------------------------------------------------------------------- #
# Premissas de negócio (usadas na escolha do threshold) — ajuste à realidade
# --------------------------------------------------------------------------- #
BUSINESS = {
    "custo_acao": 50.0,                # R$ por cliente abordado (tempo do CS + incentivo/desconto)
    "taxa_sucesso_retencao": 0.25,     # % dos clientes que iriam cancelar e são retidos pela ação
    "meses_receita_preservada": 8,     # meses de mensalidade preservados por cliente retido
}

# Faixas de risco (probabilidade estimada de churn em 60 dias).
# Definidas em ml/src/training/train.py a partir do threshold operacional e salvas
# junto com o modelo; os valores abaixo são o padrão caso o modelo não informe.
RISK_THRESHOLDS = {"medio": 0.12, "alto": 0.30}

# Artefatos do modelo
MODEL_FILE = MODELS_DIR / "churn_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
MODEL_FIG_DIR = DOCS_DIR / "images" / "model"
TEST_PREDICTIONS_FILE = PROCESSED_DIR / "test_predictions.parquet"
SCORED_CUSTOMERS_FILE = PROCESSED_DIR / "scored_customers.parquet"
