from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.session import get_db
from backend.app.schemas.schemas import Health, ModelInfo, ModelMetrics
from backend.app.services import business
from backend.app.services.scoring import get_scorer

router = APIRouter()


@router.get("/health", response_model=Health, tags=["Sistema"], summary="Verifica API, banco e modelo")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # noqa: BLE001
        db_status = f"erro: {type(exc).__name__}"
    try:
        version, model_status = get_scorer().version, "ok"
    except Exception as exc:  # noqa: BLE001
        version, model_status = None, f"erro: {type(exc).__name__}"
    return {
        "status": "ok" if db_status == model_status == "ok" else "degradado",
        "database": db_status, "model": model_status, "model_version": version,
        "data_referencia": get_settings().today, "timestamp": datetime.now(timezone.utc),
    }


model_router = APIRouter(prefix="/model", tags=["Modelo"])


@model_router.get("/info", response_model=ModelInfo, summary="Versão, parâmetros, thresholds e premissas do modelo")
def info():
    return business.model_info()


@model_router.get("/metrics", response_model=ModelMetrics, summary="Métricas do modelo no período de teste")
def metrics():
    return business.model_metrics()
