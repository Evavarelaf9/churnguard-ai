import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.security import current_user
from backend.app.db.session import get_db
from backend.app.repositories.repositories import (
    AccessLogRepository,
    CustomerRepository,
    PredictionRepository,
)
from backend.app.schemas.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    PredictionResult,
)
from backend.app.services.scoring import score_frame, to_prediction_row

router = APIRouter(prefix="/predict", tags=["Previsão"])

THRESHOLD_Q = Query(None, ge=0.01, le=0.99, description="Threshold de alto risco (padrão: o do modelo)")
MEDIO_Q = Query(None, ge=0.01, le=0.99, description="Limite de médio risco (padrão: o do modelo)")


@router.post("", response_model=PredictionResult, summary="Probabilidade de churn de um cliente",
             description="Recebe os dados agregados de um cliente e retorna a **probabilidade estimada** de "
                         "cancelamento em 60 dias, o nível de risco, os principais fatores e uma sugestão de ação. "
                         "Se `customer_id` existir no banco, a previsão é registrada para monitoramento.")
def predict(payload: CustomerFeatures, threshold: float | None = THRESHOLD_Q, limite_medio: float | None = MEDIO_Q,
            db: Session = Depends(get_db), user: str = Depends(current_user)):
    df = pd.DataFrame([payload.model_dump()])
    result = score_frame(df, threshold=threshold, medio=limite_medio)[0]
    result["customer_id"] = payload.customer_id
    if payload.customer_id and CustomerRepository(db).get(payload.customer_id):
        ref = get_settings().today
        row = to_prediction_row(payload.customer_id, ref, df.iloc[0], result, "api")
        result["prediction_id"] = PredictionRepository(db).add_many([row])[0].id
        result["data_referencia"] = ref
        AccessLogRepository(db).log(user, "prever", payload.customer_id)
        db.commit()
    return result


@router.post("/batch", response_model=BatchResponse, summary="Previsão em lote",
             description="Até `BATCH_MAX_SIZE` clientes por requisição. Use `explicar=true` para incluir fatores.")
def predict_batch(payload: BatchRequest, threshold: float | None = THRESHOLD_Q, limite_medio: float | None = MEDIO_Q,
                  user: str = Depends(current_user)):
    max_size = get_settings().batch_max_size
    if len(payload.clientes) > max_size:
        raise HTTPException(413, f"Máximo de {max_size} clientes por lote")
    df = pd.DataFrame([c.model_dump() for c in payload.clientes])
    results = score_frame(df, threshold=threshold, explain=payload.explicar, medio=limite_medio)
    for r, cid in zip(results, df.customer_id):
        r["customer_id"] = cid
    levels = pd.Series([r["nivel_risco"] for r in results])
    return {
        "resumo": {"total": len(results), "alto": int((levels == "Alto").sum()), "medio": int((levels == "Médio").sum()),
                   "baixo": int((levels == "Baixo").sum()),
                   "probabilidade_media": float(sum(r["probabilidade"] for r in results) / len(results))},
        "resultados": results,
    }
