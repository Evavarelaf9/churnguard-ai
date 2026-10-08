from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.core.security import current_user
from backend.app.db.session import get_db
from backend.app.schemas.schemas import RetentionActionOut, RetentionActionUpdate
from backend.app.services import business, monitoring

actions_router = APIRouter(prefix="/actions", tags=["Ações de retenção"])


@actions_router.get("", response_model=list[RetentionActionOut], summary="Lista ações de retenção")
def list_actions(status: Literal["Planejada", "Em andamento", "Concluída", "Cancelada"] | None = None,
                 limite: int = Query(100, ge=1, le=500),
                 db: Session = Depends(get_db), user: str = Depends(current_user)):
    return business.ActionRepository(db).list(status=status, limit=limite)


@actions_router.patch("/{action_id}", response_model=RetentionActionOut, summary="Atualiza status/resultado da ação")
def update_action(action_id: int, payload: RetentionActionUpdate,
                  db: Session = Depends(get_db), user: str = Depends(current_user)):
    data = payload.model_dump(exclude_unset=True)
    if not data:
        raise HTTPException(422, "Nada para atualizar")
    action = business.update_action(db, action_id, data, user)
    if action is None:
        raise HTTPException(404, f"Ação {action_id} não encontrada")
    return action


dashboard_router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@dashboard_router.get("/summary", summary="Indicadores, distribuições e gráficos do dashboard")
def summary(threshold: float | None = Query(None, ge=0.01, le=0.99, description="Limite de alto risco"),
            limite_medio: float | None = Query(None, ge=0.01, le=0.99, description="Limite de médio risco (padrão: o do modelo)"),
            db: Session = Depends(get_db), user: str = Depends(current_user)):
    return business.dashboard_summary(db, threshold, limite_medio)


meta_router = APIRouter(prefix="/meta", tags=["Dashboard"])


@meta_router.get("/filtros", summary="Opções para os filtros da interface")
def filtros(db: Session = Depends(get_db)):
    return business.filter_options(db)


@meta_router.get("/thresholds", summary="Trade-off precision × recall por threshold",
                 description="Calculado com previsões cujo resultado real já foi observado. Mostra também quantos "
                             "clientes ativos seriam sinalizados hoje em cada threshold.")
def thresholds(db: Session = Depends(get_db)):
    return business.threshold_tradeoff(db)


monitoring_router = APIRouter(prefix="/monitoring", tags=["Monitoramento"])


@monitoring_router.get("/summary", summary="Saúde do modelo: desempenho, drift, calibração e alertas",
                       description="Calculado na hora a partir das previsões registradas. Desempenho usa apenas previsões "
                                   "cujo resultado real (60 dias) já foi observado; drift compara os atributos atuais com "
                                   "os dados de validação do modelo (PSI).")
def monitoring_summary(db: Session = Depends(get_db)):
    return monitoring.compute(db)


@monitoring_router.post("/run", summary="Executa o monitoramento e grava o resultado")
def monitoring_run(db: Session = Depends(get_db), user: str = Depends(current_user)):
    return monitoring.run_and_store(db)


@monitoring_router.get("/runs", summary="Histórico de execuções do monitoramento")
def monitoring_runs(limite: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
    return monitoring.list_runs(db, limite)
