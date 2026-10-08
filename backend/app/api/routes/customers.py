from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.core.security import current_user
from backend.app.db.session import get_db
from backend.app.repositories.repositories import AccessLogRepository
from backend.app.schemas.schemas import (
    CustomerDetail,
    CustomerList,
    PredictionResult,
    RetentionActionCreate,
    RetentionActionOut,
    SimulationRequest,
    SimulationResult,
)
from backend.app.services import business
from backend.app.services.scoring import (
    CustomerNotFound,
    CustomerNotScorable,
    score_customer,
    simulate_customer,
)

router = APIRouter(prefix="/customers")
C = ["Clientes"]


def _not_found(cid: str):
    return HTTPException(404, f"Cliente {cid} não encontrado")


@router.get("", response_model=CustomerList, tags=C, summary="Lista clientes ativos com o risco mais recente")
def list_customers(
    nivel_risco: list[Literal["Baixo", "Médio", "Alto"]] | None = Query(None, description="Um ou mais níveis"),
    plano: str | None = None,
    regiao: str | None = None,
    valor_min: float | None = Query(None, ge=0),
    valor_max: float | None = Query(None, ge=0),
    data_referencia: date | None = Query(None, description="Período (data de referência da previsão)"),
    busca: str | None = Query(None, max_length=60, description="Nome ou ID"),
    ordenar: Literal["probabilidade", "receita_em_risco", "valor_mensal", "nome"] = "probabilidade",
    pagina: int = Query(1, ge=1),
    tamanho_pagina: int = Query(25, ge=1, le=200),
    threshold: float | None = Query(None, ge=0.01, le=0.99, description="Limite de alto risco (padrão: o do modelo)"),
    limite_medio: float | None = Query(None, ge=0.01, le=0.99, description="Limite de médio risco (padrão: o do modelo)"),
    db: Session = Depends(get_db), user: str = Depends(current_user),
):
    return business.list_customers(db, threshold=threshold, medio=limite_medio, niveis=nivel_risco, plano=plano, regiao=regiao, valor_min=valor_min,
                                   valor_max=valor_max, data_referencia=data_referencia, busca=busca,
                                   ordenar=ordenar, pagina=pagina, tamanho_pagina=tamanho_pagina)


@router.get("/{customer_id}", response_model=CustomerDetail, tags=C, summary="Detalhes, histórico e previsão do cliente")
def get_customer(customer_id: str, db: Session = Depends(get_db), user: str = Depends(current_user)):
    try:
        return business.customer_detail(db, customer_id, user)
    except CustomerNotFound:
        raise _not_found(customer_id)


@router.get("/{customer_id}/risk", response_model=PredictionResult, tags=C, summary="Calcula o risco atual do cliente",
            description="Monta os atributos a partir do banco (somente dados até a data de referência), "
                        "calcula a probabilidade, explica e registra a previsão.")
def customer_risk(customer_id: str,
                  data_referencia: date | None = Query(None, description="Padrão: hoje"),
                  threshold: float | None = Query(None, ge=0.01, le=0.99),
                  limite_medio: float | None = Query(None, ge=0.01, le=0.99, description="Limite de médio risco (padrão: o do modelo)"),
                  db: Session = Depends(get_db), user: str = Depends(current_user)):
    try:
        result = score_customer(db, customer_id, data_referencia, threshold, medio=limite_medio)
    except CustomerNotFound:
        raise _not_found(customer_id)
    except CustomerNotScorable as exc:
        raise HTTPException(422, str(exc))
    AccessLogRepository(db).log(user, "calcular_risco", customer_id)
    db.commit()
    return result


@router.post("/{customer_id}/simulate", response_model=SimulationResult, tags=C, summary="Simulação de retenção (e se...?)",
             description="Altera sinais hipotéticos (uso, atrasos, chamados, plano...) e compara a probabilidade. "
                         "**É uma simulação**, não uma previsão garantida do comportamento do cliente. Nada é gravado.")
def simulate(customer_id: str, payload: SimulationRequest, threshold: float | None = Query(None, ge=0.01, le=0.99),
             limite_medio: float | None = Query(None, ge=0.01, le=0.99, description="Limite de médio risco (padrão: o do modelo)"),
             db: Session = Depends(get_db), user: str = Depends(current_user)):
    changes = payload.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(422, "Informe ao menos uma alteração para simular")
    try:
        return simulate_customer(db, customer_id, changes, threshold=threshold, medio=limite_medio)
    except CustomerNotFound:
        raise _not_found(customer_id)
    except CustomerNotScorable as exc:
        raise HTTPException(422, str(exc))


@router.get("/{customer_id}/actions", response_model=list[RetentionActionOut], tags=["Ações de retenção"],
            summary="Ações de retenção do cliente")
def customer_actions(customer_id: str, db: Session = Depends(get_db), user: str = Depends(current_user)):
    return business.ActionRepository(db).list(customer_id=customer_id)


@router.post("/{customer_id}/actions", response_model=RetentionActionOut, status_code=201,
             tags=["Ações de retenção"], summary="Registra uma ação de retenção (decisão humana)")
def create_action(customer_id: str, payload: RetentionActionCreate,
                  db: Session = Depends(get_db), user: str = Depends(current_user)):
    try:
        return business.create_action(db, customer_id, payload.model_dump(), user)
    except CustomerNotFound:
        raise _not_found(customer_id)
