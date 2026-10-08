"""ChurnGuard AI — API REST.

Execução local (na raiz do projeto):
    python -m uvicorn backend.app.main:app --reload
Documentação interativa: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import customers, operations, predict, system
from backend.app.core.config import get_settings
from backend.app.core.security import current_user
from backend.app.services.scoring import get_scorer

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("churnguard.api")

DESCRIPTION = """
**Inteligência Artificial para prever cancelamentos antes que eles aconteçam.**

A API expõe o modelo de churn do ChurnGuard AI: probabilidade estimada de cancelamento em 60 dias,
nível de risco, explicação em linguagem de negócio, simulação de retenção e registro de ações.

⚠️ As previsões são **probabilidades estimadas**, não certezas. Recomendações são sugestões baseadas em
dados — a decisão final é sempre humana. Consultas a dados de clientes são registradas (LGPD).

**Autenticação:** header `X-API-Key` (desativada em desenvolvimento quando `API_KEYS` está vazio).
"""

TAGS = [
    {"name": "Previsão", "description": "Pontuação de clientes a partir de dados enviados na requisição."},
    {"name": "Clientes", "description": "Consulta de clientes, risco calculado a partir do banco e simulação."},
    {"name": "Ações de retenção", "description": "Registro das decisões humanas tomadas a partir das previsões."},
    {"name": "Dashboard", "description": "Dados agregados para o painel."},
    {"name": "Modelo", "description": "Versão, parâmetros e métricas do modelo."},
    {"name": "Monitoramento", "description": "Desempenho ao longo do tempo, drift dos dados e alertas."},
    {"name": "Sistema", "description": "Saúde da aplicação."},
]

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Carrega o modelo na inicialização: a primeira requisição não paga o custo.
    try:
        scorer = get_scorer()
        log.info("modelo carregado: %s", scorer.version)
    except Exception:  # noqa: BLE001 — a API sobe em modo degradado e o /health informa
        log.exception("falha ao carregar o modelo")
    yield


app = FastAPI(lifespan=lifespan, title=settings.app_name, version="1.0.0", description=DESCRIPTION, openapi_tags=TAGS,
              contact={"name": "ChurnGuard AI"}, license_info={"name": "MIT"})

app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "PATCH"], allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code, ms)
    return response


@app.exception_handler(ValueError)
async def value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# Toda a API versionada exige autenticação; /health na raiz fica público para orquestradores.
api = APIRouter(prefix="/api/v1", dependencies=[Depends(current_user)])
api.include_router(predict.router)
api.include_router(customers.router)
api.include_router(operations.actions_router)
api.include_router(operations.dashboard_router)
api.include_router(operations.meta_router)
api.include_router(operations.monitoring_router)
api.include_router(system.model_router)
api.include_router(system.router)  # /api/v1/health
app.include_router(api)
app.include_router(system.router)  # /health (orquestradores / Docker)


@app.get("/", include_in_schema=False)
def root():
    return {"app": settings.app_name, "docs": "/docs", "health": "/health"}
