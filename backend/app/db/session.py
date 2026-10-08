"""Engine e sessões do SQLAlchemy."""
from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings().database_url, pool_pre_ping=True, future=True)


def get_sessionmaker(url: str | None = None) -> sessionmaker:
    return sessionmaker(bind=get_engine(url), autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """Dependência do FastAPI: uma sessão por requisição."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()
