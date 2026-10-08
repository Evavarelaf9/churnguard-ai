"""Testes de integração da API contra um PostgreSQL real (banco de teste separado).

Configure TEST_DATABASE_URL (padrão: postgresql+psycopg://churnguard:churnguard@localhost:5432/churnguard_test).
Na primeira execução o banco de teste é migrado e populado pelo seed (~20 s).
"""
import os

import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL",
                         "postgresql+psycopg://churnguard:churnguard@localhost:5432/churnguard_test")
os.environ["DATABASE_URL"] = TEST_DB
os.environ["API_KEYS"] = ""


def _db_available() -> bool:
    from sqlalchemy import create_engine, text
    try:
        with create_engine(TEST_DB).connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def seeded():
    if not _db_available():
        pytest.skip("PostgreSQL de teste indisponível (defina TEST_DATABASE_URL)")
    from sqlalchemy import text

    from backend.app.db import seed
    from backend.app.db.session import get_engine
    engine = get_engine()
    from backend.app.db.migrate import migrate
    migrate(engine)
    with engine.connect() as c:
        # (re)carrega se vazio ou se o seed é de uma versão anterior
        # (históricos sem 'entrada' ou sem explicação/resumo)
        empty = not c.execute(text("SELECT EXISTS (SELECT 1 FROM predictions WHERE origem = 'historico' "
                                   "AND entrada IS NOT NULL AND principal_fator IS NOT NULL "
                                   "AND fatores ? 'resumo')")).scalar()
    if empty:
        import sys
        sys.argv = ["seed", "--reset"]
        seed.main()
    return engine


@pytest.fixture(scope="session")
def client(seeded):
    from fastapi.testclient import TestClient

    from backend.app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db(seeded):
    from backend.app.db.session import get_sessionmaker
    with get_sessionmaker()() as s:
        yield s


@pytest.fixture(scope="session")
def example_payload():
    from backend.app.schemas.schemas import CustomerFeatures
    return dict(CustomerFeatures.model_config["json_schema_extra"]["example"])
