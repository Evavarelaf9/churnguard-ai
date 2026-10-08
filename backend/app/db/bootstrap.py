"""Inicialização completa do ambiente (usada pelo serviço `bootstrap` do Docker Compose).

1. Gera o dataset sintético, se ainda não existir (ml/data/raw).
2. Monta a base analítica point-in-time, se ainda não existir (ml/data/processed).
3. Aplica migrations e carrega o banco (seed), se ele estiver vazio.
4. Executa o monitoramento uma vez, para o painel já abrir com um retrato de saúde do modelo.

Idempotente: pode ser executado várias vezes; só faz o que falta.

Uso:
    python -m backend.app.db.bootstrap [--reset]
"""
from __future__ import annotations

import argparse
import logging
import time

from sqlalchemy import text

from backend.app.db import seed
from backend.app.db.session import get_engine, get_sessionmaker
from ml.src import config as ml

log = logging.getLogger("churnguard.bootstrap")


def wait_for_db(retries: int = 30, delay: float = 2.0) -> None:
    for i in range(retries):
        try:
            with get_engine().connect() as c:
                c.execute(text("SELECT 1"))
            return
        except Exception:  # noqa: BLE001
            log.info("aguardando o banco (%d/%d)…", i + 1, retries)
            time.sleep(delay)
    raise RuntimeError("Banco de dados indisponível")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if not (ml.RAW_DIR / "customers.csv").exists():
        log.info("gerando dataset sintético…")
        from ml.src.data import generate_synthetic
        generate_synthetic.main()
    if not ml.ANALYTICAL_BASE_FILE.exists() or not ml.SCORING_BASE_FILE.exists():
        log.info("montando base analítica…")
        from ml.src.data import build_analytical_base
        build_analytical_base.main()

    wait_for_db()
    loaded = seed.run(reset=args.reset)
    if loaded:
        from backend.app.services.monitoring import run_and_store
        with get_sessionmaker()() as db:
            r = run_and_store(db)
        log.info("monitoramento inicial: status %s", r["status"])
    print("Ambiente pronto.")


if __name__ == "__main__":
    main()
