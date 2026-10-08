"""Job de pontuação em lote: calcula o risco de todos os clientes ativos.

Em produção, roda periodicamente (ex.: todo dia 1º ou semanalmente) via agendador.
Usa o modelo ativo e grava uma previsão por cliente com origem 'batch'.

Uso:
    python -m backend.app.jobs.score_customers [--data 2026-09-30]
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.session import get_sessionmaker
from backend.app.repositories.repositories import PredictionRepository
from backend.app.services.scoring import (
    build_features_from_db,
    score_frame,
    to_prediction_row,
)

log = logging.getLogger("churnguard.jobs")


def run(db: Session, ref: date | None = None, chunk: int = 1000) -> int:
    ref = ref or get_settings().today
    t0 = time.time()
    feats = build_features_from_db(db, ref)
    if feats.empty:
        return 0
    # Re-executar o job para a mesma data substitui as previsões em lote anteriores.
    db.execute(text("DELETE FROM predictions WHERE origem = 'batch' AND data_referencia = :ref"), {"ref": ref})
    repo = PredictionRepository(db)
    total = 0
    for start in range(0, len(feats), chunk):
        part = feats.iloc[start:start + chunk]
        results = score_frame(part)
        rows = [to_prediction_row(cid, ref, f, r, "batch")
                for cid, (_, f), r in zip(part.customer_id, part.iterrows(), results)]
        repo.add_many(rows)
        total += len(rows)
    db.commit()
    log.info("pontuação em lote: %d clientes em %.1fs (referência %s)", total, time.time() - t0, ref)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    with get_sessionmaker()() as db:
        n = run(db, args.data)
    print(f"{n} clientes pontuados")


if __name__ == "__main__":
    main()
