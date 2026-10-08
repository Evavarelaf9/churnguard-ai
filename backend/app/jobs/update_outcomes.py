"""Job que registra o resultado real das previsões cujo horizonte já terminou.

Para cada previsão sem resultado e com data_referencia + 60 dias ≤ hoje:
    churn_real = 1 se o cliente cancelou dentro da janela (referência, referência + 60 dias]
    churn_real = 0 caso contrário
Isso alimenta a comparação "o que o modelo previu" × "o que realmente aconteceu".

Uso:
    python -m backend.app.jobs.update_outcomes
"""
from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.session import get_sessionmaker
from ml.src.config import CHURN_HORIZON_DAYS

log = logging.getLogger("churnguard.jobs")

SQL = text("""
    UPDATE predictions p
    SET churn_real = CASE WHEN c.data_cancelamento > p.data_referencia
                               AND c.data_cancelamento <= p.data_referencia + :h THEN 1 ELSE 0 END,
        resultado_em = p.data_referencia + :h
    FROM customers c
    WHERE c.customer_id = p.customer_id
      AND p.churn_real IS NULL
      AND p.data_referencia + :h <= :today
""")


def run(db: Session, today: date | None = None) -> int:
    today = today or get_settings().today
    n = db.execute(SQL, {"h": CHURN_HORIZON_DAYS, "today": today}).rowcount
    db.commit()
    log.info("resultados observados registrados: %d previsões", n)
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with get_sessionmaker()() as db:
        print(f"{run(db)} previsões atualizadas com o resultado real")
