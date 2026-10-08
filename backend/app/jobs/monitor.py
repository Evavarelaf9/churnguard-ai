"""Job de monitoramento: calcula desempenho, drift e alertas e grava em monitoring_runs.

Sequência sugerida (agendada, ex.: diariamente):
    python -m backend.app.jobs.update_outcomes   # registra resultados reais vencidos
    python -m backend.app.jobs.score_customers   # pontua a carteira atual
    python -m backend.app.jobs.monitor           # avalia a saúde do modelo

Código de saída 2 quando o status é crítico (útil para acionar alertas no agendador/CI).
"""
from __future__ import annotations

import logging
import sys

from backend.app.db.session import get_sessionmaker
from backend.app.services.monitoring import run_and_store

log = logging.getLogger("churnguard.jobs")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with get_sessionmaker()() as db:
        r = run_and_store(db)
    print(f"Status do modelo {r['model_version']}: {r['status'].upper()} ({len(r['alertas'])} alerta(s))")
    for a in r["alertas"]:
        print(f"  [{a['nivel']}] {a['titulo']}: {a['detalhe']}")
    return 2 if r["status"] == "crítico" else 0


if __name__ == "__main__":
    sys.exit(main())
