"""Executor de migrations SQL versionadas (database/migrations/NNN_nome.sql).

Aplica, em ordem, os arquivos ainda não registrados em `schema_migrations`.
Cada arquivo roda em uma transação: ou aplica tudo, ou nada.

Uso:
    python -m backend.app.db.migrate            # aplica pendentes
    python -m backend.app.db.migrate --status   # lista aplicadas/pendentes
"""
from __future__ import annotations

import argparse
import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.app.core.config import get_settings
from backend.app.db.session import get_engine

log = logging.getLogger("churnguard.migrate")


def _ensure_table(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version VARCHAR(100) PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        ))


def applied(engine: Engine) -> set[str]:
    _ensure_table(engine)
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(text("SELECT version FROM schema_migrations"))}


def migrate(engine: Engine | None = None) -> list[str]:
    engine = engine or get_engine()
    done = applied(engine)
    files = sorted(get_settings().migrations_dir.glob("*.sql"))
    new = []
    for f in files:
        if f.name in done:
            continue
        with engine.begin() as conn:
            conn.exec_driver_sql(f.read_text(encoding="utf-8"))
            conn.execute(text("INSERT INTO schema_migrations (version) VALUES (:v)"), {"v": f.name})
        new.append(f.name)
        log.info("migration aplicada: %s", f.name)
    return new


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    engine = get_engine()
    if args.status:
        done = applied(engine)
        for f in sorted(get_settings().migrations_dir.glob("*.sql")):
            print(f"{'✔' if f.name in done else '·'} {f.name}")
        return
    new = migrate(engine)
    print(f"{len(new)} migration(s) aplicada(s)" + (f": {', '.join(new)}" if new else " — banco já atualizado"))


if __name__ == "__main__":
    main()
