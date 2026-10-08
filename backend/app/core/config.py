"""Configurações da aplicação (variáveis de ambiente / arquivo .env)."""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ChurnGuard AI API"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://churnguard:churnguard@localhost:5432/churnguard"
    model_path: Path = PROJECT_ROOT / "ml" / "models" / "churn_model.joblib"
    migrations_dir: Path = PROJECT_ROOT / "database" / "migrations"

    # Data de referência "hoje". No universo sintético do projeto, 30/09/2026.
    # Em produção, deixar vazio para usar a data do sistema.
    reference_date: date | None = date(2026, 9, 30)

    # Autenticação por chave de API. Formato: "chave1:usuario1,chave2:usuario2".
    # Vazio = autenticação desativada (apenas desenvolvimento local).
    api_keys: str = ""

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])
    batch_max_size: int = 1000
    log_level: str = "INFO"

    @property
    def today(self) -> date:
        return self.reference_date or date.today()

    @property
    def api_key_map(self) -> dict[str, str]:
        pairs = [p.strip() for p in self.api_keys.split(",") if p.strip()]
        return {k: (u or "api") for k, _, u in (p.partition(":") for p in pairs)}


@lru_cache
def get_settings() -> Settings:
    return Settings()
