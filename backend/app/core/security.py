"""Autenticação por chave de API (header `X-API-Key`).

- Chaves configuradas em API_KEYS ("chave:usuario,..."). O usuário associado é
  registrado no access_log sempre que dados de um cliente são consultados (LGPD).
- Sem chaves configuradas, a API roda em modo desenvolvimento (usuário "dev").
"""
from __future__ import annotations

import hmac

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.app.core.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Chave de acesso à API")


def current_user(api_key: str | None = Security(api_key_header)) -> str:
    keys = get_settings().api_key_map
    if not keys:
        return "dev"
    if api_key:
        for k, user in keys.items():
            if hmac.compare_digest(api_key, k):
                return user
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Chave de API ausente ou inválida",
                        headers={"WWW-Authenticate": "API-Key"})
