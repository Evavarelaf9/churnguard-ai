"""Leitura dos dados operacionais no formato esperado pelo pipeline de ML.

Devolve DataFrames com os mesmos nomes e tipos de `ml.src.data.build_analytical_base.load_raw()`,
para que os atributos calculados a partir do banco sejam **idênticos** aos do treino
(mesma função `build_snapshot`, mesmas regras point-in-time).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

QUERIES = {
    "customers": """SELECT customer_id, nome, idade, regiao, tipo_cliente, plano_id, ciclo_contrato,
                           forma_pagamento, renovacao_automatica, canal_aquisicao, data_inicio,
                           data_cancelamento, motivo_cancelamento, status
                    FROM customers {where}""",
    "plans": "SELECT plano_id, plano, preco_base, funcionalidades_disponiveis, usuarios_incluidos FROM plans",
    "usage": """SELECT customer_id, mes_referencia, acessos, sessoes, minutos_uso, funcionalidades_usadas,
                       dias_ativos, ultimo_acesso
                FROM usage_monthly {where} {and_} mes_referencia <= :ref""",
    "payments": """SELECT payment_id, customer_id, mes_referencia, valor_cobrado, data_vencimento, data_pagamento, status
                   FROM payments {where} {and_} data_vencimento <= :ref""",
    "tickets": """SELECT ticket_id, customer_id, data_abertura, categoria, tempo_resolucao_horas, avaliacao, data_resolucao
                  FROM tickets {where} {and_} data_abertura <= :ref""",
    "surveys": """SELECT customer_id, data_resposta, nota_satisfacao
                  FROM surveys {where} {and_} data_resposta <= :ref""",
}

DATE_COLUMNS = {
    "customers": ["data_inicio", "data_cancelamento"],
    "usage": ["mes_referencia", "ultimo_acesso"],
    "payments": ["mes_referencia", "data_vencimento", "data_pagamento"],
    "tickets": ["data_abertura", "data_resolucao"],
    "surveys": ["data_resposta"],
}
FLOAT_COLUMNS = {
    "customers": ["idade"],
    "payments": ["valor_cobrado"],
    "tickets": ["tempo_resolucao_horas", "avaliacao"],
    "plans": ["preco_base"],
}


def load_raw_frames(db: Session, ref: date, customer_ids: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Carrega os dados brutos até a data de referência (opcionalmente para alguns clientes)."""
    frames = {}
    conn = db.connection()
    for name, sql in QUERIES.items():
        if customer_ids is not None and name != "plans":
            where, and_ = "WHERE customer_id IN :ids", "AND"
        else:
            where, and_ = ("", "WHERE") if name != "customers" else ("", "")
        stmt = text(sql.format(where=where, and_=and_))
        params = {"ref": ref}
        if customer_ids is not None and name != "plans":
            stmt = stmt.bindparams(bindparam("ids", expanding=True))
            params["ids"] = list(customer_ids)
        if name in ("customers", "plans"):
            params.pop("ref")
        df = pd.read_sql(stmt, conn, params=params)
        for c in DATE_COLUMNS.get(name, []):
            df[c] = pd.to_datetime(df[c])
        for c in FLOAT_COLUMNS.get(name, []):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        frames[name] = df
    return frames
