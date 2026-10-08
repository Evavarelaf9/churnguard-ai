"""Construção da base analítica (uma linha por cliente × data de referência).

Cada linha responde à pergunta:
    "Com o que sabíamos em `data_referencia`, este cliente cancelou nos
     próximos CHURN_HORIZON_DAYS dias?"

Regras anti-vazamento (point-in-time correctness)
-------------------------------------------------
1. Só entram clientes ATIVOS na data de referência (sem cancelamento até ela).
2. Uso: apenas meses <= mês da data de referência.
3. Faturas: apenas vencimentos <= data de referência. O atraso é medido como
   ele era visível naquele dia (uma fatura paga depois da referência conta
   como "em aberto" naquele momento).
4. Chamados: apenas abertos até a referência; tempo de resolução só é usado se o
   chamado já estava resolvido naquela data.
5. `data_cancelamento` e `motivo_cancelamento` NUNCA viram atributos — só são
   usados para construir o rótulo. O motivo é registrado depois do evento.

Uso:
    python -m ml.src.data.build_analytical_base
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.src import config

LEAKAGE_COLUMNS = ["data_cancelamento", "motivo_cancelamento", "status"]


def load_raw() -> dict[str, pd.DataFrame]:
    r = config.RAW_DIR
    customers = pd.read_csv(r / "customers.csv", parse_dates=["data_inicio", "data_cancelamento"])
    customers = customers.drop_duplicates()  # 25 duplicatas exatas vindas do "CRM"
    return {
        "customers": customers,
        "plans": pd.read_csv(r / "plans.csv"),
        "usage": pd.read_csv(r / "usage_monthly.csv", parse_dates=["mes_referencia", "ultimo_acesso"]),
        "payments": pd.read_csv(r / "payments.csv", parse_dates=["mes_referencia", "data_vencimento", "data_pagamento"]),
        "tickets": pd.read_csv(r / "tickets.csv", parse_dates=["data_abertura", "data_resolucao"]),
        "surveys": pd.read_csv(r / "surveys.csv", parse_dates=["data_resposta"]),
    }


def _usage_features(usage: pd.DataFrame, ids: pd.Index, ref: pd.Timestamp) -> pd.DataFrame:
    ref_month = ref.to_period("M").to_timestamp()
    start = ref_month - pd.DateOffset(months=config.USAGE_LAG_MONTHS - 1)
    u = usage[(usage.mes_referencia >= start) & (usage.mes_referencia <= ref_month) & usage.customer_id.isin(ids)].copy()
    # lag 1 = mês da referência, lag 6 = cinco meses antes
    u["lag"] = (ref_month.year - u.mes_referencia.dt.year) * 12 + (ref_month.month - u.mes_referencia.dt.month) + 1

    out = pd.DataFrame(index=ids)
    wide = u.pivot_table(index="customer_id", columns="lag", values="acessos", aggfunc="sum")
    for lag in range(1, config.USAGE_LAG_MONTHS + 1):
        out[f"acessos_m{lag}"] = wide.get(lag)

    last = u[u.lag == 1].set_index("customer_id")
    out["sessoes_m1"] = last.sessoes
    out["minutos_uso_m1"] = last.minutos_uso
    out["funcionalidades_usadas_m1"] = last.funcionalidades_usadas
    out["dias_ativos_m1"] = last.dias_ativos
    out["minutos_uso_m2"] = u[u.lag == 2].set_index("customer_id").minutos_uso

    hist = usage[(usage.mes_referencia <= ref_month) & usage.customer_id.isin(ids)]
    last_access = hist.groupby("customer_id").ultimo_acesso.max()
    out["dias_desde_ultimo_acesso"] = (ref - last_access).dt.days
    return out


def _payment_features(pay: pd.DataFrame, ids: pd.Index, ref: pd.Timestamp) -> pd.DataFrame:
    p = pay[(pay.data_vencimento <= ref) & pay.customer_id.isin(ids)].copy()
    paid_by_ref = p.data_pagamento.notna() & (p.data_pagamento <= ref)
    effective = p.data_pagamento.where(paid_by_ref, ref)
    p["dias_atraso"] = (effective - p.data_vencimento).dt.days.clip(lower=0)
    p["atrasada"] = p.dias_atraso > 0
    p["em_aberto"] = ~paid_by_ref
    p["valor_pago"] = np.where(paid_by_ref, p.valor_cobrado, 0.0)

    recent = p[p.data_vencimento > ref - pd.DateOffset(months=6)]
    g = recent.groupby("customer_id")
    out = pd.DataFrame(index=ids)
    out["faturas_6m"] = g.size()
    out["faturas_atrasadas_6m"] = g.atrasada.sum()
    out["dias_atraso_medio_6m"] = g.dias_atraso.mean()
    out["faturas_em_aberto"] = p[p.em_aberto & (p.dias_atraso > 5)].groupby("customer_id").size()
    out["valor_total_pago"] = p.groupby("customer_id").valor_pago.sum()
    out["qtd_cobrancas_total"] = p.groupby("customer_id").size()

    # Valor mensal vigente e reajuste recente (comparando última fatura com a de 6 meses antes).
    last_two = p.sort_values("data_vencimento").groupby("customer_id").valor_cobrado
    out["valor_mensal"] = last_two.last()
    older = p[p.data_vencimento <= ref - pd.DateOffset(months=5)].sort_values("data_vencimento")
    prev = older.groupby("customer_id").valor_cobrado.last()
    out["reajuste_pct_6m"] = ((out.valor_mensal / prev) - 1).clip(lower=0).round(4)
    for c in ["faturas_6m", "faturas_atrasadas_6m", "faturas_em_aberto", "qtd_cobrancas_total", "valor_total_pago"]:
        out[c] = out[c].fillna(0)
    return out


def _ticket_features(tk: pd.DataFrame, ids: pd.Index, ref: pd.Timestamp) -> pd.DataFrame:
    t = tk[(tk.data_abertura <= ref) & tk.customer_id.isin(ids)].copy()
    t6 = t[t.data_abertura > ref - pd.DateOffset(months=6)]
    t3 = t[t.data_abertura > ref - pd.DateOffset(months=3)]
    resolved = t6[t6.data_resolucao <= ref]
    out = pd.DataFrame(index=ids)
    out["tickets_3m"] = t3.groupby("customer_id").size()
    out["tickets_6m"] = t6.groupby("customer_id").size()
    out["tickets_tecnicos_6m"] = t6[t6.categoria == "Técnico"].groupby("customer_id").size()
    out["reclamacoes_6m"] = t6[t6.categoria == "Reclamação"].groupby("customer_id").size()
    out["tempo_medio_resolucao_h_6m"] = resolved.groupby("customer_id").tempo_resolucao_horas.mean()
    rated = resolved[resolved.avaliacao.notna()]
    out["avaliacao_media_6m"] = rated.groupby("customer_id").avaliacao.mean()
    out["dias_desde_ultimo_chamado"] = (ref - t.groupby("customer_id").data_abertura.max()).dt.days
    for c in ["tickets_3m", "tickets_6m", "tickets_tecnicos_6m", "reclamacoes_6m"]:
        out[c] = out[c].fillna(0).astype(int)
    return out


def _survey_features(sv: pd.DataFrame, ids: pd.Index, ref: pd.Timestamp) -> pd.DataFrame:
    s = sv[(sv.data_resposta <= ref) & (sv.data_resposta > ref - pd.DateOffset(months=12)) & sv.customer_id.isin(ids)]
    s = s.sort_values("data_resposta").groupby("customer_id").last()
    return pd.DataFrame({"ultima_nota_satisfacao": s.nota_satisfacao}, index=ids)


def build_snapshot(raw: dict[str, pd.DataFrame], ref: pd.Timestamp, labeled: bool = True) -> pd.DataFrame:
    c = raw["customers"]
    # Ativo na referência e com pelo menos 1 mês completo de histórico.
    alive = (c.data_inicio <= ref - pd.Timedelta(days=30)) & (c.data_cancelamento.isna() | (c.data_cancelamento > ref))
    base = c.loc[alive].set_index("customer_id")
    ids = base.index

    plans = raw["plans"].set_index("plano_id")
    df = pd.DataFrame(index=ids)
    df["data_referencia"] = ref
    df["nome"] = base.nome
    df["idade"] = base.idade
    df["regiao"] = base.regiao
    df["tipo_cliente"] = base.tipo_cliente
    df["plano"] = plans.loc[base.plano_id, "plano"].to_numpy()
    df["funcionalidades_disponiveis"] = plans.loc[base.plano_id, "funcionalidades_disponiveis"].to_numpy()
    df["ciclo_contrato"] = base.ciclo_contrato
    df["forma_pagamento"] = base.forma_pagamento
    df["renovacao_automatica"] = base.renovacao_automatica.astype(int)
    df["canal_aquisicao"] = base.canal_aquisicao
    df["tempo_cliente_meses"] = ((ref - base.data_inicio).dt.days / 30.44).round(1)

    df = df.join(_usage_features(raw["usage"], ids, ref))
    df = df.join(_payment_features(raw["payments"], ids, ref))
    df = df.join(_ticket_features(raw["tickets"], ids, ref))
    df = df.join(_survey_features(raw["surveys"], ids, ref))

    if labeled:
        horizon_end = ref + pd.Timedelta(days=config.CHURN_HORIZON_DAYS)
        cd = base.data_cancelamento
        df["churn"] = ((cd > ref) & (cd <= horizon_end)).astype(int)
    return df.reset_index()


def build_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw()
    frames = []
    for ref in config.labeled_snapshots():
        snap = build_snapshot(raw, ref, labeled=True)
        frames.append(snap)
        print(f"  {ref.date()}  clientes ativos={len(snap):>5,}  churn_{config.CHURN_HORIZON_DAYS}d={snap.churn.mean():.2%}")
    base = pd.concat(frames, ignore_index=True)
    scoring = build_snapshot(raw, config.SCORING_SNAPSHOT, labeled=False)
    return base, scoring


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    base, scoring = build_all()
    base.to_parquet(config.ANALYTICAL_BASE_FILE, index=False)
    scoring.to_parquet(config.SCORING_BASE_FILE, index=False)
    print(f"\nBase analítica: {len(base):,} linhas × {base.shape[1]} colunas | churn médio {base.churn.mean():.2%}")
    print(f"Base de scoring ({config.SCORING_SNAPSHOT.date()}): {len(scoring):,} clientes ativos")


if __name__ == "__main__":
    main()
