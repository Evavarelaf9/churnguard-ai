"""Gerador do dataset sintético do ChurnGuard AI.

Simula uma empresa SaaS brasileira com planos mensais entre 2022-01 e 2026-09.
A simulação é feita mês a mês: cada cliente tem traços latentes (engajamento,
fragilidade financeira, sensibilidade a preço, satisfação) que evoluem no tempo
e determinam o que fica registrado nas tabelas operacionais — uso, faturas,
chamados, pesquisas — e, por fim, o cancelamento.

Por que sintético?
    Bases públicas de churn (ex.: Telco/IBM) são fotografias estáticas, sem
    histórico de uso, chamados ou pagamentos ao longo do tempo. Sem histórico
    não dá para criar atributos de *mudança de comportamento*, fazer divisão
    temporal honesta nem simular monitoramento de drift. O gerador foi desenhado
    para reproduzir padrões documentados na literatura de retenção (queda de uso
    antes do cancelamento, efeito de reajuste, inadimplência, atrito com suporte).

Importante:
    - Os traços latentes NÃO são exportados. O modelo só enxerga o que uma empresa
      real teria registrado em seus sistemas.
    - Região e idade não entram no risco de cancelamento por construção. Isso é
      usado depois para checar se o modelo não aprende correlações espúrias.
    - Alguns problemas de qualidade são injetados de propósito (ausentes,
      duplicados, outliers, erros de digitação) para que a etapa de preparação
      tenha trabalho real a fazer.

Uso:
    python -m ml.src.data.generate_synthetic
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.src import config

# --------------------------------------------------------------------------- #
# Catálogos
# --------------------------------------------------------------------------- #
PLANS = pd.DataFrame(
    {
        "plano_id": [1, 2, 3, 4],
        "plano": ["Básico", "Essencial", "Profissional", "Empresarial"],
        "preco_base": [49.90, 99.90, 199.90, 499.90],
        "funcionalidades_disponiveis": [5, 9, 14, 20],
        "usuarios_incluidos": [1, 3, 10, 50],
        # Uso esperado (acessos/mês) de um cliente com engajamento médio.
        "acessos_referencia": [28, 45, 85, 190],
    }
)

# Reajuste anual aplicado em janeiro para clientes com 6+ meses de casa.
PRICE_ADJUSTMENTS = pd.DataFrame(
    {
        "ano": [2023, 2024, 2025, 2026],
        "percentual": [0.06, 0.05, 0.07, 0.09],
        "descricao": [
            "Reajuste anual (IPCA + ajuste)",
            "Reajuste anual (IPCA)",
            "Reajuste anual (IPCA + novas funcionalidades)",
            "Reajuste anual (reposicionamento de preço)",
        ],
    }
)

REGIONS = ["Sudeste", "Nordeste", "Sul", "Centro-Oeste", "Norte"]
REGION_P = [0.44, 0.22, 0.16, 0.10, 0.08]

CUSTOMER_TYPES = ["Pessoa Física", "Pequena Empresa", "Média Empresa"]
CUSTOMER_TYPE_P = [0.55, 0.35, 0.10]

PLAN_BY_TYPE = {
    "Pessoa Física": [0.47, 0.38, 0.15, 0.00],
    "Pequena Empresa": [0.15, 0.40, 0.35, 0.10],
    "Média Empresa": [0.00, 0.15, 0.45, 0.40],
}

PAYMENT_METHODS = ["Cartão de crédito", "Boleto", "Pix", "Débito automático"]
PAYMENT_BY_TYPE = {
    "Pessoa Física": [0.52, 0.18, 0.20, 0.10],
    "Pequena Empresa": [0.38, 0.32, 0.16, 0.14],
    "Média Empresa": [0.25, 0.45, 0.10, 0.20],
}

CHANNELS = ["Orgânico", "Indicação", "Anúncios pagos", "Parceiros", "Vendas diretas"]
CHANNEL_P = [0.30, 0.18, 0.30, 0.10, 0.12]
CHANNEL_ENGAGEMENT = {  # efeito do canal no engajamento de base
    "Orgânico": 0.05,
    "Indicação": 0.35,
    "Anúncios pagos": -0.30,
    "Parceiros": 0.10,
    "Vendas diretas": 0.20,
}

FIRST_NAMES = [
    "Ana", "Bruno", "Camila", "Diego", "Eduarda", "Felipe", "Gabriela", "Henrique",
    "Isabela", "João", "Karina", "Lucas", "Mariana", "Nicolas", "Olívia", "Pedro",
    "Rafaela", "Samuel", "Tatiane", "Vinícius", "Beatriz", "Caio", "Débora", "Enzo",
    "Fernanda", "Gustavo", "Helena", "Igor", "Juliana", "Leonardo", "Letícia", "Mateus",
    "Natália", "Otávio", "Paula", "Renato", "Sofia", "Thiago", "Valéria", "Yasmin",
]
LAST_NAMES = [
    "Almeida", "Barbosa", "Cardoso", "Dias", "Esteves", "Ferreira", "Gomes", "Haddad",
    "Irala", "Jardim", "Lima", "Moreira", "Nogueira", "Oliveira", "Pereira", "Queiroz",
    "Ribeiro", "Santos", "Teixeira", "Vasconcelos", "Azevedo", "Borges", "Castro",
    "Duarte", "Farias", "Guimarães", "Martins", "Rocha", "Siqueira", "Vieira",
]
COMPANY_WORDS = [
    "Aurora", "Horizonte", "Vértice", "Nexo", "Atlas", "Prisma", "Delta", "Órbita",
    "Norte", "Raiz", "Farol", "Ponte", "Cume", "Trilha", "Âncora", "Ciclo", "Bússola",
    "Fênix", "Lume", "Vetor",
]
COMPANY_SECTORS = [
    "Contabilidade", "Logística", "Consultoria", "Tecnologia", "Educação", "Saúde",
    "Engenharia", "Comércio", "Alimentos", "Design", "Marketing", "Imóveis",
]
COMPANY_SUFFIX = ["Ltda", "ME", "EIRELI", "S.A."]

CANCEL_REASONS = {
    "low_usage": "Baixa utilização",
    "price": "Preço / reajuste",
    "technical": "Problemas técnicos",
    "support": "Insatisfação com atendimento",
    "delinquency": "Inadimplência",
    "competitor": "Migrou para concorrente",
    "other": "Outros motivos",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _month_range() -> pd.DatetimeIndex:
    return pd.date_range(config.SIM_START, config.SIM_END, freq="MS")


# --------------------------------------------------------------------------- #
# 1. Clientes (atributos estáticos + traços latentes)
# --------------------------------------------------------------------------- #
@dataclass
class Latent:
    engagement_base: np.ndarray
    activated: np.ndarray
    fin_fragility: np.ndarray
    price_sensitivity: np.ndarray
    satisfaction_base: np.ndarray
    competitor_affinity: np.ndarray


def _generate_customers(rng: np.random.Generator) -> tuple[pd.DataFrame, Latent]:
    n = config.N_CUSTOMERS
    months = _month_range()[:-1]  # sem aquisição no último mês simulado

    # Crescimento da base: mais aquisições em meses recentes, com sazonalidade leve.
    t = np.arange(len(months))
    weights = np.exp(t / 36) * (1 + 0.15 * np.isin(months.month, [3, 8, 10]))
    weights /= weights.sum()
    signup_month = rng.choice(months, size=n, p=weights)
    signup_day = np.array(
        [rng.integers(1, calendar.monthrange(m.year, m.month)[1] + 1) for m in pd.to_datetime(signup_month)]
    )
    signup = pd.to_datetime(signup_month) + pd.to_timedelta(signup_day - 1, unit="D")

    tipo = rng.choice(CUSTOMER_TYPES, size=n, p=CUSTOMER_TYPE_P)
    plano_id = np.array([rng.choice(PLANS.plano_id, p=PLAN_BY_TYPE[x]) for x in tipo])
    forma_pag = np.array([rng.choice(PAYMENT_METHODS, p=PAYMENT_BY_TYPE[x]) for x in tipo])
    regiao = rng.choice(REGIONS, size=n, p=REGION_P)
    canal = rng.choice(CHANNELS, size=n, p=CHANNEL_P)

    # Contrato anual (fidelidade 12 meses, 15% de desconto) mais comum em planos maiores.
    p_annual = np.select([plano_id == 1, plano_id == 2, plano_id == 3], [0.18, 0.30, 0.42], 0.60)
    ciclo = np.where(rng.random(n) < p_annual, "Anual", "Mensal")

    # Renovação automática: mais comum em cartão/débito.
    p_auto = np.where(np.isin(forma_pag, ["Cartão de crédito", "Débito automático"]), 0.82, 0.48)
    renov_auto = rng.random(n) < p_auto

    # Idade do titular da conta.
    idade = np.clip(rng.normal(np.where(tipo == "Pessoa Física", 36, 43), 11), 18, 79).round()

    # Nomes fictícios.
    nomes = []
    for tp in tipo:
        if tp == "Pessoa Física":
            nomes.append(f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}")
        else:
            nomes.append(
                f"{rng.choice(COMPANY_WORDS)} {rng.choice(COMPANY_SECTORS)} {rng.choice(COMPANY_SUFFIX)}"
            )

    # Traços latentes (não exportados).
    eng = rng.normal(0, 0.8, n) + np.vectorize(CHANNEL_ENGAGEMENT.get)(canal)
    activated = rng.random(n) < np.where(canal == "Anúncios pagos", 0.72, 0.88)
    fin = rng.beta(2, 7, n) + 0.10 * (tipo == "Pessoa Física") + 0.08 * (forma_pag == "Boleto")
    price_sens = rng.beta(2, 4, n) + 0.15 * (plano_id == 1) + 0.10 * (tipo == "Pessoa Física")
    sat = rng.normal(0.1, 0.7, n)
    competitor = rng.beta(1.5, 8, n)

    customers = pd.DataFrame(
        {
            "customer_id": [f"CLI-{i:05d}" for i in range(1, n + 1)],
            "nome": nomes,
            "idade": idade,
            "regiao": regiao,
            "tipo_cliente": tipo,
            "plano_id": plano_id,
            "ciclo_contrato": ciclo,
            "forma_pagamento": forma_pag,
            "renovacao_automatica": renov_auto,
            "canal_aquisicao": canal,
            "data_inicio": signup.normalize(),
        }
    )
    latent = Latent(eng, activated, fin, price_sens, sat, competitor)
    return customers, latent


# --------------------------------------------------------------------------- #
# 2. Simulação mensal
# --------------------------------------------------------------------------- #
def _simulate(customers: pd.DataFrame, lat: Latent, rng: np.random.Generator):
    n = len(customers)
    plans = PLANS.set_index("plano_id")
    plano_id = customers.plano_id.to_numpy()
    ref_usage = plans.loc[plano_id, "acessos_referencia"].to_numpy(float)
    n_features = plans.loc[plano_id, "funcionalidades_disponiveis"].to_numpy()
    discount = np.where(customers.ciclo_contrato.eq("Anual"), 0.85, 1.0)
    price = plans.loc[plano_id, "preco_base"].to_numpy() * discount
    signup = customers.data_inicio.to_numpy("datetime64[D]")
    signup_ms = customers.data_inicio.dt.to_period("M").dt.to_timestamp().to_numpy()
    forma = customers.forma_pagamento.to_numpy()
    canal = customers.canal_aquisicao.to_numpy()
    annual = customers.ciclo_contrato.eq("Anual").to_numpy()
    auto = customers.renovacao_automatica.to_numpy()
    adj = PRICE_ADJUSTMENTS.set_index("ano")["percentual"]

    # Estado dinâmico
    eng = lat.engagement_base.copy() - 0.9 * (~lat.activated)
    eng_hist = np.zeros((n, 4))  # engajamento dos últimos 4 meses (para "queda")
    sat = lat.satisfaction_base.copy()
    drift = np.zeros(n)
    drift_left = np.zeros(n, int)
    issue_left = np.zeros(n, int)
    late_hist = np.zeros((n, 3))
    complaint_hist = np.zeros((n, 3))
    open_invoice = np.zeros(n, bool)
    months_since_adj = np.full(n, 99)
    last_adj_pct = np.zeros(n)
    tenure = np.zeros(n, int)
    active = np.zeros(n, bool)
    churned = np.zeros(n, bool)
    cancel_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    cancel_reason = np.empty(n, dtype=object)

    usage_rows, pay_rows, ticket_rows, survey_rows = [], [], [], []
    ticket_seq = 0

    for month in _month_range():
        m64 = np.datetime64(month, "D")
        dim = calendar.monthrange(month.year, month.month)[1]
        # Ativa quem começou neste mês.
        new = (signup_ms == np.datetime64(month)) & ~churned
        active |= new
        idx = np.where(active)[0]
        if len(idx) == 0:
            continue
        tenure[idx] += 1
        k = len(idx)

        # ---- Reajuste anual (janeiro) --------------------------------------- #
        months_since_adj[idx] += 1
        if month.month == 1 and month.year in adj.index:
            eligible = idx[tenure[idx] > 6]
            price[eligible] *= 1 + adj[month.year]
            last_adj_pct[eligible] = adj[month.year]
            months_since_adj[eligible] = 0

        # ---- Choques de comportamento -------------------------------------- #
        # Episódio de desengajamento (projeto encerrado, troca de ferramenta, etc.)
        start_drift = (drift_left[idx] == 0) & (rng.random(k) < 0.028 + 0.05 * lat.competitor_affinity[idx])
        sd = idx[start_drift]
        drift[sd] = -rng.uniform(0.25, 0.6, len(sd))
        drift_left[sd] = rng.integers(2, 6, len(sd))
        # Episódio de problema técnico.
        start_issue = (issue_left[idx] == 0) & (rng.random(k) < 0.018)
        si = idx[start_issue]
        issue_left[si] = rng.integers(1, 4, len(si))

        d = np.where(drift_left[idx] > 0, drift[idx], 0.0)
        issue = issue_left[idx] > 0
        onboarding = np.clip((3 - tenure[idx]) * 0.15, 0, None) * (~lat.activated[idx])

        eng[idx] = (
            0.80 * eng[idx]
            + 0.20 * lat.engagement_base[idx]
            + d
            - onboarding
            - 0.25 * issue
            + rng.normal(0, 0.22, k)
        )
        drift_left[idx] = np.maximum(drift_left[idx] - 1, 0)
        issue_left[idx] = np.maximum(issue_left[idx] - 1, 0)

        # ---- Uso do produto ------------------------------------------------- #
        season = 0.88 if month.month in (1, 12) else (0.95 if month.month == 7 else 1.0)
        lam = ref_usage[idx] * np.exp(0.55 * eng[idx]) * season
        acessos = rng.poisson(lam)
        # Clientes muito desengajados podem passar o mês inteiro sem acessar.
        silent = rng.random(k) < _sigmoid(-5.0 - 2.0 * eng[idx])
        acessos = np.where(silent, 0, acessos)
        dias_ativos = np.minimum(dim, np.round(dim * (1 - np.exp(-acessos / (ref_usage[idx] * 0.9)))).astype(int))
        sessoes = np.round(acessos * rng.uniform(1.05, 1.6, k)).astype(int)
        min_sessao = rng.lognormal(np.log(14) + 0.15 * eng[idx], 0.35)
        minutos = np.round(sessoes * min_sessao).astype(int)
        p_feat = _sigmoid(0.2 + 0.9 * eng[idx])
        func = rng.binomial(n_features[idx], p_feat)
        func = np.where(acessos == 0, 0, np.maximum(func, 1))
        # Dias entre o último acesso e o fim do mês: cresce quando o engajamento cai.
        gap = np.minimum(dim - 1, rng.exponential(2.5 * np.exp(-1.1 * eng[idx]))).astype(int)
        last_day = np.where(acessos > 0, dim - gap, 0)

        # ---- Satisfação ------------------------------------------------------ #
        sat[idx] = 0.75 * sat[idx] + 0.25 * lat.satisfaction_base[idx] - 0.35 * issue + rng.normal(0, 0.15, k)

        # ---- Faturamento ----------------------------------------------------- #
        venc = m64 + np.timedelta64(9, "D")  # vencimento dia 10
        boleto = forma[idx] == "Boleto"
        auto_debit = np.isin(forma[idx], ["Débito automático", "Cartão de crédito"])
        p_late = _sigmoid(
            -3.4 + 4.2 * lat.fin_fragility[idx] + 0.7 * boleto - 0.6 * auto_debit
            + 0.4 * (eng[idx] < -1) + 0.5 * (months_since_adj[idx] <= 2)
        )
        late = rng.random(k) < p_late
        delay = np.where(late, rng.geometric(0.08, k), 0)
        delay = np.where(~late & (rng.random(k) < 0.25), -rng.integers(0, 6, k), delay)  # pagos antes
        unpaid = late & (delay > 45)
        late_hist[idx] = np.column_stack([late_hist[idx, 1:], late])
        open_invoice[idx] = unpaid

        # ---- Chamados de suporte ------------------------------------------- #
        month_adj = months_since_adj[idx] <= 1
        lam_t = (
            0.10
            + 0.12 * np.log1p(ref_usage[idx] / 30)
            + 0.9 * issue
            + 0.25 * np.maximum(0, -sat[idx])
            + 0.35 * late
            + 0.30 * month_adj * lat.price_sensitivity[idx]
            + 0.30 * (tenure[idx] <= 2)
        )
        n_tk = rng.poisson(lam_t)
        complaints_this_month = np.zeros(k)
        support_degradation = 1.0 + 0.35 * (month >= pd.Timestamp("2026-03-01"))  # drift proposital
        for j in np.where(n_tk > 0)[0]:
            ci = idx[j]
            for _ in range(n_tk[j]):
                ticket_seq += 1
                w = np.array([
                    0.25 + 1.6 * issue[j],                      # Técnico
                    0.10 + 0.8 * late[j] + 0.6 * month_adj[j],  # Cobrança
                    0.30 + 0.5 * (tenure[ci] <= 3),             # Dúvida
                    0.08 + 0.6 * max(0.0, -sat[ci]),            # Reclamação
                ])
                cat = rng.choice(["Técnico", "Cobrança", "Dúvida", "Reclamação"], p=w / w.sum())
                base_h = {"Técnico": 18, "Cobrança": 10, "Dúvida": 5, "Reclamação": 30}[cat]
                hours = float(np.round(rng.lognormal(np.log(base_h * support_degradation), 0.8), 1))
                opened = m64 + np.timedelta64(int(rng.integers(0, dim)), "D")
                rating = 4.6 - 0.35 * np.log1p(hours / 8) - 0.8 * (cat == "Reclamação") + 0.45 * sat[ci]
                rating = int(np.clip(np.round(rating + rng.normal(0, 0.6)), 1, 5))
                ticket_rows.append((
                    f"TCK-{ticket_seq:06d}", customers.customer_id.iat[ci], opened, cat, hours,
                    rating if rng.random() < 0.55 else None,
                ))
                if cat == "Reclamação" or (rating <= 2):
                    complaints_this_month[j] += 1
                sat[ci] -= 0.08 * (hours > 48) + 0.05 * (cat == "Reclamação")
        complaint_hist[idx] = np.column_stack([complaint_hist[idx, 1:], complaints_this_month])

        # ---- Pesquisa de satisfação trimestral ------------------------------ #
        if month.month in (3, 6, 9, 12):
            resp = idx[rng.random(k) < 0.33]
            if len(resp):
                score = np.clip(np.round(7.3 + 1.6 * sat[resp] + 0.4 * eng[resp] + rng.normal(0, 1.0, len(resp))), 0, 10)
                day = rng.integers(15, dim + 1, len(resp))
                for c, s, dd in zip(resp, score, day):
                    survey_rows.append((customers.customer_id.iat[c], m64 + np.timedelta64(int(dd) - 1, "D"), int(s)))

        # ---- Risco de cancelamento (hazard mensal) -------------------------- #
        eng_hist[idx] = np.column_stack([eng_hist[idx, 1:], eng[idx]])
        decline = np.maximum(0, eng_hist[idx, 0] - eng[idx]) * (tenure[idx] > 3)
        comp = {
            "low_usage": 0.95 * np.maximum(0, -eng[idx]),
            "competitor": 1.25 * decline + 1.5 * lat.competitor_affinity[idx] * (d < 0),
            "delinquency": 0.55 * late_hist[idx].sum(1) + 1.6 * open_invoice[idx],
            "support": 0.45 * complaint_hist[idx].sum(1) + 0.45 * np.maximum(0, -sat[idx]),
            "technical": 1.0 * issue,
            "price": 22.0 * lat.price_sensitivity[idx] * last_adj_pct[idx] * (months_since_adj[idx] <= 2),
        }
        logit = (
            -6.00
            + 1.90 * sum(comp.values())
            + 0.45 * (canal[idx] == "Anúncios pagos")
            - 0.45 * auto[idx]
            + 0.35 * boleto
            + 0.55 * (tenure[idx] <= 3)
            - 0.30 * (tenure[idx] > 24)
        )
        p = _sigmoid(logit)
        # Contratos anuais: saída concentrada no aniversário; fora dele, só com multa.
        anniversary = (tenure[idx] % 12 == 0)
        p = np.where(annual[idx], np.where(anniversary, np.minimum(p * 4.0, 0.9), p * 0.12), p)
        # Ninguém cancela no primeiro mês parcial.
        p = np.where(tenure[idx] <= 1, 0.0, p)

        cancels = rng.random(k) < p
        ci = idx[cancels]
        if len(ci):
            cdays = rng.integers(1, dim + 1, len(ci))
            cancel_date[ci] = m64 + (cdays - 1).astype("timedelta64[D]")
            contrib = np.column_stack([comp[key][cancels] for key in comp])
            dominant = np.array(list(comp))[contrib.argmax(1)]
            weak = contrib.max(1) < 0.3
            cancel_reason[ci] = np.where(weak, CANCEL_REASONS["other"], [CANCEL_REASONS[x] for x in dominant])
            churned[ci] = True
            active[ci] = False
        else:
            cdays = np.array([], int)

        # ---- Registros do mês ------------------------------------------------ #
        frac = np.ones(k)
        cancel_day_by_pos = dict(zip(np.where(cancels)[0], cdays))
        for pos, cd in cancel_day_by_pos.items():
            frac[pos] = cd / dim  # uso parcial no mês do cancelamento
        start_frac = np.where(signup_ms[idx] == np.datetime64(month),
                              1 - (signup[idx] - m64).astype(int) / dim, 1.0)
        frac = frac * start_frac

        for j in range(k):
            ci_ = idx[j]
            f = frac[j]
            a = int(round(acessos[j] * f))
            s_ = int(round(sessoes[j] * f))
            ld = int(min(last_day[j], max(1, int(dim * f)))) if a > 0 else None
            usage_rows.append((
                customers.customer_id.iat[ci_], month, a, s_, int(round(minutos[j] * f)),
                int(func[j]) if a > 0 else 0, int(round(dias_ativos[j] * f)),
                (m64 + np.timedelta64(ld - 1, "D")) if ld else None,
            ))
            # Fatura: emitida se o cliente estava ativo no vencimento.
            if cancels[j] and cancel_day_by_pos[j] < 10:
                continue
            if signup_ms[ci_] == np.datetime64(month) and signup[ci_] > venc:
                continue
            valor = round(float(price[ci_]), 2)
            if unpaid[j]:
                pay_date, status = None, "Em aberto"
            else:
                pay_date = venc + np.timedelta64(int(delay[j]), "D")
                status = "Pago com atraso" if delay[j] > 0 else "Pago em dia"
            pay_rows.append((customers.customer_id.iat[ci_], month, valor, venc, pay_date, status))

    customers = customers.copy()
    customers["data_cancelamento"] = pd.to_datetime(cancel_date)
    customers["motivo_cancelamento"] = cancel_reason
    customers["status"] = np.where(churned, "Cancelado", "Ativo")

    usage = pd.DataFrame(usage_rows, columns=[
        "customer_id", "mes_referencia", "acessos", "sessoes", "minutos_uso",
        "funcionalidades_usadas", "dias_ativos", "ultimo_acesso"])
    payments = pd.DataFrame(pay_rows, columns=[
        "customer_id", "mes_referencia", "valor_cobrado", "data_vencimento", "data_pagamento", "status"])
    tickets = pd.DataFrame(ticket_rows, columns=[
        "ticket_id", "customer_id", "data_abertura", "categoria", "tempo_resolucao_horas", "avaliacao"])
    surveys = pd.DataFrame(survey_rows, columns=["customer_id", "data_resposta", "nota_satisfacao"])
    return customers, usage, payments, tickets, surveys


# --------------------------------------------------------------------------- #
# 3. Pós-processamento: tipos, ids e problemas de qualidade propositais
# --------------------------------------------------------------------------- #
def _finalize(customers, usage, payments, tickets, surveys, rng):
    for df, cols in [
        (usage, ["mes_referencia", "ultimo_acesso"]),
        (payments, ["mes_referencia", "data_vencimento", "data_pagamento"]),
        (tickets, ["data_abertura"]),
        (surveys, ["data_resposta"]),
    ]:
        for c in cols:
            df[c] = pd.to_datetime(df[c])

    payments.insert(0, "payment_id", [f"PAG-{i:07d}" for i in range(1, len(payments) + 1)])
    # Resolução só é conhecida depois que o chamado é fechado.
    tickets["data_resolucao"] = tickets.data_abertura + pd.to_timedelta(tickets.tempo_resolucao_horas, unit="h")
    tickets["avaliacao"] = tickets["avaliacao"].astype("Int64")

    n = len(customers)
    # (a) idade ausente (~3%) e erros de digitação (idade 0 ou > 110)
    miss = rng.random(n) < 0.03
    customers.loc[miss, "idade"] = np.nan
    typo = rng.choice(np.where(~miss)[0], size=12, replace=False)
    customers.loc[typo, "idade"] = rng.choice([0, 1, 112, 120, 150], size=12)
    # (b) região ausente (~0,6%)
    customers.loc[rng.random(n) < 0.006, "regiao"] = np.nan
    # (c) outliers de uso: integrações/automação gerando minutos absurdos (~0,2%)
    out = rng.random(len(usage)) < 0.002
    usage.loc[out, "minutos_uso"] = (usage.loc[out, "minutos_uso"] * rng.uniform(15, 40, out.sum())).round()
    # (d) duplicatas exatas no cadastro (falha de integração com o CRM)
    dup = customers.sample(25, random_state=config.RANDOM_SEED)
    customers = pd.concat([customers, dup]).sort_values("customer_id").reset_index(drop=True)
    return customers, usage, payments, tickets, surveys


def generate(seed: int = config.RANDOM_SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    customers, latent = _generate_customers(rng)
    customers, usage, payments, tickets, surveys = _simulate(customers, latent, rng)
    customers, usage, payments, tickets, surveys = _finalize(customers, usage, payments, tickets, surveys, rng)
    plans = PLANS.drop(columns="acessos_referencia")
    return {
        "plans": plans,
        "price_adjustments": PRICE_ADJUSTMENTS,
        "customers": customers,
        "usage_monthly": usage,
        "payments": payments,
        "tickets": tickets,
        "surveys": surveys,
    }


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    tables = generate()
    for name, df in tables.items():
        path = config.RAW_DIR / f"{name}.csv"
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
        print(f"  {name:<18} {len(df):>9,} linhas  -> {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
