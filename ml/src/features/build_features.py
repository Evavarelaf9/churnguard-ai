"""Engenharia de atributos do ChurnGuard AI.

Transforma a base analítica (agregados brutos por cliente × data) em atributos
que descrevem **comportamento e mudança de comportamento**.

Princípio anti-vazamento
------------------------
Todas as transformações aqui são **por linha e sem estado**: cada atributo usa
apenas colunas da própria linha (que já são *point-in-time*). Nada é "aprendido"
com a base inteira — médias, limites de outlier e imputações que dependem dos
dados ficam no pipeline do scikit-learn, ajustados só no treino.

Por isso `add_features` pode ser aplicada antes ou depois da divisão treino/teste
sem risco, e é a mesma função usada pela API em produção.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_MONTH = 4.33
NO_TICKET_DAYS = 365  # valor usado quando o cliente nunca abriu chamado


def _safe_ratio(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona os atributos derivados. Não altera o DataFrame de entrada."""
    d = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Utilização atual
    # ------------------------------------------------------------------ #
    d["media_acessos_semana"] = d["acessos_m1"] / WEEKS_PER_MONTH
    d["taxa_utilizacao_funcionalidades"] = _safe_ratio(d["funcionalidades_usadas_m1"], d["funcionalidades_disponiveis"])
    d["taxa_dias_ativos"] = (d["dias_ativos_m1"] / 30).clip(0, 1)
    d["minutos_por_sessao"] = _safe_ratio(d["minutos_uso_m1"], d["sessoes_m1"])

    # ------------------------------------------------------------------ #
    # 2. Mudança de comportamento (o sinal mais forte da EDA)
    # ------------------------------------------------------------------ #
    recent = d[["acessos_m1", "acessos_m2", "acessos_m3"]].mean(axis=1)
    previous = d[["acessos_m4", "acessos_m5", "acessos_m6"]].mean(axis=1)
    d["historico_curto"] = previous.isna().astype(int)  # < 6 meses de histórico
    # Variação dos últimos 3 meses vs. os 3 anteriores. Sem histórico → 0 (neutro) + flag acima.
    d["variacao_uso"] = _safe_ratio(recent - previous, previous, fill=0.0).clip(-1, 3)
    # Variação do último mês vs. o anterior (queda abrupta).
    d["variacao_uso_ultimo_mes"] = _safe_ratio(d["acessos_m1"] - d["acessos_m2"], d["acessos_m2"], fill=0.0).clip(-1, 3)
    d["variacao_minutos"] = _safe_ratio(d["minutos_uso_m1"] - d["minutos_uso_m2"], d["minutos_uso_m2"], fill=0.0).clip(-1, 3)

    # Tendência: inclinação da reta de acessos nos últimos meses, normalizada pela média.
    lags = d[[f"acessos_m{i}" for i in range(6, 0, -1)]].to_numpy(dtype=float)  # do mais antigo ao mais recente
    x = np.arange(lags.shape[1], dtype=float)
    mask = ~np.isnan(lags)
    n = mask.sum(1)
    xm = np.where(mask, x, 0).sum(1) / np.maximum(n, 1)
    ym = np.where(mask, lags, 0).sum(1) / np.maximum(n, 1)
    cov = np.where(mask, (x - xm[:, None]) * (np.nan_to_num(lags) - ym[:, None]), 0).sum(1)
    var = np.where(mask, (x - xm[:, None]) ** 2, 0).sum(1)
    slope = np.divide(cov, var, out=np.zeros_like(cov), where=var > 0)
    d["tendencia_uso"] = np.clip(np.divide(slope, ym, out=np.zeros_like(slope), where=ym > 0), -1, 1)
    d["meses_consecutivos_queda"] = _consecutive_declines(lags)

    # ------------------------------------------------------------------ #
    # 3. Financeiro
    # ------------------------------------------------------------------ #
    d["percentual_pagamentos_atrasados"] = _safe_ratio(d["faturas_atrasadas_6m"], d["faturas_6m"])
    d["tem_fatura_em_aberto"] = (d["faturas_em_aberto"] > 0).astype(int)
    d["reajuste_pct_6m"] = d["reajuste_pct_6m"].fillna(0.0)
    d["teve_reajuste_recente"] = (d["reajuste_pct_6m"] > 0).astype(int)
    d["valor_total_pago"] = d["valor_total_pago"].fillna(0.0)

    # ------------------------------------------------------------------ #
    # 4. Relacionamento / suporte
    # ------------------------------------------------------------------ #
    months_obs = d["tempo_cliente_meses"].clip(lower=1, upper=6)
    d["relacao_tickets_por_mes"] = d["tickets_6m"] / months_obs
    # Aceleração: chamados dos últimos 3 meses menos os dos 3 anteriores.
    d["aceleracao_tickets"] = d["tickets_3m"] - (d["tickets_6m"] - d["tickets_3m"])
    d["proporcao_reclamacoes"] = _safe_ratio(d["reclamacoes_6m"], d["tickets_6m"])
    d["nunca_abriu_chamado"] = d["dias_desde_ultimo_chamado"].isna().astype(int)
    d["tempo_desde_ultimo_chamado"] = d["dias_desde_ultimo_chamado"].fillna(NO_TICKET_DAYS).clip(upper=NO_TICKET_DAYS)
    d["sem_avaliacao_suporte"] = d["avaliacao_media_6m"].isna().astype(int)
    d["sem_pesquisa_satisfacao"] = d["ultima_nota_satisfacao"].isna().astype(int)

    # ------------------------------------------------------------------ #
    # 5. Ciclo de vida / contrato
    # ------------------------------------------------------------------ #
    d["cliente_novo"] = (d["tempo_cliente_meses"] <= 3).astype(int)
    month_in_contract = np.floor(d["tempo_cliente_meses"]).astype(int) % 12
    annual = d["ciclo_contrato"].eq("Anual")
    # Contrato anual: meses até o aniversário; mensal: renova todo mês (1).
    d["meses_ate_renovacao"] = np.where(annual, 12 - month_in_contract, 1)
    d["renovacao_proxima"] = (annual & (d["meses_ate_renovacao"] <= 2)).astype(int)
    return d


def _consecutive_declines(lags: np.ndarray) -> np.ndarray:
    """Quantos meses seguidos o uso caiu, contando a partir do mês mais recente."""
    out = np.zeros(len(lags), dtype=int)
    active = np.ones(len(lags), dtype=bool)
    for k in range(lags.shape[1] - 1, 0, -1):
        cur, prev = lags[:, k], lags[:, k - 1]
        dec = active & ~np.isnan(cur) & ~np.isnan(prev) & (cur < prev)
        out += dec
        active &= dec
    return out


# ---------------------------------------------------------------------- #
# Conjunto de atributos usado pelo modelo
# ---------------------------------------------------------------------- #
CATEGORICAL_FEATURES = [
    "tipo_cliente",
    "plano",
    "ciclo_contrato",
    "forma_pagamento",
    "canal_aquisicao",
]

NUMERIC_FEATURES = [
    # contrato / valor
    "tempo_cliente_meses", "valor_mensal", "valor_total_pago", "reajuste_pct_6m", "meses_ate_renovacao",
    # utilização atual
    "media_acessos_semana", "taxa_utilizacao_funcionalidades", "taxa_dias_ativos", "minutos_por_sessao",
    "dias_desde_ultimo_acesso",
    # mudança de comportamento
    "variacao_uso", "variacao_uso_ultimo_mes", "variacao_minutos", "tendencia_uso", "meses_consecutivos_queda",
    # financeiro
    "percentual_pagamentos_atrasados", "dias_atraso_medio_6m", "faturas_em_aberto",
    # suporte e satisfação
    "relacao_tickets_por_mes", "aceleracao_tickets", "reclamacoes_6m", "proporcao_reclamacoes",
    "tickets_tecnicos_6m", "tempo_medio_resolucao_h_6m", "avaliacao_media_6m", "tempo_desde_ultimo_chamado",
    "ultima_nota_satisfacao",
]

BINARY_FEATURES = [
    "renovacao_automatica", "historico_curto", "tem_fatura_em_aberto", "teve_reajuste_recente",
    "nunca_abriu_chamado", "sem_avaliacao_suporte", "sem_pesquisa_satisfacao", "cliente_novo", "renovacao_proxima",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES

# Colunas da base analítica que NÃO podem entrar no modelo, e por quê.
EXCLUDED_COLUMNS = {
    "customer_id": "identificador",
    "nome": "dado pessoal, sem valor preditivo",
    "data_referencia": "metadado temporal (usado só para dividir treino/teste)",
    "idade": "atributo pessoal sem sinal na EDA — mantido só para auditoria de viés (LGPD)",
    "regiao": "atributo geográfico sem sinal na EDA — mantido só para auditoria de viés",
    "churn": "variável-alvo",
}

# Colunas brutas necessárias para calcular os atributos (contrato de entrada da API).
RAW_INPUT_COLUMNS = [
    "tipo_cliente", "plano", "funcionalidades_disponiveis", "ciclo_contrato", "forma_pagamento",
    "renovacao_automatica", "canal_aquisicao", "tempo_cliente_meses",
    "acessos_m1", "acessos_m2", "acessos_m3", "acessos_m4", "acessos_m5", "acessos_m6",
    "sessoes_m1", "minutos_uso_m1", "minutos_uso_m2", "funcionalidades_usadas_m1", "dias_ativos_m1",
    "dias_desde_ultimo_acesso", "faturas_6m", "faturas_atrasadas_6m", "dias_atraso_medio_6m", "faturas_em_aberto",
    "valor_total_pago", "valor_mensal", "reajuste_pct_6m", "tickets_3m", "tickets_6m", "tickets_tecnicos_6m",
    "reclamacoes_6m", "tempo_medio_resolucao_h_6m", "avaliacao_media_6m", "dias_desde_ultimo_chamado",
    "ultima_nota_satisfacao",
]

# Nomes de negócio para relatórios e dashboard.
FEATURE_LABELS = {
    "tempo_cliente_meses": "Tempo como cliente",
    "valor_mensal": "Valor mensal",
    "valor_total_pago": "Valor total já pago",
    "reajuste_pct_6m": "Reajuste recente",
    "meses_ate_renovacao": "Janela de renovação do contrato",
    "media_acessos_semana": "Acessos por semana",
    "taxa_utilizacao_funcionalidades": "Uso das funcionalidades do plano",
    "taxa_dias_ativos": "Dias ativos no mês",
    "minutos_por_sessao": "Minutos por sessão",
    "dias_desde_ultimo_acesso": "Dias desde o último acesso",
    "variacao_uso": "Variação de uso (3 meses)",
    "variacao_uso_ultimo_mes": "Variação de uso no último mês",
    "variacao_minutos": "Variação do tempo de uso",
    "tendencia_uso": "Tendência de uso (6 meses)",
    "meses_consecutivos_queda": "Meses seguidos de queda no uso",
    "percentual_pagamentos_atrasados": "Pagamentos em atraso",
    "dias_atraso_medio_6m": "Atraso médio de pagamento",
    "faturas_em_aberto": "Faturas em aberto",
    "relacao_tickets_por_mes": "Chamados por mês",
    "aceleracao_tickets": "Aumento de chamados",
    "reclamacoes_6m": "Reclamações",
    "proporcao_reclamacoes": "Proporção de reclamações",
    "tickets_tecnicos_6m": "Chamados técnicos",
    "tempo_medio_resolucao_h_6m": "Tempo de resolução dos chamados",
    "avaliacao_media_6m": "Avaliação do suporte",
    "tempo_desde_ultimo_chamado": "Tempo desde o último chamado",
    "ultima_nota_satisfacao": "Nota de satisfação",
    "renovacao_automatica": "Renovação automática",
    "historico_curto": "Histórico curto",
    "tem_fatura_em_aberto": "Possui fatura em aberto",
    "teve_reajuste_recente": "Teve reajuste recente",
    "nunca_abriu_chamado": "Nunca abriu chamado",
    "sem_avaliacao_suporte": "Sem avaliação do suporte",
    "sem_pesquisa_satisfacao": "Não respondeu pesquisa",
    "cliente_novo": "Cliente novo (≤ 3 meses)",
    "renovacao_proxima": "Renovação anual próxima",
    "tipo_cliente": "Tipo de cliente",
    "plano": "Plano",
    "ciclo_contrato": "Tipo de contrato",
    "forma_pagamento": "Forma de pagamento",
    "canal_aquisicao": "Canal de aquisição",
}
