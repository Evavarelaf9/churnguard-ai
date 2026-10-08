"""Explicabilidade individual e global do ChurnGuard AI.

Objetivo: transformar contribuições SHAP (números técnicos) em frases que uma
pessoa de negócio entende e em uma **sugestão** de ação — nunca uma decisão
automática.

Como funciona
-------------
1. O pipeline é dividido em duas partes: engenharia de atributos + pré-processamento
   (tudo que vem antes do modelo) e o modelo em si.
2. O SHAP calcula quanto cada atributo empurrou a previsão para cima ou para baixo
   em relação à média (em log-odds para boosting/logística).
3. As colunas one-hot de uma mesma variável categórica são somadas, para que o
   usuário veja "Forma de pagamento", e não "forma_pagamento_Boleto".
4. Cada atributo tem um *template* de frase que usa o valor real do cliente.
5. A categoria do fator mais relevante (uso, financeiro, suporte, preço, contrato)
   define a sugestão de ação.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.src.features.build_features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    add_features,
)

# --------------------------------------------------------------------------- #
# Formatação
# --------------------------------------------------------------------------- #

def _pct(v: float) -> str:
    return f"{abs(v) * 100:.0f}%"


def _n(v: float) -> int:
    return int(round(float(v)))


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# (categoria, função que gera a frase de RISCO a partir da linha do cliente)
RISK_TEMPLATES = {
    "variacao_uso": ("uso", lambda r: f"queda de {_pct(r.variacao_uso)} na utilização (últimos 3 meses vs. 3 anteriores)"
                     if r.variacao_uso < -0.02 else "utilização estagnada nos últimos meses"),
    "variacao_uso_ultimo_mes": ("uso", lambda r: f"queda de {_pct(r.variacao_uso_ultimo_mes)} nos acessos no último mês"
                                if r.variacao_uso_ultimo_mes < -0.02 else "oscilação recente nos acessos"),
    "tendencia_uso": ("uso", lambda r: "tendência de queda no uso ao longo dos últimos 6 meses"
                      if r.tendencia_uso < 0 else "uso sem crescimento nos últimos 6 meses"),
    "meses_consecutivos_queda": ("uso", lambda r: f"uso em queda há {_plural(_n(r.meses_consecutivos_queda), 'mês', 'meses')} seguidos"),
    "media_acessos_semana": ("uso", lambda r: f"baixa frequência de uso: {r.media_acessos_semana:.1f} acessos por semana".replace(".", ",")),
    "dias_desde_ultimo_acesso": ("uso", lambda r: f"{_plural(_n(r.dias_desde_ultimo_acesso), 'dia', 'dias')} desde o último acesso"),
    "taxa_utilizacao_funcionalidades": ("uso", lambda r: f"usa apenas {_pct(r.taxa_utilizacao_funcionalidades)} das funcionalidades do plano"),
    "taxa_dias_ativos": ("uso", lambda r: "nenhum dia com acesso no último mês" if _n(r.taxa_dias_ativos * 30) == 0
                         else f"ativo em apenas {_plural(_n(r.taxa_dias_ativos * 30), 'dia', 'dias')} no último mês"),
    "minutos_por_sessao": ("uso", lambda r: f"sessões curtas ({_n(r.minutos_por_sessao)} min em média)"),
    "variacao_minutos": ("uso", lambda r: f"queda de {_pct(r.variacao_minutos)} no tempo de uso"
                         if r.variacao_minutos < -0.02 else "tempo de uso estagnado"),
    "historico_curto": ("uso", lambda r: "pouco histórico de uso para comparação"),
    "percentual_pagamentos_atrasados": ("financeiro", lambda r: f"{_plural(_n(r.faturas_atrasadas_6m), 'pagamento atrasado', 'pagamentos atrasados')} nos últimos 6 meses"),
    "dias_atraso_medio_6m": ("financeiro", lambda r: f"atraso médio de {_plural(_n(r.dias_atraso_medio_6m), 'dia', 'dias')} nos pagamentos"),
    "faturas_em_aberto": ("financeiro", lambda r: f"{_plural(_n(r.faturas_em_aberto), 'fatura em aberto', 'faturas em aberto')}"),
    "tem_fatura_em_aberto": ("financeiro", lambda r: "possui fatura vencida em aberto"),
    "relacao_tickets_por_mes": ("suporte", lambda r: f"{r.relacao_tickets_por_mes:.1f} chamados de suporte por mês".replace(".", ",")),
    "aceleracao_tickets": ("suporte", lambda r: f"aumento de chamados de suporte (+{_n(r.aceleracao_tickets)} no último trimestre)"
                           if r.aceleracao_tickets > 0 else "padrão de chamados de suporte"),
    "reclamacoes_6m": ("suporte", lambda r: f"{_plural(_n(r.reclamacoes_6m), 'reclamação', 'reclamações')} nos últimos 6 meses"),
    "proporcao_reclamacoes": ("suporte", lambda r: f"{_pct(r.proporcao_reclamacoes)} dos chamados são reclamações"),
    "tickets_tecnicos_6m": ("suporte", lambda r: f"{_plural(_n(r.tickets_tecnicos_6m), 'chamado técnico', 'chamados técnicos')} nos últimos 6 meses"),
    "tempo_medio_resolucao_h_6m": ("suporte", lambda r: f"chamados levam {_n(r.tempo_medio_resolucao_h_6m)}h em média para serem resolvidos"),
    "avaliacao_media_6m": ("suporte", lambda r: f"avaliação do suporte de {r.avaliacao_media_6m:.1f}/5".replace(".", ",")),
    "tempo_desde_ultimo_chamado": ("suporte", lambda r: f"chamado recente (há {_plural(_n(r.tempo_desde_ultimo_chamado), 'dia', 'dias')})"
                                   if r.tempo_desde_ultimo_chamado < 60 else "pouco contato com o suporte"),
    "nunca_abriu_chamado": ("suporte", lambda r: "nunca abriu chamado (pouco relacionamento com a empresa)"),
    "sem_avaliacao_suporte": ("suporte", lambda r: "não avaliou o atendimento recente"),
    "ultima_nota_satisfacao": ("suporte", lambda r: f"nota de satisfação {_n(r.ultima_nota_satisfacao)}/10"),
    "sem_pesquisa_satisfacao": ("suporte", lambda r: "não respondeu à pesquisa de satisfação"),
    "reajuste_pct_6m": ("preco", lambda r: f"reajuste recente de {_pct(r.reajuste_pct_6m)} na mensalidade"
                        if r.reajuste_pct_6m > 0 else "sem reajuste recente"),
    "teve_reajuste_recente": ("preco", lambda r: "teve reajuste de preço recentemente"),
    "valor_mensal": ("preco", lambda r: f"mensalidade de {_brl(r.valor_mensal)} em relação ao uso"),
    "valor_total_pago": ("contrato", lambda r: f"relacionamento financeiro ainda curto ({_brl(r.valor_total_pago)} pagos)"),
    "tempo_cliente_meses": ("contrato", lambda r: f"cliente há {_plural(_n(r.tempo_cliente_meses), 'mês', 'meses')}"),
    "cliente_novo": ("contrato", lambda r: "cliente novo, ainda em período de adaptação"),
    "meses_ate_renovacao": ("contrato", lambda r: "contrato mensal, sem fidelidade — pode cancelar a qualquer momento"
                            if r.ciclo_contrato == "Mensal"
                            else f"renovação do contrato anual em {_plural(_n(r.meses_ate_renovacao), 'mês', 'meses')}"),
    "renovacao_proxima": ("contrato", lambda r: "renovação do contrato anual se aproximando"),
    "renovacao_automatica": ("contrato", lambda r: "renovação automática desativada" if r.renovacao_automatica == 0 else "renovação automática ativada"),
    "tipo_cliente": ("perfil", lambda r: f"perfil {r.tipo_cliente}"),
    "plano": ("preco", lambda r: f"plano {r.plano}"),
    "ciclo_contrato": ("contrato", lambda r: f"contrato {str(r.ciclo_contrato).lower()} (sem fidelidade)" if r.ciclo_contrato == "Mensal" else f"contrato {str(r.ciclo_contrato).lower()}"),
    "forma_pagamento": ("financeiro", lambda r: f"pagamento via {str(r.forma_pagamento).lower()}"),
    "canal_aquisicao": ("perfil", lambda r: f"cliente adquirido via {str(r.canal_aquisicao).lower()}"),
}

# Frases para quando o dado não existe (ex.: cliente sem chamados resolvidos).
MISSING_TEXTS = {
    "tempo_medio_resolucao_h_6m": "sem chamados resolvidos nos últimos 6 meses",
    "avaliacao_media_6m": "sem avaliação recente do suporte",
    "ultima_nota_satisfacao": "sem nota de satisfação nos últimos 12 meses",
    "dias_desde_ultimo_acesso": "sem registro de acesso",
    "dias_atraso_medio_6m": "sem faturas recentes",
}

CATEGORY_LABELS = {
    "uso": "Utilização",
    "financeiro": "Financeiro",
    "suporte": "Suporte e satisfação",
    "preco": "Preço e plano",
    "contrato": "Contrato e ciclo de vida",
    "perfil": "Perfil",
}

RECOMMENDATIONS = {
    "uso": "Priorizar contato do Customer Success para investigar a queda de utilização e oferecer "
           "treinamento ou reonboarding das funcionalidades do plano.",
    "financeiro": "Oferecer negociação das pendências e migração para pagamento automático (cartão ou débito), "
                  "revisando a régua de cobrança.",
    "suporte": "Revisar o histórico de chamados com o suporte e fazer contato de follow-up sobre os problemas "
               "relatados antes de qualquer oferta comercial.",
    "preco": "Avaliar se o plano está adequado ao uso real (ajuste de plano ou condição especial) e reforçar o "
             "valor entregue após o reajuste.",
    "contrato": "Iniciar conversa de renovação com antecedência, mostrando resultados obtidos com o produto.",
    "contrato_mensal": "Apresentar os resultados obtidos com o produto e avaliar oferta de migração para o "
                       "contrato anual com benefício.",
    "perfil": "Incluir em campanha de engajamento e acompanhar a evolução do uso.",
}

PRIORITY = {
    "Alto": "Prioridade alta — sugerido contato humano em até 48h.",
    "Médio": "Prioridade média — sugerida ação de engajamento (e-mail, conteúdo, webinar) e acompanhamento.",
    "Baixo": "Sem ação imediata — manter monitoramento.",
}


def neutral_value(feature: str, r) -> str:
    """Valor formatado de forma neutra (usado para fatores de proteção)."""
    v = getattr(r, feature)
    if feature in CATEGORICAL_FEATURES:
        return str(v)
    if feature == "tempo_desde_ultimo_chamado" and v >= 365:
        return "nenhum chamado no último ano"
    if feature == "tempo_desde_ultimo_chamado":
        return f"{int(v)} dias"
    if feature in BINARY_FEATURES:
        return "Sim" if int(v) == 1 else "Não"
    if feature in ("variacao_uso", "variacao_uso_ultimo_mes", "variacao_minutos"):
        return f"{v * 100:+.0f}%"
    if feature in ("taxa_utilizacao_funcionalidades", "percentual_pagamentos_atrasados", "proporcao_reclamacoes",
                   "reajuste_pct_6m", "taxa_dias_ativos"):
        return f"{v * 100:.0f}%"
    if feature in ("valor_mensal", "valor_total_pago"):
        return _brl(v)
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.1f}".replace(".", ",")


STRUCTURAL_FEATURES = {"tipo_cliente", "canal_aquisicao", "plano", "ciclo_contrato", "tempo_cliente_meses",
                       "valor_total_pago", "historico_curto", "cliente_novo"}


def _is_structural(feature: str, row) -> bool:
    """Características que a empresa não muda com uma ação de retenção no curto prazo."""
    if feature == "meses_ate_renovacao":
        return getattr(row, "ciclo_contrato", None) == "Mensal"
    return feature in STRUCTURAL_FEATURES


# --------------------------------------------------------------------------- #
# Explainer
# --------------------------------------------------------------------------- #
@dataclass
class Factor:
    feature: str
    label: str
    category: str
    impact: float
    value: str
    text: str
    structural: bool = False

    def as_dict(self) -> dict:
        return {
            "feature": self.feature, "label": self.label, "category": self.category,
            "category_label": CATEGORY_LABELS[self.category], "impact": round(float(self.impact), 4),
            "value": self.value, "text": self.text, "acionavel": not self.structural,
        }


class ChurnExplainer:
    """Explica previsões de um pipeline treinado (features → preprocess → model)."""

    def __init__(self, pipeline, background: pd.DataFrame | None = None, risk_thresholds: dict | None = None):
        import shap

        self.pipeline = pipeline
        self.transformer = pipeline[:-1]
        self.model = pipeline[-1]
        self.risk_thresholds = risk_thresholds
        self.feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
        self._groups = self._build_groups()
        name = type(self.model).__name__
        if name == "LogisticRegression":
            if background is None:
                raise ValueError("LogisticRegression precisa de amostra de background para o SHAP")
            self.explainer = shap.LinearExplainer(self.model, self.transformer.transform(background))
        else:
            self.explainer = shap.TreeExplainer(self.model)

    def _build_groups(self) -> dict[str, list[int]]:
        """Mapeia cada variável original para os índices das colunas transformadas."""
        groups: dict[str, list[int]] = {}
        for i, col in enumerate(self.feature_names):
            base = next((c for c in CATEGORICAL_FEATURES if col.startswith(c + "_")), col)
            groups.setdefault(base, []).append(i)
        return groups

    def shap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """SHAP agregado por variável original (linhas = clientes)."""
        Xt = self.transformer.transform(df)
        sv = self.explainer.shap_values(Xt)
        sv = np.asarray(sv)
        if sv.ndim == 3:  # RandomForest: (n, features, classes)
            sv = sv[:, :, 1]
        agg = {feat: sv[:, idx].sum(axis=1) for feat, idx in self._groups.items()}
        return pd.DataFrame(agg, index=df.index)

    def explain(self, df: pd.DataFrame, top_k: int = 4) -> list[dict]:
        from ml.src.evaluation.metrics import classify_risk

        proba = self.pipeline.predict_proba(df)[:, 1]
        risk = classify_risk(proba, self.risk_thresholds)
        contrib = self.shap_values(df)
        feats = add_features(df)
        out = []
        for i, (idx, row) in enumerate(feats.iterrows()):
            c = contrib.loc[idx].sort_values(ascending=False)
            risk_factors = [self._factor(f, row, v) for f, v in c.items() if v > 0.01][:top_k]
            protective = [self._factor(f, row, v, neutral=True) for f, v in c.sort_values().items() if v < -0.01][:3]
            # Fatores estruturais (perfil, canal, contrato mensal...) continuam na lista, mas o
            # "principal fator" e a recomendação priorizam o que a empresa pode influenciar.
            actionable = [f for f in risk_factors if not f.structural]
            lead = actionable or risk_factors
            main_cat = self._main_category(lead)
            out.append({
                "probabilidade": float(proba[i]),
                "nivel_risco": str(risk[i]),
                "fatores_risco": [f.as_dict() for f in risk_factors],
                "fatores_protecao": [f.as_dict() for f in protective],
                "principal_fator": lead[0].text if lead else "nenhum fator relevante de risco",
                "resumo": self.narrative(lead, risk_factors, protective),
                "categoria_principal": main_cat,
                "recomendacao": self.recommend(str(risk[i]), main_cat, row.get("ciclo_contrato")),
            })
        return out

    @staticmethod
    def narrative(lead: list[Factor], risk_factors: list[Factor], protective: list[Factor]) -> str:
        """Frase-resumo em linguagem de negócio (ex.: para a equipe de relacionamento)."""
        if not risk_factors:
            return "Nenhum sinal relevante de risco foi identificado para este cliente."
        main = lead[0]
        others = [f.text for f in risk_factors if f is not main][:2]
        text = f"O fator que mais contribuiu para aumentar o risco deste cliente foi: {main.text}"
        text += f"; também pesaram: {' e '.join(others)}." if others else "."
        if protective:
            p = protective[0]
            text += f" Em sentido contrário, {p.label.lower()} ({p.value}) ajuda a reduzir o risco."
        return text

    @staticmethod
    def _factor(feature: str, row, impact: float, neutral: bool = False) -> Factor:
        category, template = RISK_TEMPLATES.get(feature, ("perfil", lambda r: FEATURE_LABELS.get(feature, feature)))
        label = FEATURE_LABELS.get(feature, feature)
        raw = getattr(row, feature, None)
        if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
            value = "sem informação"
            text = f"{label}: sem informação" if neutral else MISSING_TEXTS.get(feature, f"{label}: sem informação")
        else:
            value = neutral_value(feature, row)
            text = f"{label}: {value}" if neutral else template(row)
        return Factor(feature, label, category, impact, value, text, _is_structural(feature, row))

    @staticmethod
    def _main_category(factors: list[Factor]) -> str | None:
        """Categoria com maior contribuição somada entre os principais fatores."""
        if not factors:
            return None
        totals: dict[str, float] = {}
        for f in factors:
            totals[f.category] = totals.get(f.category, 0) + f.impact
        return max(totals, key=totals.get)

    @staticmethod
    def recommend(risk_level: str, category: str | None, contract: str | None = None) -> dict:
        key = category or "perfil"
        if key == "contrato" and contract == "Mensal":
            key = "contrato_mensal"
        action = RECOMMENDATIONS[key]
        if risk_level == "Baixo":
            action = "Nenhuma ação de retenção necessária no momento."
        return {
            "prioridade": PRIORITY[risk_level],
            "acao_sugerida": action,
            "aviso": "Sugestão baseada em padrões dos dados. A decisão final é da equipe responsável.",
        }
