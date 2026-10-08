"""Contratos da API (Pydantic v2). Geram automaticamente a documentação OpenAPI/Swagger."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["Baixo", "Médio", "Alto"]
Plano = Literal["Básico", "Essencial", "Profissional", "Empresarial"]
FormaPagamento = Literal["Cartão de crédito", "Boleto", "Pix", "Débito automático"]


# --------------------------------------------------------------------------- #
# Entrada do modelo
# --------------------------------------------------------------------------- #
class CustomerFeatures(BaseModel):
    """Dados de um cliente na data de referência (agregados dos últimos meses)."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "customer_id": "CLI-00042", "tipo_cliente": "Pequena Empresa", "plano": "Profissional",
        "funcionalidades_disponiveis": 14, "ciclo_contrato": "Mensal", "forma_pagamento": "Boleto",
        "renovacao_automatica": 0, "canal_aquisicao": "Anúncios pagos", "tempo_cliente_meses": 8.5,
        "acessos_m1": 22, "acessos_m2": 41, "acessos_m3": 58, "acessos_m4": 63, "acessos_m5": 60, "acessos_m6": 66,
        "sessoes_m1": 28, "minutos_uso_m1": 390, "minutos_uso_m2": 720, "funcionalidades_usadas_m1": 3,
        "dias_ativos_m1": 9, "dias_desde_ultimo_acesso": 12, "faturas_6m": 6, "faturas_atrasadas_6m": 2,
        "dias_atraso_medio_6m": 4.5, "faturas_em_aberto": 0, "valor_total_pago": 1599.2, "valor_mensal": 199.9,
        "reajuste_pct_6m": 0.0, "tickets_3m": 3, "tickets_6m": 4, "tickets_tecnicos_6m": 2, "reclamacoes_6m": 2,
        "tempo_medio_resolucao_h_6m": 26.0, "avaliacao_media_6m": 3.0, "dias_desde_ultimo_chamado": 9,
        "ultima_nota_satisfacao": 6}})

    customer_id: str | None = Field(None, description="Opcional. Se informado e existir no banco, a previsão é registrada.")
    tipo_cliente: Literal["Pessoa Física", "Pequena Empresa", "Média Empresa"]
    plano: Plano
    funcionalidades_disponiveis: int = Field(ge=1, le=100)
    ciclo_contrato: Literal["Mensal", "Anual"]
    forma_pagamento: FormaPagamento
    renovacao_automatica: int = Field(ge=0, le=1)
    canal_aquisicao: str
    tempo_cliente_meses: float = Field(ge=0, description="Meses desde a contratação")
    acessos_m1: float = Field(ge=0, description="Acessos no mês de referência")
    acessos_m2: float | None = Field(None, ge=0, description="Acessos 1 mês antes")
    acessos_m3: float | None = Field(None, ge=0)
    acessos_m4: float | None = Field(None, ge=0)
    acessos_m5: float | None = Field(None, ge=0)
    acessos_m6: float | None = Field(None, ge=0, description="Acessos 5 meses antes")
    sessoes_m1: float = Field(ge=0)
    minutos_uso_m1: float = Field(ge=0)
    minutos_uso_m2: float | None = Field(None, ge=0)
    funcionalidades_usadas_m1: float = Field(ge=0)
    dias_ativos_m1: float = Field(ge=0, le=31)
    dias_desde_ultimo_acesso: float | None = Field(None, ge=0)
    faturas_6m: float = Field(ge=0)
    faturas_atrasadas_6m: float = Field(ge=0)
    dias_atraso_medio_6m: float | None = Field(None, ge=0)
    faturas_em_aberto: float = Field(ge=0)
    valor_total_pago: float = Field(ge=0)
    valor_mensal: float = Field(gt=0)
    reajuste_pct_6m: float | None = Field(None, ge=0, le=1)
    tickets_3m: float = Field(ge=0)
    tickets_6m: float = Field(ge=0)
    tickets_tecnicos_6m: float = Field(ge=0)
    reclamacoes_6m: float = Field(ge=0)
    tempo_medio_resolucao_h_6m: float | None = Field(None, ge=0)
    avaliacao_media_6m: float | None = Field(None, ge=1, le=5)
    dias_desde_ultimo_chamado: float | None = Field(None, ge=0)
    ultima_nota_satisfacao: float | None = Field(None, ge=0, le=10)


# --------------------------------------------------------------------------- #
# Saída de previsão
# --------------------------------------------------------------------------- #
class Factor(BaseModel):
    feature: str
    label: str
    category: str
    category_label: str
    impact: float = Field(description="Contribuição SHAP (log-odds). Positivo aumenta o risco.")
    value: str
    text: str = Field(description="Explicação em linguagem de negócio")
    acionavel: bool = Field(description="Se a empresa pode influenciar esse fator com uma ação")


class Recommendation(BaseModel):
    prioridade: str
    acao_sugerida: str
    aviso: str


class PredictionResult(BaseModel):
    customer_id: str | None = None
    data_referencia: date | None = None
    probabilidade: float = Field(ge=0, le=1, description="Probabilidade estimada de cancelamento em 60 dias")
    nivel_risco: RiskLevel
    acima_do_threshold: bool
    threshold: float
    faixas_risco: dict[str, float]
    model_version: str
    principal_fator: str | None = None
    resumo: str | None = Field(None, description="Explicação em uma frase, em linguagem de negócio")
    categoria_principal: str | None = None
    fatores_risco: list[Factor] = []
    fatores_protecao: list[Factor] = []
    recomendacao: Recommendation | None = None
    prediction_id: int | None = None
    aviso: str


class BatchRequest(BaseModel):
    clientes: list[CustomerFeatures] = Field(min_length=1)
    explicar: bool = Field(False, description="Incluir fatores e recomendação (mais lento)")


class BatchSummary(BaseModel):
    total: int
    alto: int
    medio: int
    baixo: int
    probabilidade_media: float


class BatchResponse(BaseModel):
    resumo: BatchSummary
    resultados: list[PredictionResult]


# --------------------------------------------------------------------------- #
# Simulação
# --------------------------------------------------------------------------- #
class SimulationRequest(BaseModel):
    """Alterações hipotéticas. Somente os campos informados são alterados."""

    model_config = ConfigDict(json_schema_extra={"example": {"aumento_uso_percentual": 40, "remover_atrasos": True}})

    aumento_uso_percentual: float | None = Field(None, ge=-100, le=300,
                                                 description="Varia acessos, sessões, minutos e dias ativos recentes")
    remover_atrasos: bool | None = Field(None, description="Zera atrasos e faturas em aberto")
    reclamacoes_6m: int | None = Field(None, ge=0)
    tickets_3m: int | None = Field(None, ge=0)
    tickets_6m: int | None = Field(None, ge=0)
    tickets_tecnicos_6m: int | None = Field(None, ge=0)
    plano: Plano | None = None
    forma_pagamento: FormaPagamento | None = None
    renovacao_automatica: bool | None = None
    ciclo_contrato: Literal["Mensal", "Anual"] | None = None
    ultima_nota_satisfacao: int | None = Field(None, ge=0, le=10)
    dias_desde_ultimo_acesso: int | None = Field(None, ge=0)


class SimulationResult(BaseModel):
    customer_id: str
    data_referencia: date
    alteracoes_aplicadas: list[str]
    atual: PredictionResult
    simulado: PredictionResult
    variacao_pontos_percentuais: float
    aviso: str


# --------------------------------------------------------------------------- #
# Clientes
# --------------------------------------------------------------------------- #
class CustomerRiskRow(BaseModel):
    prediction_id: int
    customer_id: str
    nome: str
    plano: str
    regiao: str | None
    tipo_cliente: str
    ciclo_contrato: str
    valor_mensal: float | None
    tempo_cliente_meses: float | None
    probabilidade: float
    nivel_risco: RiskLevel
    principal_fator: str | None
    categoria_principal: str | None
    acao_sugerida: str | None
    data_referencia: date
    model_version: str
    origem: str | None = None
    churn_real: int | None = Field(None, description="Resultado observado em 60 dias (períodos históricos)")


class Page(BaseModel):
    total: int
    pagina: int
    tamanho_pagina: int


class CustomerList(Page):
    faixas_risco: dict[str, float]
    itens: list[CustomerRiskRow]


class UsagePoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    mes_referencia: date
    acessos: int
    sessoes: int
    minutos_uso: int
    funcionalidades_usadas: int
    dias_ativos: int


class PaymentPoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    mes_referencia: date
    valor_cobrado: float
    data_vencimento: date
    data_pagamento: date | None
    status: str
    dias_atraso: int | None = None


class TicketItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    ticket_id: str
    data_abertura: date
    categoria: str
    tempo_resolucao_horas: float | None
    avaliacao: int | None


class SurveyItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    data_resposta: date
    nota_satisfacao: int


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    data_referencia: date
    probabilidade: float
    nivel_risco: RiskLevel
    origem: str
    churn_real: int | None
    model_version: str


class CustomerProfile(BaseModel):
    customer_id: str
    nome: str
    tipo_cliente: str
    regiao: str | None
    plano: str
    ciclo_contrato: str
    forma_pagamento: str
    renovacao_automatica: bool
    canal_aquisicao: str
    data_inicio: date
    status: str
    tempo_cliente_meses: float
    valor_mensal: float | None


class RetentionActionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    customer_id: str
    prediction_id: int | None
    tipo_acao: str
    status: str
    responsavel: str | None
    observacoes: str | None
    resultado: str | None
    created_at: datetime
    updated_at: datetime


class CustomerDetail(BaseModel):
    perfil: CustomerProfile
    previsao_atual: PredictionResult | None
    historico_uso: list[UsagePoint]
    historico_pagamentos: list[PaymentPoint]
    chamados: list[TicketItem]
    pesquisas: list[SurveyItem]
    historico_previsoes: list[PredictionHistoryItem]
    acoes_retencao: list[RetentionActionOut]


# --------------------------------------------------------------------------- #
# Ações de retenção
# --------------------------------------------------------------------------- #
TipoAcao = Literal["Contato do Customer Success", "Treinamento / reonboarding", "Negociação financeira",
                   "Oferta de ajuste de plano", "Follow-up de suporte", "Conversa de renovação",
                   "Campanha de engajamento", "Revisão humana: sem ação", "Outra"]
StatusAcao = Literal["Planejada", "Em andamento", "Concluída", "Cancelada"]


class RetentionActionCreate(BaseModel):
    tipo_acao: TipoAcao
    responsavel: str | None = Field(None, max_length=80)
    observacoes: str | None = Field(None, max_length=2000)
    prediction_id: int | None = None


class RetentionActionUpdate(BaseModel):
    status: StatusAcao | None = None
    responsavel: str | None = Field(None, max_length=80)
    observacoes: str | None = Field(None, max_length=2000)
    resultado: Literal["Retido", "Cancelou", "Sem resposta"] | None = None


# --------------------------------------------------------------------------- #
# Modelo e dashboard
# --------------------------------------------------------------------------- #
class ModelInfo(BaseModel):
    version: str
    model_name: str
    trained_at: datetime
    target: str
    horizon_days: int
    threshold: float
    risk_thresholds: dict[str, float]
    best_params: dict
    imbalance_strategy: str | None
    business_assumptions: dict
    features: dict
    training_data: dict


class ConfusionMatrix(BaseModel):
    verdadeiro_positivo: int
    falso_positivo: int
    verdadeiro_negativo: int
    falso_negativo: int


class ModelMetrics(BaseModel):
    version: str
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    brier: float
    base_rate: float
    matriz_confusao: ConfusionMatrix
    validacao_cruzada: dict
    periodo_teste: str
    resultado_esperado: dict
    observacao: str


class Health(BaseModel):
    status: Literal["ok", "degradado"]
    database: str
    model: str
    model_version: str | None
    data_referencia: date
    timestamp: datetime
