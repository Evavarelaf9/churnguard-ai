// Contratos da API (espelham backend/app/schemas/schemas.py)

export type RiskLevel = "Baixo" | "Médio" | "Alto";
export type RiskThresholds = { medio: number; alto: number };

export interface Factor {
  feature: string;
  label: string;
  category: string;
  category_label: string;
  impact: number;
  value: string;
  text: string;
  acionavel: boolean;
}

export interface Recommendation {
  prioridade: string;
  acao_sugerida: string;
  aviso: string;
}

export interface PredictionResult {
  customer_id: string | null;
  data_referencia: string | null;
  probabilidade: number;
  nivel_risco: RiskLevel;
  acima_do_threshold: boolean;
  threshold: number;
  faixas_risco: RiskThresholds;
  model_version: string;
  principal_fator: string | null;
  resumo?: string | null;
  categoria_principal: string | null;
  fatores_risco: Factor[];
  fatores_protecao: Factor[];
  recomendacao: Recommendation | null;
  prediction_id: number | null;
  aviso: string;
}

export interface CustomerRow {
  prediction_id: number;
  customer_id: string;
  nome: string;
  plano: string;
  regiao: string | null;
  tipo_cliente: string;
  ciclo_contrato: string;
  valor_mensal: number | null;
  tempo_cliente_meses: number | null;
  probabilidade: number;
  nivel_risco: RiskLevel;
  principal_fator: string | null;
  categoria_principal: string | null;
  acao_sugerida: string | null;
  data_referencia: string;
  model_version: string;
  origem?: "batch" | "api" | "historico";
  churn_real?: number | null;
}

export interface CustomerList {
  total: number;
  pagina: number;
  tamanho_pagina: number;
  faixas_risco: RiskThresholds;
  itens: CustomerRow[];
}

export interface CustomerFilters {
  nivel_risco?: RiskLevel[];
  plano?: string;
  regiao?: string;
  valor_min?: number;
  valor_max?: number;
  data_referencia?: string;
  busca?: string;
  ordenar?: "probabilidade" | "receita_em_risco" | "valor_mensal" | "nome";
  pagina?: number;
  tamanho_pagina?: number;
}

export interface UsagePoint {
  mes_referencia: string;
  acessos: number;
  sessoes: number;
  minutos_uso: number;
  funcionalidades_usadas: number;
  dias_ativos: number;
}

export interface PaymentPoint {
  mes_referencia: string;
  valor_cobrado: number;
  data_vencimento: string;
  data_pagamento: string | null;
  status: "Pago em dia" | "Pago com atraso" | "Em aberto";
  dias_atraso: number | null;
}

export interface TicketItem {
  ticket_id: string;
  data_abertura: string;
  categoria: string;
  tempo_resolucao_horas: number | null;
  avaliacao: number | null;
}

export interface PredictionHistoryItem {
  id: number;
  data_referencia: string;
  probabilidade: number;
  nivel_risco: RiskLevel;
  origem: "batch" | "api" | "historico";
  churn_real: number | null;
  model_version: string;
}

export type ActionStatus = "Planejada" | "Em andamento" | "Concluída" | "Cancelada";
export type ActionResult = "Retido" | "Cancelou" | "Sem resposta";

export const ACTION_TYPES = [
  "Contato do Customer Success",
  "Treinamento / reonboarding",
  "Negociação financeira",
  "Oferta de ajuste de plano",
  "Follow-up de suporte",
  "Conversa de renovação",
  "Campanha de engajamento",
  "Revisão humana: sem ação",
  "Outra",
] as const;
export type ActionType = (typeof ACTION_TYPES)[number];

export interface RetentionAction {
  id: number;
  customer_id: string;
  prediction_id: number | null;
  tipo_acao: ActionType;
  status: ActionStatus;
  responsavel: string | null;
  observacoes: string | null;
  resultado: ActionResult | null;
  created_at: string;
  updated_at: string;
}

export interface CustomerProfile {
  customer_id: string;
  nome: string;
  tipo_cliente: string;
  regiao: string | null;
  plano: string;
  ciclo_contrato: string;
  forma_pagamento: string;
  renovacao_automatica: boolean;
  canal_aquisicao: string;
  data_inicio: string;
  status: string;
  tempo_cliente_meses: number;
  valor_mensal: number | null;
}

export interface CustomerDetail {
  perfil: CustomerProfile;
  previsao_atual: PredictionResult | null;
  historico_uso: UsagePoint[];
  historico_pagamentos: PaymentPoint[];
  chamados: TicketItem[];
  pesquisas: { data_resposta: string; nota_satisfacao: number }[];
  historico_previsoes: PredictionHistoryItem[];
  acoes_retencao: RetentionAction[];
}

export interface SimulationChanges {
  aumento_uso_percentual?: number;
  remover_atrasos?: boolean;
  reclamacoes_6m?: number;
  tickets_3m?: number;
  tickets_6m?: number;
  plano?: string;
  forma_pagamento?: string;
  renovacao_automatica?: boolean;
  ciclo_contrato?: "Mensal" | "Anual";
  ultima_nota_satisfacao?: number;
}

export interface SimulationResult {
  customer_id: string;
  data_referencia: string;
  alteracoes_aplicadas: string[];
  atual: PredictionResult;
  simulado: PredictionResult;
  variacao_pontos_percentuais: number;
  aviso: string;
}

export interface GroupStat {
  grupo: string;
  clientes: number;
  probabilidade_media: number;
  alto_risco: number;
  receita_mensal: number;
  receita_em_risco: number;
}

export interface ConfusionMatrix {
  verdadeiro_positivo: number;
  falso_positivo: number;
  verdadeiro_negativo: number;
  falso_negativo: number;
}

export interface DashboardSummary {
  data_referencia: string;
  model_version: string;
  faixas_risco: RiskThresholds;
  faixas_risco_modelo: RiskThresholds;
  kpis: {
    clientes_monitorados: number;
    alto_risco: number;
    medio_risco: number;
    baixo_risco: number;
    clientes_atencao: number;
    taxa_churn_estimada: number;
    receita_mensal_total: number;
    receita_mensal_em_risco: number;
    receita_esperada_perdida: number;
    previsoes_realizadas: number;
  };
  distribuicao_risco: { nivel: RiskLevel; clientes: number; percentual: number; receita_mensal: number }[];
  histograma_probabilidade: { faixa: string; clientes: number }[];
  principais_fatores: { fator: string; categoria: string; clientes: number; percentual: number }[];
  importancia_global: {
    feature: string;
    label: string;
    permutation_importance: number;
    mean_abs_shap: number;
    direction: number | null;
  }[];
  categorias_alto_risco: { categoria: string; clientes: number }[];
  por_plano: GroupStat[];
  por_faixa_valor: GroupStat[];
  por_comportamento: GroupStat[];
  por_regiao: GroupStat[];
  evolucao_churn: { mes: string; clientes_ativos: number; cancelamentos: number; taxa_churn: number }[];
  previsto_vs_real: { data_referencia: string; churn_previsto: number; churn_real: number }[];
  metricas_modelo: {
    nome: string;
    roc_auc: number;
    pr_auc: number;
    precision: number;
    recall: number;
    f1: number;
    threshold: number;
    matriz_confusao: ConfusionMatrix;
  };
}

export interface ModelInfo {
  version: string;
  model_name: string;
  trained_at: string;
  target: string;
  horizon_days: number;
  threshold: number;
  risk_thresholds: RiskThresholds;
  best_params: Record<string, unknown>;
  imbalance_strategy: string | null;
  business_assumptions: { custo_acao: number; taxa_sucesso_retencao: number; meses_receita_preservada: number };
  features: { categorical: string[]; numeric: string[]; binary: string[]; excluded: Record<string, string> };
  training_data: Record<string, unknown>;
}

export interface ModelMetrics {
  version: string;
  threshold: number;
  roc_auc: number;
  pr_auc: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  brier: number;
  base_rate: number;
  matriz_confusao: ConfusionMatrix;
  validacao_cruzada: Record<string, number>;
  periodo_teste: string;
  resultado_esperado: Record<string, number>;
  observacao: string;
}

export interface ThresholdPoint {
  threshold: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  percentual_sinalizado: number | null;
  clientes_sinalizados_hoje: number;
  abordados_historico: number;
  churns_identificados: number;
  churns_totais: number;
}

export interface ThresholdTradeoff {
  threshold_modelo: number;
  faixas_risco: RiskThresholds;
  base_observacoes: number;
  taxa_real: number | null;
  clientes_ativos: number;
  pontos: ThresholdPoint[];
  nota: string;
}

export interface FilterOptions {
  planos: string[];
  regioes: string[];
  niveis_risco: RiskLevel[];
  datas_referencia: string[];
  faixas_valor: { rotulo: string; min: number; max: number | null }[];
}

export interface Health {
  status: "ok" | "degradado";
  database: string;
  model: string;
  model_version: string | null;
  data_referencia: string;
}

// ---------------------------------------------------------------- Monitoramento
export type MonitoringStatus = "ok" | "atenção" | "crítico";
export type DriftStatus = "estável" | "moderado" | "significativo";

export interface MonitoringAlert {
  nivel: "crítico" | "atenção" | "info";
  tipo: string;
  titulo: string;
  detalhe: string;
  recomendacao: string;
}

export interface MonthlyPerformance {
  data_referencia: string;
  previsoes: number;
  churn_real: number;
  churn_previsto: number;
  gap_calibracao: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  roc_auc: number | null;
  pr_auc: number | null;
  vp: number;
  fp: number;
  fn: number;
  vn: number;
}

export interface DriftFeature {
  feature: string;
  label: string;
  tipo: string;
  psi: number;
  status: DriftStatus;
  media_referencia: number | null;
  media_atual: number | null;
  categoria_referencia?: string;
  ausentes_referencia: number;
  ausentes_atual: number;
  importante: boolean;
  ranking_importancia: number | null;
}

export interface MonitoringSummary {
  data_referencia: string;
  model_version: string;
  status: MonitoringStatus;
  alertas: MonitoringAlert[];
  politica: Record<string, number>;
  referencia_drift: string;
  linha_de_base: { precision: number; recall: number; f1: number; roc_auc: number; pr_auc: number; base_rate: number };
  desempenho_mensal: MonthlyPerformance[];
  matriz_acumulada: { vp: number; fp: number; fn: number; vn: number };
  previsoes_aguardando_resultado: number;
  drift_atual: { data_referencia: string; atributos: DriftFeature[] };
  drift_temporal: { data_referencia: string; origem: string; psi: Record<string, number> }[];
  volume: { data_referencia: string; origem: string; previsoes: number }[];
  distribuicao_previsoes: { data_referencia: string; probabilidade_media: number; percentual_alto: number }[];
  rotulos: Record<string, string>;
}

export interface MonitoringRun {
  id: number;
  executado_em: string;
  data_referencia: string;
  model_version: string;
  status: MonitoringStatus;
  alertas: number;
}
