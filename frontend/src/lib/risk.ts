import type { RiskLevel, RiskThresholds } from "./types";

export const RISK_ORDER: RiskLevel[] = ["Alto", "Médio", "Baixo"];

export const RISK_STYLE: Record<RiskLevel, { color: string; ink: string; bg: string; label: string }> = {
  Alto: { color: "var(--color-risk-high)", ink: "var(--color-risk-high-ink)", bg: "var(--color-risk-high-bg)", label: "Alto risco" },
  Médio: { color: "var(--color-risk-mid)", ink: "var(--color-risk-mid-ink)", bg: "var(--color-risk-mid-bg)", label: "Médio risco" },
  Baixo: { color: "var(--color-risk-low)", ink: "var(--color-risk-low-ink)", bg: "var(--color-risk-low-bg)", label: "Baixo risco" },
};

/** Mesma regra do backend (ml/src/evaluation/metrics.classify_risk). */
export function classifyRisk(p: number, th: RiskThresholds): RiskLevel {
  if (p >= th.alto) return "Alto";
  if (p >= th.medio) return "Médio";
  return "Baixo";
}

/** Faixas efetivas quando o usuário altera o limite de alto risco. */
export function effectiveThresholds(model: RiskThresholds, alto: number | null): RiskThresholds {
  if (alto == null) return model;
  return { alto, medio: Math.min(model.medio, alto) };
}

export const CATEGORY_LABEL: Record<string, string> = {
  uso: "Utilização",
  financeiro: "Financeiro",
  suporte: "Suporte e satisfação",
  preco: "Preço e plano",
  contrato: "Contrato e ciclo de vida",
  perfil: "Perfil",
};
