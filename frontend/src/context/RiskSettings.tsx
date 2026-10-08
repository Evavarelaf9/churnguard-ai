import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

/**
 * Limites de risco escolhidos pelo usuário (null = usar os do modelo).
 * - threshold: limite de ALTO risco
 * - medio: limite de MÉDIO risco
 * Guardados no navegador apenas como preferência pessoal; a regra oficial fica no modelo.
 */
type Ctx = {
  threshold: number | null;
  medio: number | null;
  setThreshold: (t: number | null) => void;
  setLimits: (alto: number | null, medio: number | null) => void;
};
const RiskSettingsContext = createContext<Ctx>({ threshold: null, medio: null, setThreshold: () => {}, setLimits: () => {} });
const KEY = "churnguard.threshold";
const KEY_MEDIO = "churnguard.limite_medio";

function read(key: string): number | null {
  try {
    const v = localStorage.getItem(key);
    const n = v ? Number(v) : NaN;
    return Number.isFinite(n) && n > 0 && n < 1 ? n : null;
  } catch {
    return null;
  }
}

function write(key: string, v: number | null) {
  try {
    if (v == null) localStorage.removeItem(key);
    else localStorage.setItem(key, String(v));
  } catch { /* armazenamento indisponível: mantém só em memória */ }
}

export function RiskSettingsProvider({ children }: { children: ReactNode }) {
  const [threshold, setT] = useState<number | null>(() => read(KEY));
  const [medio, setM] = useState<number | null>(() => read(KEY_MEDIO));
  const value = useMemo<Ctx>(() => ({
    threshold,
    medio,
    setThreshold: (t) => { setT(t); write(KEY, t); },
    setLimits: (alto, med) => { setT(alto); setM(med); write(KEY, alto); write(KEY_MEDIO, med); },
  }), [threshold, medio]);
  return <RiskSettingsContext.Provider value={value}>{children}</RiskSettingsContext.Provider>;
}

export const useRiskSettings = () => useContext(RiskSettingsContext);
