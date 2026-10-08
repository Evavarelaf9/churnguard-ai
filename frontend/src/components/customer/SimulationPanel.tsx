import { ArrowRight, FlaskConical, RotateCcw } from "lucide-react";
import { useState } from "react";

import { useSimulation } from "../../hooks/useApi";
import { fmtPct } from "../../lib/format";
import { RISK_STYLE } from "../../lib/risk";
import type { CustomerProfile, SimulationChanges } from "../../lib/types";
import { Card } from "../ui/Card";
import { RiskBadge } from "../ui/RiskBadge";

const PLANS = ["Básico", "Essencial", "Profissional", "Empresarial"];
const PAYMENTS = ["Cartão de crédito", "Boleto", "Pix", "Débito automático"];

export function SimulationPanel({ customerId, profile, complaints, tickets3m }: { customerId: string; profile: CustomerProfile; complaints: number; tickets3m: number }) {
  const sim = useSimulation(customerId);
  const [uso, setUso] = useState(0);
  const [atrasos, setAtrasos] = useState(false);
  const [recl, setRecl] = useState<number | "">("");
  const [tick, setTick] = useState<number | "">("");
  const [plano, setPlano] = useState(profile.plano);
  const [pag, setPag] = useState(profile.forma_pagamento);
  const [renov, setRenov] = useState(profile.renovacao_automatica);

  const changes: SimulationChanges = {
    ...(uso !== 0 && { aumento_uso_percentual: uso }),
    ...(atrasos && { remover_atrasos: true }),
    ...(recl !== "" && { reclamacoes_6m: Number(recl) }),
    ...(tick !== "" && { tickets_3m: Number(tick) }),
    ...(plano !== profile.plano && { plano }),
    ...(pag !== profile.forma_pagamento && { forma_pagamento: pag }),
    ...(renov !== profile.renovacao_automatica && { renovacao_automatica: renov }),
  };
  const hasChanges = Object.keys(changes).length > 0;
  const reset = () => { setUso(0); setAtrasos(false); setRecl(""); setTick(""); setPlano(profile.plano); setPag(profile.forma_pagamento); setRenov(profile.renovacao_automatica); sim.reset(); };
  const r = sim.data;

  return (
    <Card title={<><FlaskConical size={17} className="text-brand-600" aria-hidden /> Simulação de retenção</>}
      subtitle="E se alguns sinais mudassem? Compare a probabilidade atual com um cenário hipotético.">
      <div role="note" className="mb-5 rounded-lg border border-risk-mid/40 bg-risk-mid-bg px-3 py-2 text-xs leading-relaxed text-risk-mid-ink">
        <b>Simulação hipotética.</b> Mostra como o modelo reagiria se esses sinais fossem diferentes — não é uma previsão garantida do comportamento do cliente, e nada é gravado.
      </div>
      <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
        <form className="grid gap-4 sm:grid-cols-2" onSubmit={(e) => { e.preventDefault(); if (hasChanges) sim.mutate(changes); }}>
          <div className="sm:col-span-2">
            <label className="label" htmlFor="uso">Frequência de utilização recente: <b className="tabular text-ink">{uso > 0 ? "+" : ""}{uso}%</b></label>
            <input id="uso" type="range" min={-50} max={150} step={10} value={uso} onChange={(e) => setUso(Number(e.target.value))} className="w-full accent-[var(--color-brand-600)]" />
            <div className="flex justify-between text-[11px] text-muted"><span>−50%</span><span>sem mudança</span><span>+150%</span></div>
          </div>
          <label className="flex items-center gap-2 text-sm sm:col-span-2">
            <input type="checkbox" checked={atrasos} onChange={(e) => setAtrasos(e.target.checked)} className="h-4 w-4 accent-[var(--color-brand-600)]" />
            Remover atrasos e faturas em aberto
          </label>
          <div>
            <label className="label" htmlFor="recl">Reclamações nos últimos 6 meses</label>
            <input id="recl" type="number" min={0} max={20} className="input" placeholder={`atual: ${complaints}`} value={recl}
              onChange={(e) => setRecl(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} />
          </div>
          <div>
            <label className="label" htmlFor="tick">Chamados nos últimos 3 meses</label>
            <input id="tick" type="number" min={0} max={30} className="input" placeholder={`atual: ${tickets3m}`} value={tick}
              onChange={(e) => setTick(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} />
          </div>
          <div>
            <label className="label" htmlFor="sim-plano">Plano</label>
            <select id="sim-plano" className="input" value={plano} onChange={(e) => setPlano(e.target.value)}>
              {PLANS.map((p) => <option key={p}>{p}</option>)}
            </select>
          </div>
          <div>
            <label className="label" htmlFor="sim-pag">Forma de pagamento</label>
            <select id="sim-pag" className="input" value={pag} onChange={(e) => setPag(e.target.value)}>
              {PAYMENTS.map((p) => <option key={p}>{p}</option>)}
            </select>
          </div>
          <label className="flex items-center gap-2 self-end pb-2 text-sm">
            <input type="checkbox" checked={renov} onChange={(e) => setRenov(e.target.checked)} className="h-4 w-4 accent-[var(--color-brand-600)]" />
            Renovação automática
          </label>
          <div className="flex gap-2 sm:col-span-2">
            <button type="submit" className="btn-primary" disabled={!hasChanges || sim.isPending}>{sim.isPending ? "Simulando…" : "Simular cenário"}</button>
            <button type="button" className="btn-ghost" onClick={reset}><RotateCcw size={15} aria-hidden /> Restaurar</button>
          </div>
        </form>

        <div className="rounded-xl border border-line bg-surface-2 p-5" aria-live="polite">
          {sim.isError && <p role="alert" className="text-sm text-risk-high-ink">{(sim.error as Error).message}</p>}
          {!r && !sim.isError && (
            <p className="flex h-full items-center justify-center text-center text-sm text-muted">Ajuste um ou mais sinais e clique em “Simular cenário”.</p>
          )}
          {r && (
            <div>
              <div className="flex items-center justify-around gap-3 text-center">
                <div>
                  <p className="text-xs text-muted">Atual</p>
                  <p className="tabular text-3xl font-semibold" style={{ color: RISK_STYLE[r.atual.nivel_risco].ink }}>{fmtPct(r.atual.probabilidade, 0)}</p>
                  <RiskBadge level={r.atual.nivel_risco} size="sm" />
                </div>
                <ArrowRight className="text-muted" aria-hidden />
                <div>
                  <p className="text-xs text-muted">Simulado</p>
                  <p className="tabular text-3xl font-semibold" style={{ color: RISK_STYLE[r.simulado.nivel_risco].ink }}>{fmtPct(r.simulado.probabilidade, 0)}</p>
                  <RiskBadge level={r.simulado.nivel_risco} size="sm" />
                </div>
              </div>
              <p className="mt-4 text-center text-sm">
                Variação de <b className="tabular">{r.variacao_pontos_percentuais > 0 ? "+" : ""}{r.variacao_pontos_percentuais.toLocaleString("pt-BR", { maximumFractionDigits: 1 })} p.p.</b>
              </p>
              <p className="mt-3 text-xs text-ink-2">Alterações: {r.alteracoes_aplicadas.join(" · ")}</p>
              {r.simulado.fatores_risco[0] && (
                <p className="mt-1 text-xs text-ink-2">Principal fator restante: <b>{r.simulado.principal_fator}</b></p>
              )}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
