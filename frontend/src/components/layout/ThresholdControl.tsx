import { SlidersHorizontal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { useRiskSettings } from "../../context/RiskSettings";
import { useThresholds } from "../../hooks/useApi";
import { fmtDec, fmtInt, fmtPct } from "../../lib/format";

/** Controle global dos limites de risco: alto (threshold) e médio, configuráveis na aplicação. */
export function ThresholdControl() {
  const { threshold, medio, setLimits } = useRiskSettings();
  const { data } = useThresholds();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const modelT = data?.threshold_modelo ?? 0.23;
  const modelM = data?.faixas_risco?.medio ?? 0.12;
  const current = threshold ?? modelT;
  const currentM = Math.min(medio ?? modelM, current);
  const [draft, setDraft] = useState(current);
  const [draftM, setDraftM] = useState(currentM);
  useEffect(() => { setDraft(current); setDraftM(currentM); }, [current, currentM, open]);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", esc); };
  }, [open]);
  const point = data?.pontos.find((p) => Math.abs(p.threshold - draft) < 0.005);

  return (
    <div className="relative" ref={ref}>
      <button className="btn-secondary py-1.5" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-haspopup="dialog">
        <SlidersHorizontal size={15} aria-hidden />
        <span className="hidden sm:inline">Limite de alto risco:</span>
        <span className="tabular font-semibold">{fmtDec(current)}</span>
        {(threshold != null || medio != null) && <span className="rounded bg-brand-50 px-1.5 text-[10px] font-semibold uppercase text-brand-700">ajustado</span>}
      </button>
      {open && (
        <div role="dialog" aria-label="Configurar limites de risco"
          className="absolute right-0 z-40 mt-2 w-[min(360px,calc(100vw-32px))] rounded-xl border border-line bg-surface p-4 shadow-[var(--shadow-pop)]">
          <p className="text-sm font-semibold">Limite de alto risco (threshold)</p>
          <p className="mt-1 text-xs leading-relaxed text-ink-2">
            Clientes com probabilidade estimada acima deste valor são classificados como <b>alto risco</b>.
            Valores menores encontram mais cancelamentos, mas geram mais alertas falsos.
          </p>
          <label htmlFor="thr" className="sr-only">Threshold</label>
          <input id="thr" type="range" min={0.05} max={0.7} step={0.01} value={draft}
            onChange={(e) => { const v = Number(e.target.value); setDraft(v); setDraftM((m) => Math.min(m, v)); }}
            className="mt-4 w-full accent-[var(--color-brand-600)]" />
          <div className="mt-1 flex justify-between text-[11px] text-muted"><span>0,05 · mais alertas</span><span>0,70 · mais conservador</span></div>
          <div className="mt-4 border-t border-line pt-3">
            <div className="flex items-baseline justify-between">
              <label htmlFor="thr-medio" className="text-xs font-semibold">Limite de médio risco</label>
              <span className="tabular text-xs font-semibold">{fmtDec(draftM)}</span>
            </div>
            <p className="mt-0.5 text-[11px] text-muted">Entre este valor e o limite alto, o cliente é <b>médio risco</b> (monitorar). Não pode passar do limite alto.</p>
            <input id="thr-medio" type="range" min={0.02} max={draft} step={0.01} value={draftM}
              onChange={(e) => setDraftM(Math.min(Number(e.target.value), draft))} className="mt-2 w-full accent-[var(--color-risk-mid)]" />
          </div>
          <dl className="mt-3 grid grid-cols-3 gap-2 rounded-lg bg-surface-2 p-3 text-center">
            <div><dt className="text-[11px] text-muted">Threshold</dt><dd className="tabular text-base font-semibold">{fmtDec(draft)}</dd></div>
            <div><dt className="text-[11px] text-muted">Precision</dt><dd className="tabular text-base font-semibold">{fmtPct(point?.precision, 0)}</dd></div>
            <div><dt className="text-[11px] text-muted">Recall</dt><dd className="tabular text-base font-semibold">{fmtPct(point?.recall, 0)}</dd></div>
          </dl>
          <p className="mt-2 text-xs text-ink-2">
            Hoje seriam sinalizados <b className="tabular">{fmtInt(point?.clientes_sinalizados_hoje)}</b> clientes.
            Recomendado pelo modelo: <b>{fmtDec(modelT)}</b> (alto) e <b>{fmtDec(modelM)}</b> (médio).
          </p>
          <div className="mt-4 flex items-center justify-between gap-2">
            <Link to="/modelo#threshold" className="text-xs font-medium text-brand-600 hover:underline" onClick={() => setOpen(false)}>Entender o trade-off</Link>
            <div className="flex gap-2">
              <button className="btn-ghost py-1.5 text-xs" onClick={() => { setLimits(null, null); setOpen(false); }}>Usar os do modelo</button>
              <button className="btn-primary py-1.5 text-xs" onClick={() => {
                setLimits(Math.abs(draft - modelT) < 0.001 ? null : draft, Math.abs(draftM - modelM) < 0.001 ? null : draftM);
                setOpen(false);
              }}>Aplicar</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
