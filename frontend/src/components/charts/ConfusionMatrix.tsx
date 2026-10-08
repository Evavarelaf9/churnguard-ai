import { fmtInt, fmtPct } from "../../lib/format";
import type { ConfusionMatrix as CM } from "../../lib/types";

export function ConfusionMatrix({ m }: { m: CM }) {
  const cell = (v: number, label: string, hint: string, good: boolean) => (
    <div className={`rounded-lg p-3 text-center ${good ? "bg-brand-50" : "bg-[#fdf0ea]"}`}>
      <p className="tabular text-xl font-semibold text-ink">{fmtInt(v)}</p>
      <p className="text-[11px] font-medium text-ink-2">{label}</p>
      <p className="mt-0.5 text-[10.5px] leading-snug text-muted">{hint}</p>
    </div>
  );
  const precision = m.verdadeiro_positivo / Math.max(m.verdadeiro_positivo + m.falso_positivo, 1);
  const recall = m.verdadeiro_positivo / Math.max(m.verdadeiro_positivo + m.falso_negativo, 1);
  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] items-center gap-2 text-xs">
        <span />
        <span className="text-center font-medium text-ink-2">Previsto: permanece</span>
        <span className="text-center font-medium text-ink-2">Previsto: cancela</span>
        <span className="pr-1 text-right font-medium text-ink-2">Real:<br />permaneceu</span>
        {cell(m.verdadeiro_negativo, "Verdadeiro negativo", "corretamente sem alerta", true)}
        {cell(m.falso_positivo, "Falso positivo", "alerta sem necessidade", false)}
        <span className="pr-1 text-right font-medium text-ink-2">Real:<br />cancelou</span>
        {cell(m.falso_negativo, "Falso negativo", "cancelamento não detectado", false)}
        {cell(m.verdadeiro_positivo, "Verdadeiro positivo", "risco detectado a tempo", true)}
      </div>
      <p className="mt-3 text-xs text-ink-2">
        Precision <b className="tabular">{fmtPct(precision)}</b> · Recall <b className="tabular">{fmtPct(recall)}</b> — período de teste, dados nunca vistos no treino.
      </p>
    </div>
  );
}
