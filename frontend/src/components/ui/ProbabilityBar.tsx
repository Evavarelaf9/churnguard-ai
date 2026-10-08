import { RISK_STYLE } from "../../lib/risk";
import { fmtPct } from "../../lib/format";
import type { RiskLevel } from "../../lib/types";

/** Barra compacta de probabilidade para tabelas. */
export function ProbabilityBar({ value, level }: { value: number; level: RiskLevel }) {
  return (
    <div className="flex items-center gap-2.5" title={`Probabilidade estimada de cancelamento em 60 dias: ${fmtPct(value)}`}>
      <span className="tabular w-12 text-right text-sm font-semibold text-ink">{fmtPct(value, 0)}</span>
      <span className="h-1.5 w-20 overflow-hidden rounded-full bg-black/[0.07]" aria-hidden>
        <span className="block h-full rounded-full" style={{ width: `${Math.max(value * 100, 2)}%`, background: RISK_STYLE[level].color }} />
      </span>
    </div>
  );
}
