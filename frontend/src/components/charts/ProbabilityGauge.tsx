import { fmtPct } from "../../lib/format";
import { RISK_STYLE } from "../../lib/risk";
import type { RiskLevel, RiskThresholds } from "../../lib/types";

/** Medidor semicircular com as três faixas de risco e um marcador na probabilidade estimada. */
export function ProbabilityGauge({ value, level, thresholds }: { value: number; level: RiskLevel; thresholds: RiskThresholds }) {
  const R = 80, CX = 100, CY = 96, W = 14;
  const scale = (v: number) => Math.min(v, 1); // escala linear 0–100%
  const pt = (v: number, r = R) => {
    const a = Math.PI * (1 - scale(v));
    return [CX + r * Math.cos(a), CY - r * Math.sin(a)] as const;
  };
  const arc = (from: number, to: number) => {
    const [x1, y1] = pt(from), [x2, y2] = pt(to);
    return `M ${x1} ${y1} A ${R} ${R} 0 0 1 ${x2} ${y2}`;
  };
  const [nx, ny] = pt(value, R);
  return (
    <figure className="flex flex-col items-center" aria-label={`Probabilidade estimada: ${fmtPct(value)}, ${RISK_STYLE[level].label}`}>
      <svg viewBox="0 0 200 112" className="w-full max-w-[240px]" role="img" aria-hidden>
        <path d={arc(0, thresholds.medio)} stroke={RISK_STYLE.Baixo.color} strokeWidth={W} fill="none" opacity={0.85} />
        <path d={arc(thresholds.medio, thresholds.alto)} stroke={RISK_STYLE.Médio.color} strokeWidth={W} fill="none" opacity={0.85} />
        <path d={arc(thresholds.alto, 1)} stroke={RISK_STYLE.Alto.color} strokeWidth={W} fill="none" opacity={0.85} />
        <circle cx={nx} cy={ny} r={9} fill="#fff" stroke="#16171a" strokeWidth={2.5} />
        <text x={CX} y={CY - 18} textAnchor="middle" fontSize="30" fontWeight="650" fill="#16171a">{fmtPct(value, 0)}</text>
        <text x={CX} y={CY + 2} textAnchor="middle" fontSize="9.5" fill="#7c7d84">prob. estimada · 60 dias</text>
        <text x={CX - R} y={CY + 14} textAnchor="middle" fontSize="8.5" fill="#7c7d84">0%</text>
        <text x={CX + R} y={CY + 14} textAnchor="middle" fontSize="8.5" fill="#7c7d84">100%</text>
      </svg>
    </figure>
  );
}
