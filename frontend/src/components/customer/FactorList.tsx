import type { Factor } from "../../lib/types";

export function FactorList({ factors, kind }: { factors: Factor[]; kind: "risco" | "protecao" }) {
  if (!factors.length) return <p className="text-sm text-muted">Nenhum fator relevante.</p>;
  const max = Math.max(...factors.map((f) => Math.abs(f.impact)));
  const color = kind === "risco" ? "var(--color-risk-high)" : "var(--color-brand-500)";
  return (
    <ul className="space-y-3">
      {factors.map((f) => (
        <li key={f.feature}>
          <div className="flex items-start justify-between gap-3">
            <p className="text-sm leading-snug text-ink">{f.text}</p>
            <span className="shrink-0 rounded-md bg-surface-2 px-1.5 py-0.5 text-[10.5px] font-medium text-ink-2">{f.category_label}</span>
          </div>
          <div className="mt-1.5 flex items-center gap-2">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-black/[0.05]" aria-hidden>
              <div className="h-full rounded-full" style={{ width: `${(Math.abs(f.impact) / max) * 100}%`, background: color }} />
            </div>
            {!f.acionavel && kind === "risco" && (
              <span className="text-[10.5px] text-muted" title="Característica que a empresa não muda com uma ação de retenção no curto prazo">estrutural</span>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
