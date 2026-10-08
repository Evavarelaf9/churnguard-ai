import type { ReactNode } from "react";

export type BarItem = { key: string; label: ReactNode; value: number; display: ReactNode; sub?: ReactNode; color?: string };

/** Barras horizontais em HTML: rótulos longos legíveis, acessível para leitores de tela. */
export function BarList({ items, max, ariaLabel }: { items: BarItem[]; max?: number; ariaLabel: string }) {
  const m = max ?? Math.max(...items.map((i) => i.value), 1e-9);
  return (
    <ul className="space-y-3" aria-label={ariaLabel}>
      {items.map((i) => (
        <li key={i.key}>
          <div className="mb-1 flex items-baseline justify-between gap-3 text-[13px]">
            <span className="min-w-0 truncate text-ink">{i.label}</span>
            <span className="tabular shrink-0 font-semibold text-ink">{i.display}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-black/[0.05]" aria-hidden>
            <div className="h-full rounded-full" style={{ width: `${Math.max((i.value / m) * 100, 1.5)}%`, background: i.color ?? "var(--color-brand-500)" }} />
          </div>
          {i.sub && <p className="mt-0.5 text-[11px] text-muted">{i.sub}</p>}
        </li>
      ))}
    </ul>
  );
}
