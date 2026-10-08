import type { ReactNode } from "react";

type Row = { label: string; value: ReactNode; color?: string };

/** Conteúdo padrão do tooltip: título + linhas (cor como marcador, texto sempre em tinta neutra). */
export function TooltipBox({ title, rows }: { title: ReactNode; rows: Row[] }) {
  return (
    <div className="min-w-44 rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-[var(--shadow-pop)]">
      <p className="mb-1.5 font-semibold text-ink">{title}</p>
      <dl className="space-y-1">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center justify-between gap-4">
            <dt className="flex items-center gap-1.5 text-ink-2">
              {r.color && <span className="h-2 w-2 rounded-full" style={{ background: r.color }} aria-hidden />}
              {r.label}
            </dt>
            <dd className="tabular font-medium text-ink">{r.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
