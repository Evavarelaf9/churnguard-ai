import { AlertOctagon, Inbox, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const msg = error instanceof Error ? error.message : "Erro inesperado.";
  return (
    <div role="alert" className="flex flex-col items-center justify-center gap-3 rounded-xl border border-risk-high/30 bg-risk-high-bg/60 px-6 py-10 text-center">
      <AlertOctagon className="text-risk-high" size={28} aria-hidden />
      <div>
        <p className="font-semibold text-risk-high-ink">Não foi possível carregar os dados</p>
        <p className="mt-1 text-sm text-ink-2">{msg}</p>
      </div>
      {onRetry && (
        <button className="btn-secondary" onClick={onRetry}>
          <RefreshCw size={15} aria-hidden /> Tentar novamente
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      <span className="rounded-full bg-brand-50 p-3 text-brand-600">
        <Inbox size={22} aria-hidden />
      </span>
      <p className="font-semibold text-ink">{title}</p>
      {description && <p className="max-w-sm text-sm text-ink-2">{description}</p>}
      {action}
    </div>
  );
}

export function SkeletonBlock({ className = "h-48" }: { className?: string }) {
  return <div className={`skeleton w-full ${className}`} aria-hidden />;
}

export function Disclaimer({ children }: { children: ReactNode }) {
  return (
    <p className="flex items-start gap-2 rounded-lg border border-line bg-surface-2 px-3 py-2 text-xs leading-relaxed text-ink-2">
      <span aria-hidden className="mt-px">ⓘ</span>
      <span>{children}</span>
    </p>
  );
}
