import clsx from "clsx";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { InfoTip } from "./InfoTip";

type Props = {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  icon?: LucideIcon;
  accent?: string;
  info?: string;
  loading?: boolean;
};

export function KpiCard({ label, value, hint, icon: Icon, accent = "var(--color-brand-500)", info, loading }: Props) {
  return (
    <div className="card relative overflow-hidden p-4">
      <span className="absolute inset-y-0 left-0 w-1" style={{ background: accent }} aria-hidden />
      <div className="flex items-center justify-between gap-2">
        <p className="flex items-center gap-1.5 text-[13px] font-medium text-ink-2">
          {label}
          {info && <InfoTip text={info} />}
        </p>
        {Icon && <Icon size={17} className="text-muted" aria-hidden />}
      </div>
      {loading ? (
        <div className="skeleton mt-2 h-8 w-24" />
      ) : (
        <p className={clsx("mt-1.5 text-[26px] font-semibold leading-tight tracking-tight text-ink")}>{value}</p>
      )}
      {hint && !loading && <p className="mt-1 text-xs text-muted">{hint}</p>}
    </div>
  );
}
