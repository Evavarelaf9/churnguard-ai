import clsx from "clsx";
import { AlertTriangle, CheckCircle2, CircleAlert } from "lucide-react";

import { RISK_STYLE } from "../../lib/risk";
import type { RiskLevel } from "../../lib/types";

const ICON = { Alto: AlertTriangle, Médio: CircleAlert, Baixo: CheckCircle2 } as const;

/** Nível de risco sempre com ícone + texto (nunca só cor). */
export function RiskBadge({ level, size = "md", className }: { level: RiskLevel; size?: "sm" | "md" | "lg"; className?: string }) {
  const s = RISK_STYLE[level];
  const Icon = ICON[level];
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full font-medium",
        size === "sm" && "px-2 py-0.5 text-[11px]",
        size === "md" && "px-2.5 py-0.5 text-xs",
        size === "lg" && "px-3 py-1 text-sm",
        className,
      )}
      style={{ background: s.bg, color: s.ink }}
    >
      <Icon size={size === "lg" ? 15 : 12} aria-hidden style={{ color: s.color }} />
      {s.label}
    </span>
  );
}
