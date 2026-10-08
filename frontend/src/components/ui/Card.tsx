import clsx from "clsx";
import type { ReactNode } from "react";

import { InfoTip } from "./InfoTip";

type Props = {
  title?: ReactNode;
  subtitle?: ReactNode;
  info?: string;
  actions?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
  as?: "section" | "div" | "article";
};

export function Card({ title, subtitle, info, actions, className, bodyClassName, children, as: Tag = "section" }: Props) {
  return (
    <Tag className={clsx("card min-w-0", className)}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 px-5 pt-4">
          <div className="min-w-0">
            {title && (
              <h2 className="flex items-center gap-1.5 text-[15px] font-semibold text-ink">
                {title}
                {info && <InfoTip text={info} />}
              </h2>
            )}
            {subtitle && <p className="mt-0.5 text-[13px] text-ink-2">{subtitle}</p>}
          </div>
          {actions && <div className="shrink-0">{actions}</div>}
        </header>
      )}
      <div className={clsx("px-5 pb-5", title ? "pt-3" : "pt-5", bodyClassName)}>{children}</div>
    </Tag>
  );
}
