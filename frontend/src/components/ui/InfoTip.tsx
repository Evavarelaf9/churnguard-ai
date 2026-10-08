import { Info } from "lucide-react";
import { useId, useState } from "react";

/** Tooltip acessível: abre com hover, foco de teclado ou toque. */
export function InfoTip({ text, label = "Mais informações" }: { text: string; label?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={label}
        aria-describedby={open ? id : undefined}
        className="rounded-full text-muted hover:text-ink-2"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((o) => !o)}
      >
        <Info size={14} aria-hidden />
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className="absolute left-1/2 top-full z-50 mt-2 w-64 -translate-x-1/2 rounded-lg bg-navy-800 px-3 py-2 text-xs font-normal leading-relaxed text-white shadow-[var(--shadow-pop)]"
        >
          {text}
        </span>
      )}
    </span>
  );
}
