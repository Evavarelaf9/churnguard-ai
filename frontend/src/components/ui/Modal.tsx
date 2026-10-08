import { X } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";

/** Modal acessível baseado em <dialog> (foco preso, Esc fecha). */
export function Modal({ open, onClose, title, children }: { open: boolean; onClose: () => void; title: string; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      aria-labelledby="modal-title"
      className="m-auto w-[min(520px,calc(100vw-32px))] rounded-2xl border border-line bg-surface p-0 text-ink shadow-[var(--shadow-pop)] backdrop:bg-navy-900/40 backdrop:backdrop-blur-[2px]"
    >
      <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
        <h2 id="modal-title" className="text-base font-semibold">{title}</h2>
        <button className="btn-ghost p-1.5" onClick={onClose} aria-label="Fechar">
          <X size={18} aria-hidden />
        </button>
      </div>
      <div className="p-5">{children}</div>
    </dialog>
  );
}
