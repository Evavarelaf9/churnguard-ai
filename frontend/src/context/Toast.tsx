import { CheckCircle2, X, XCircle } from "lucide-react";
import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";

type Kind = "success" | "error";
type Toast = { id: number; kind: Kind; message: string };
type Ctx = { notify: (message: string, kind?: Kind) => void };

const ToastContext = createContext<Ctx>({ notify: () => {} });

/** Feedback breve e acessível (aria-live) após ações do usuário. */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const seq = useRef(0);
  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const notify = useCallback((message: string, kind: Kind = "success") => {
    const id = ++seq.current;
    setToasts((t) => [...t.slice(-2), { id, kind, message }]);
    window.setTimeout(() => dismiss(id), 4000);
  }, [dismiss]);

  return (
    <ToastContext.Provider value={{ notify }}>
      {children}
      <div aria-live="polite" role="status" className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-[min(360px,calc(100vw-32px))] flex-col gap-2">
        {toasts.map((t) => (
          <div key={t.id} className="pointer-events-auto flex items-start gap-2 rounded-lg border border-line bg-surface p-3 text-sm shadow-[var(--shadow-pop)]">
            {t.kind === "success"
              ? <CheckCircle2 size={18} className="mt-0.5 shrink-0 text-risk-low" aria-hidden />
              : <XCircle size={18} className="mt-0.5 shrink-0 text-risk-high" aria-hidden />}
            <p className="flex-1 text-ink">{t.message}</p>
            <button onClick={() => dismiss(t.id)} className="text-muted hover:text-ink" aria-label="Fechar aviso"><X size={16} /></button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export const useToast = () => useContext(ToastContext);
