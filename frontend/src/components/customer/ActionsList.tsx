import clsx from "clsx";
import { Link } from "react-router-dom";

import { useToast } from "../../context/Toast";
import { useUpdateAction } from "../../hooks/useApi";
import { fmtDate } from "../../lib/format";
import type { ActionStatus, RetentionAction } from "../../lib/types";

export const STATUS_STYLE: Record<ActionStatus, string> = {
  Planejada: "bg-brand-50 text-brand-800",
  "Em andamento": "bg-risk-mid-bg text-risk-mid-ink",
  Concluída: "bg-risk-low-bg text-risk-low-ink",
  Cancelada: "bg-black/5 text-ink-2",
};
const STATUSES: ActionStatus[] = ["Planejada", "Em andamento", "Concluída", "Cancelada"];
const RESULTS = ["", "Retido", "Cancelou", "Sem resposta"] as const;

export function ActionsList({ actions, showCustomer = false }: { actions: RetentionAction[]; showCustomer?: boolean }) {
  const update = useUpdateAction();
  const { notify } = useToast();
  const opts = { onSuccess: () => notify("Ação atualizada."), onError: (e: Error) => notify(e.message, "error") };
  if (!actions.length) return <p className="py-4 text-sm text-muted">Nenhuma ação registrada.</p>;
  return (
    <ul className="divide-y divide-line">
      {actions.map((a) => (
        <li key={a.id} className="flex flex-col gap-2 py-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <p className="text-sm font-medium text-ink">{a.tipo_acao}</p>
            <p className="text-xs text-muted">
              {showCustomer && <Link to={`/clientes/${a.customer_id}`} className="font-medium text-brand-600 hover:underline">{a.customer_id}</Link>}
              {showCustomer && " · "}{fmtDate(a.created_at)} · {a.responsavel ?? "sem responsável"}
            </p>
            {a.observacoes && <p className="mt-0.5 line-clamp-1 text-xs text-ink-2">{a.observacoes}</p>}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <select aria-label={`Status da ação ${a.id}`} value={a.status} disabled={update.isPending}
              onChange={(e) => update.mutate({ id: a.id, status: e.target.value }, opts)}
              className={clsx("rounded-full border-0 px-2.5 py-1 text-xs font-medium", STATUS_STYLE[a.status])}>
              {STATUSES.map((s) => <option key={s}>{s}</option>)}
            </select>
            <select aria-label={`Resultado da ação ${a.id}`} value={a.resultado ?? ""} disabled={update.isPending}
              onChange={(e) => update.mutate({ id: a.id, resultado: e.target.value || null }, opts)}
              className="rounded-full border border-line bg-surface px-2.5 py-1 text-xs text-ink-2">
              {RESULTS.map((r) => <option key={r} value={r}>{r || "Resultado…"}</option>)}
            </select>
          </div>
        </li>
      ))}
    </ul>
  );
}
