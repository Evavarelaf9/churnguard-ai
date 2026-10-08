import clsx from "clsx";
import { useState } from "react";

import { ActionsList, STATUS_STYLE } from "../components/customer/ActionsList";
import { Card } from "../components/ui/Card";
import { PageHeader } from "../components/ui/PageHeader";
import { EmptyState, ErrorState, SkeletonBlock } from "../components/ui/States";
import { useActions } from "../hooks/useApi";
import { fmtInt, fmtPct } from "../lib/format";
import type { ActionStatus } from "../lib/types";

const STATUSES: ActionStatus[] = ["Planejada", "Em andamento", "Concluída", "Cancelada"];

export default function Actions() {
  const [status, setStatus] = useState<ActionStatus | undefined>();
  const all = useActions();
  const { data, isLoading, isError, error, refetch } = useActions(status);
  const counts = STATUSES.map((s) => ({ s, n: all.data?.filter((a) => a.status === s).length ?? 0 }));
  const done = all.data?.filter((a) => a.resultado) ?? [];
  const retained = done.filter((a) => a.resultado === "Retido").length;

  return (
    <div>
      <PageHeader eyebrow="Decisão humana" title="Ações de retenção"
        description="Ações registradas pela equipe a partir das previsões. Cada ação fica vinculada à previsão que a motivou, permitindo medir se as intervenções funcionam." />
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-5">
        {counts.map(({ s, n }) => (
          <button key={s} onClick={() => setStatus(status === s ? undefined : s)} aria-pressed={status === s}
            className={clsx("card p-4 text-left transition-shadow hover:shadow-md", status === s && "ring-2 ring-brand-500")}>
            <span className={clsx("rounded-full px-2 py-0.5 text-[11px] font-medium", STATUS_STYLE[s])}>{s}</span>
            <p className="tabular mt-2 text-2xl font-semibold">{fmtInt(n)}</p>
          </button>
        ))}
        <div className="card p-4">
          <span className="text-[11px] font-medium text-ink-2">Taxa de retenção registrada</span>
          <p className="tabular mt-2 text-2xl font-semibold">{done.length ? fmtPct(retained / done.length, 0) : "—"}</p>
          <p className="text-[11px] text-muted">{fmtInt(retained)} de {fmtInt(done.length)} com resultado</p>
        </div>
      </div>
      <Card title={status ? `Ações — ${status}` : "Todas as ações"} subtitle="Atualize o status e o resultado diretamente na lista">
        {isError ? <ErrorState error={error} onRetry={() => refetch()} />
          : isLoading ? <SkeletonBlock className="h-64" />
            : !data?.length ? <EmptyState title="Nenhuma ação" description="Registre ações a partir da página de um cliente." />
              : <ActionsList actions={data} showCustomer />}
      </Card>
      <p className="mt-3 text-xs text-muted">A taxa acima é descritiva: sem grupo de controle, não mede o efeito causal das ações.</p>
    </div>
  );
}
