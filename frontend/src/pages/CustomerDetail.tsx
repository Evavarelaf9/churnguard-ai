import { ArrowLeft, CalendarClock, ClipboardPlus, CreditCard, RefreshCw, Repeat, Timer } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { PaymentsChart, ProbabilityHistoryChart, UsageHistoryChart } from "../components/charts/Charts";
import { ProbabilityGauge } from "../components/charts/ProbabilityGauge";
import { ActionModal } from "../components/customer/ActionModal";
import { ActionsList } from "../components/customer/ActionsList";
import { FactorList } from "../components/customer/FactorList";
import { SimulationPanel } from "../components/customer/SimulationPanel";
import { Card } from "../components/ui/Card";
import { RiskBadge } from "../components/ui/RiskBadge";
import { EmptyState, ErrorState, SkeletonBlock } from "../components/ui/States";
import { useCustomer, useRecalculate } from "../hooks/useApi";
import { ApiError } from "../services/api";
import { fmtDate, fmtDec, fmtMoney, fmtTenure } from "../lib/format";

function Meta({ icon: Icon, label, value }: { icon: typeof Timer; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <Icon size={15} className="text-muted" aria-hidden />
      <span className="text-xs text-muted">{label}</span>
      <span className="text-sm font-medium text-ink">{value}</span>
    </div>
  );
}

export default function CustomerDetail() {
  const { id = "" } = useParams();
  const { data, isLoading, isError, error, refetch } = useCustomer(id);
  const recalc = useRecalculate(id);
  const [modal, setModal] = useState(false);

  if (isError) {
    if (error instanceof ApiError && error.status === 404)
      return <div className="card"><EmptyState title="Cliente não encontrado" description={`Não existe cliente com o ID ${id}.`} action={<Link to="/clientes" className="btn-primary mt-2">Voltar para clientes</Link>} /></div>;
    return <ErrorState error={error} onRetry={() => refetch()} />;
  }
  if (isLoading || !data) {
    return (
      <div className="space-y-4" aria-busy="true">
        <SkeletonBlock className="h-24" />
        <div className="grid gap-4 lg:grid-cols-3"><SkeletonBlock className="h-72" /><SkeletonBlock className="h-72" /><SkeletonBlock className="h-72" /></div>
        <SkeletonBlock className="h-64" />
      </div>
    );
  }

  const { perfil: c, previsao_atual: p } = data;
  const complaints = data.chamados.filter((t) => t.categoria === "Reclamação").length;
  const latestScore = data.pesquisas[0];
  const refDate = p?.data_referencia ? new Date(p.data_referencia) : new Date();
  const tickets3m = data.chamados.filter((t) => {
    const d = new Date(t.data_abertura);
    return d <= refDate && refDate.getTime() - d.getTime() <= 92 * 86_400_000;
  }).length;

  return (
    <div>
      <Link to="/clientes" className="mb-4 inline-flex items-center gap-1 text-sm text-ink-2 hover:text-ink"><ArrowLeft size={16} aria-hidden /> Clientes</Link>

      {/* Cabeçalho */}
      <header className="card mb-4 p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-2xl font-semibold tracking-tight">{c.nome}</h1>
              {p && <RiskBadge level={p.nivel_risco} />}
              {c.status !== "Ativo" && <span className="rounded-full bg-black/5 px-2.5 py-0.5 text-xs font-medium text-ink-2">{c.status}</span>}
            </div>
            <p className="mt-1 text-sm text-ink-2">{c.customer_id} · {c.tipo_cliente} · {c.regiao ?? "região não informada"} · plano <b className="text-ink">{c.plano}</b></p>
            <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2">
              <Meta icon={Timer} label="Cliente há" value={fmtTenure(c.tempo_cliente_meses)} />
              <Meta icon={CreditCard} label="Mensalidade" value={fmtMoney(c.valor_mensal)} />
              <Meta icon={CalendarClock} label="Contrato" value={`${c.ciclo_contrato} · desde ${fmtDate(c.data_inicio)}`} />
              <Meta icon={Repeat} label="Pagamento" value={`${c.forma_pagamento}${c.renovacao_automatica ? " · renovação automática" : ""}`} />
            </div>
          </div>
          <div className="flex shrink-0 gap-2">
            <button className="btn-secondary" onClick={() => recalc.mutate()} disabled={recalc.isPending || c.status !== "Ativo"}>
              <RefreshCw size={15} className={recalc.isPending ? "animate-spin" : ""} aria-hidden /> Recalcular risco
            </button>
            <button className="btn-primary" onClick={() => setModal(true)}><ClipboardPlus size={15} aria-hidden /> Registrar ação</button>
          </div>
        </div>
        {recalc.isError && <p role="alert" className="mt-3 text-sm text-risk-high-ink">{(recalc.error as Error).message}</p>}
        {recalc.isSuccess && <p role="status" className="mt-3 text-sm text-risk-low-ink">Risco recalculado com os dados mais recentes e registrado no histórico.</p>}
      </header>

      {!p ? (
        <div className="card mb-4"><EmptyState title="Sem previsão para este cliente" description="Clientes cancelados ou com menos de 30 dias de histórico não são pontuados." /></div>
      ) : (
        <div className="mb-4 grid gap-4 lg:grid-cols-[0.9fr_1.3fr_1fr]">
          <Card title="Probabilidade de churn" info="Probabilidade estimada de cancelamento nos próximos 60 dias, calculada a partir de padrões históricos. Não é uma certeza.">
            <ProbabilityGauge value={p.probabilidade} level={p.nivel_risco} thresholds={p.faixas_risco} />
            <div className="mt-2 flex justify-center"><RiskBadge level={p.nivel_risco} size="lg" /></div>
            <dl className="mt-4 grid grid-cols-2 gap-2 text-center text-xs">
              <div className="rounded-lg bg-surface-2 py-2"><dt className="text-muted">Referência</dt><dd className="font-medium">{fmtDate(p.data_referencia)}</dd></div>
              <div className="rounded-lg bg-surface-2 py-2"><dt className="text-muted">Limite alto risco</dt><dd className="tabular font-medium">{fmtDec(p.threshold)}</dd></div>
            </dl>
            <p className="mt-3 text-center text-[11px] text-muted">Modelo v{p.model_version}</p>
          </Card>

          <Card title="Por que este risco?" subtitle="Principais fatores da previsão (SHAP)" info="Cada fator mostra quanto contribuiu para aumentar ou reduzir o risco deste cliente em relação à média. Associação não implica causalidade.">
            {p.resumo && <p className="mb-4 rounded-lg border border-line bg-surface-2 p-3 text-sm leading-relaxed text-ink" data-testid="resumo">{p.resumo}</p>}
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-risk-high-ink">Aumentam o risco</h3>
            <FactorList factors={p.fatores_risco} kind="risco" />
            <h3 className="mb-2 mt-5 text-xs font-semibold uppercase tracking-wide text-brand-700">Reduzem o risco</h3>
            <FactorList factors={p.fatores_protecao} kind="protecao" />
          </Card>

          <Card title="Recomendação" subtitle="Sugestão baseada nos dados">
            {p.recomendacao && (
              <div className="flex h-full flex-col">
                <p className="rounded-lg bg-brand-50 p-4 text-[15px] leading-relaxed text-brand-900">{p.recomendacao.acao_sugerida}</p>
                <p className="mt-3 text-sm font-medium text-ink">{p.recomendacao.prioridade}</p>
                {p.principal_fator && <p className="mt-2 text-sm text-ink-2">Ponto de atenção principal: <b className="text-ink">{p.principal_fator}</b>.</p>}
                <p className="mt-4 border-t border-line pt-3 text-xs leading-relaxed text-muted">{p.recomendacao.aviso}</p>
                <button className="btn-secondary mt-4 w-full" onClick={() => setModal(true)}><ClipboardPlus size={15} aria-hidden /> Registrar ação</button>
              </div>
            )}
          </Card>
        </div>
      )}

      <div className="mb-4 grid gap-4 lg:grid-cols-2">
        <Card title="Utilização" subtitle="Acessos por mês · últimos 12 meses">
          {data.historico_uso.length ? <UsageHistoryChart data={data.historico_uso} /> : <EmptyState title="Sem dados de uso" />}
        </Card>
        <Card title="Pagamentos" subtitle="Dias de atraso por fatura · últimos 12 meses">
          {data.historico_pagamentos.length ? <PaymentsChart data={data.historico_pagamentos} /> : <EmptyState title="Sem faturas" />}
        </Card>
      </div>

      <div className="mb-4 grid gap-4 lg:grid-cols-2">
        <Card title="Evolução do risco" subtitle="Probabilidade estimada em cada data de referência"
          info="Inclui previsões do período de avaliação do modelo e as atuais. Pontos coloridos pelo nível de risco da época.">
          {p && data.historico_previsoes.length ? <ProbabilityHistoryChart data={data.historico_previsoes} thresholds={p.faixas_risco} /> : <EmptyState title="Sem histórico de previsões" />}
        </Card>
        <Card title="Chamados de suporte" subtitle={latestScore ? `Última nota de satisfação: ${latestScore.nota_satisfacao}/10 em ${fmtDate(latestScore.data_resposta)}` : "Sem pesquisa de satisfação recente"} bodyClassName="!px-0">
          {data.chamados.length ? (
            <div className="max-h-[250px] overflow-y-auto">
              <table className="w-full text-sm">
                <caption className="sr-only">Chamados recentes</caption>
                <thead className="sticky top-0 bg-surface">
                  <tr className="border-b border-line text-left text-xs text-ink-2">
                    <th scope="col" className="px-5 py-2 font-medium">Abertura</th>
                    <th scope="col" className="px-3 py-2 font-medium">Categoria</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">Resolução</th>
                    <th scope="col" className="px-5 py-2 text-right font-medium">Avaliação</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {data.chamados.map((t) => (
                    <tr key={t.ticket_id}>
                      <td className="tabular px-5 py-2 text-ink-2">{fmtDate(t.data_abertura)}</td>
                      <td className="px-3 py-2">
                        <span className={t.categoria === "Reclamação" ? "font-medium text-risk-high-ink" : "text-ink"}>{t.categoria}</span>
                      </td>
                      <td className="tabular px-3 py-2 text-right text-ink-2">{t.tempo_resolucao_horas != null ? `${Math.round(t.tempo_resolucao_horas)} h` : "—"}</td>
                      <td className="tabular px-5 py-2 text-right text-ink-2">{t.avaliacao != null ? `${t.avaliacao}/5` : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <EmptyState title="Nenhum chamado" description="Este cliente não abriu chamados recentemente." />}
        </Card>
      </div>

      {p && c.status === "Ativo" && (
        <div className="mb-4"><SimulationPanel customerId={c.customer_id} profile={c} complaints={complaints} tickets3m={tickets3m} /></div>
      )}

      <Card title="Ações de retenção" subtitle="Decisões registradas pela equipe para este cliente"
        actions={<button className="btn-secondary py-1.5 text-xs" onClick={() => setModal(true)}><ClipboardPlus size={14} aria-hidden /> Nova ação</button>}>
        <ActionsList actions={data.acoes_retencao} />
      </Card>

      {modal && <ActionModal open={modal} onClose={() => setModal(false)} customerId={c.customer_id} category={p?.categoria_principal} />}
    </div>
  );
}
