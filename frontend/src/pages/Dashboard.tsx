import { AlertTriangle, ArrowRight, Banknote, Percent, Users } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";

import { BarList } from "../components/charts/BarList";
import {
  ChurnTrendChart, GroupRiskChart, PredictedVsRealChart, ProbabilityHistogram,
} from "../components/charts/Charts";
import { ConfusionMatrix } from "../components/charts/ConfusionMatrix";
import { Card } from "../components/ui/Card";
import { KpiCard } from "../components/ui/KpiCard";
import { PageHeader } from "../components/ui/PageHeader";
import { ProbabilityBar } from "../components/ui/ProbabilityBar";
import { RiskBadge } from "../components/ui/RiskBadge";
import { Disclaimer, EmptyState, ErrorState, SkeletonBlock } from "../components/ui/States";
import { useRiskSettings } from "../context/RiskSettings";
import { useCustomers, useDashboard } from "../hooks/useApi";
import { fmtDate, fmtDec, fmtInt, fmtMoney, fmtPct } from "../lib/format";
import { RISK_ORDER, RISK_STYLE } from "../lib/risk";

function RiskDistribution({ data, total }: { data: { nivel: "Alto" | "Médio" | "Baixo"; clientes: number; percentual: number; receita_mensal: number }[]; total: number }) {
  const ordered = RISK_ORDER.map((l) => data.find((d) => d.nivel === l)!).filter(Boolean);
  return (
    <div>
      <div className="flex h-3 overflow-hidden rounded-full" role="img"
        aria-label={ordered.map((d) => `${d.nivel}: ${fmtPct(d.percentual)}`).join(", ")}>
        {[...ordered].reverse().map((d) => (
          <div key={d.nivel} style={{ width: `${d.percentual * 100}%`, background: RISK_STYLE[d.nivel].color }} className="border-r-2 border-white last:border-r-0" />
        ))}
      </div>
      <ul className="mt-5 divide-y divide-line">
        {ordered.map((d) => (
          <li key={d.nivel} className="flex items-center justify-between gap-3 py-3">
            <Link to={`/clientes?nivel_risco=${d.nivel}`} className="rounded hover:opacity-80"><RiskBadge level={d.nivel} /></Link>
            <div className="text-right">
              <p className="tabular text-sm font-semibold">{fmtInt(d.clientes)} <span className="font-normal text-muted">· {fmtPct(d.percentual)}</span></p>
              <p className="tabular text-xs text-muted">{fmtMoney(d.receita_mensal, false)}/mês</p>
            </div>
          </li>
        ))}
      </ul>
      <p className="text-xs text-muted">Total: {fmtInt(total)} clientes ativos</p>
    </div>
  );
}

function AttentionTable() {
  const nav = useNavigate();
  const { data, isLoading } = useCustomers({ nivel_risco: ["Alto"], ordenar: "receita_em_risco", tamanho_pagina: 8 });
  if (isLoading) return <SkeletonBlock className="h-64" />;
  if (!data?.itens.length) return <EmptyState title="Nenhum cliente em alto risco" description="Com o limite atual, nenhum cliente foi classificado como alto risco." />;
  return (
    <div className="-mx-5 overflow-x-auto">
      <table className="w-full min-w-[640px] text-sm">
        <caption className="sr-only">Clientes em alto risco com maior receita em risco</caption>
        <thead>
          <tr className="border-y border-line bg-surface-2 text-left text-xs text-ink-2">
            <th scope="col" className="px-5 py-2 font-medium">Cliente</th>
            <th scope="col" className="px-3 py-2 font-medium">Plano</th>
            <th scope="col" className="px-3 py-2 font-medium">Probabilidade</th>
            <th scope="col" className="px-3 py-2 font-medium">Principal fator</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {data.itens.map((c) => (
            <tr key={c.customer_id} className="cursor-pointer hover:bg-surface-2" onClick={() => nav(`/clientes/${c.customer_id}`)}>
              <td className="px-5 py-2.5">
                <Link to={`/clientes/${c.customer_id}`} className="font-medium text-ink hover:text-brand-600" onClick={(e) => e.stopPropagation()}>{c.nome}</Link>
                <p className="text-xs text-muted">{c.customer_id} · {fmtMoney(c.valor_mensal)}/mês</p>
              </td>
              <td className="px-3 py-2.5 text-ink-2">{c.plano}</td>
              <td className="px-3 py-2.5"><ProbabilityBar value={c.probabilidade} level={c.nivel_risco} /></td>
              <td className="max-w-[260px] px-3 py-2.5 text-ink-2"><span className="line-clamp-2">{c.principal_fator}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Dashboard() {
  const { data, isLoading, isError, error, refetch, isFetching } = useDashboard();
  const { threshold } = useRiskSettings();

  if (isError) return <ErrorState error={error} onRetry={() => refetch()} />;
  const k = data?.kpis;

  return (
    <div className={isFetching && !isLoading ? "opacity-70 transition-opacity" : ""}>
      <PageHeader
        eyebrow="Painel de retenção"
        title="Visão geral"
        description={data ? <>Clientes ativos em <b>{fmtDate(data.data_referencia)}</b> · probabilidade estimada de cancelamento nos próximos 60 dias · modelo v{data.model_version}</> : "Carregando…"}
        actions={<Link to="/clientes?nivel_risco=Alto" className="btn-primary">Ver clientes em risco <ArrowRight size={16} aria-hidden /></Link>}
      />

      {threshold != null && data && (
        <div className="mb-5 rounded-lg border border-brand-200 bg-brand-50 px-4 py-2.5 text-sm text-brand-900" role="status">
          Classificação usando limite de alto risco <b>ajustado para {fmtDec(threshold)}</b> (recomendado pelo modelo: {fmtDec(data.faixas_risco_modelo.alto)}).
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard label="Clientes monitorados" icon={Users} loading={isLoading} value={fmtInt(k?.clientes_monitorados)} hint="ativos com previsão atual" />
        <KpiCard label="Precisam de atenção" icon={AlertTriangle} loading={isLoading} accent="var(--color-risk-high)"
          value={fmtInt(k?.clientes_atencao)} hint={k && <>{fmtInt(k.alto_risco)} em alto · {fmtInt(k.medio_risco)} em médio risco</>}
          info="Clientes classificados como alto ou médio risco com os limites atuais." />
        <KpiCard label="Taxa estimada de churn" icon={Percent} loading={isLoading} accent="var(--color-churn)"
          value={fmtPct(k?.taxa_churn_estimada)} hint="média das probabilidades · 60 dias"
          info="Média das probabilidades estimadas de cancelamento em 60 dias dos clientes ativos. É uma estimativa, não uma certeza." />
        <KpiCard label="Receita mensal em risco" icon={Banknote} loading={isLoading} accent="var(--color-risk-mid)"
          value={fmtMoney(k?.receita_mensal_em_risco, false)} hint={k && <>de {fmtMoney(k.receita_mensal_total, false)} · perda esperada {fmtMoney(k.receita_esperada_perdida, false)}</>}
          info="Soma das mensalidades dos clientes em alto risco. Perda esperada = Σ probabilidade × mensalidade de todos os clientes." />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_1.4fr]">
        <Card title="Distribuição de risco" subtitle="Clientes e receita mensal por nível" info="Baixo, médio e alto risco são definidos pelos limites de probabilidade. O limite de alto risco é configurável no topo da página.">
          {data ? <RiskDistribution data={data.distribuicao_risco} total={data.kpis.clientes_monitorados} /> : <SkeletonBlock className="h-56" />}
        </Card>
        <Card title="Distribuição das probabilidades" subtitle="A maioria dos clientes tem risco baixo; a cauda à direita concentra a atenção">
          {data ? <ProbabilityHistogram data={data.histograma_probabilidade} /> : <SkeletonBlock className="h-56" />}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Evolução do churn" subtitle="Taxa mensal de cancelamento real · últimos 24 meses" info="Cancelamentos no mês ÷ clientes ativos no início do mês. Os picos em janeiro coincidem com os reajustes anuais.">
          {data ? <ChurnTrendChart data={data.evolucao_churn} /> : <SkeletonBlock className="h-60" />}
        </Card>
        <Card title="Previsto × real" subtitle="Probabilidade média prevista vs. churn observado em 60 dias" info="Compara o que o modelo previu com o que realmente aconteceu. Em jan–mar/2026, o reajuste de 9% gerou mais cancelamentos que o previsto — um exemplo de drift.">
          {data ? <PredictedVsRealChart data={data.previsto_vs_real} /> : <SkeletonBlock className="h-60" />}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Sinais mais frequentes" subtitle="Fatores acionáveis entre os clientes que precisam de atenção"
          info="Contagem dos 3 principais fatores acionáveis (que a empresa pode influenciar) de cada cliente em alto ou médio risco.">
          {data ? (
            data.principais_fatores.length ? (
              <BarList ariaLabel="Fatores mais frequentes" items={data.principais_fatores.map((f) => ({
                key: f.fator, label: f.fator, value: f.percentual, display: fmtPct(f.percentual, 0),
                sub: `${f.categoria} · ${fmtInt(f.clientes)} clientes`,
              }))} />
            ) : <EmptyState title="Sem clientes em atenção" />
          ) : <SkeletonBlock className="h-64" />}
        </Card>
        <Card title="O que mais influencia o risco" subtitle="Importância global no modelo (permutação, período de teste)"
          info="Quanto a qualidade do modelo cai quando a informação é embaralhada. Mostra associação, não causalidade.">
          {data ? (
            <BarList ariaLabel="Importância global" items={data.importancia_global.slice(0, 8).map((f) => ({
              key: f.feature, label: f.label, value: f.permutation_importance, display: fmtDec(f.permutation_importance, 3),
              sub: f.direction == null ? "efeito depende da categoria" : f.direction > 0.2 ? "valores altos aumentam o risco" : f.direction < -0.2 ? "valores altos reduzem o risco" : "efeito não linear",
              color: "var(--color-brand-700)",
            }))} />
          ) : <SkeletonBlock className="h-64" />}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Risco por plano" subtitle="Probabilidade média estimada">
          {data ? <GroupRiskChart data={data.por_plano} xLabel="Plano" /> : <SkeletonBlock className="h-56" />}
        </Card>
        <Card title="Risco por faixa de valor" subtitle="Mensalidade atual">
          {data ? <GroupRiskChart data={data.por_faixa_valor} xLabel="Faixa" /> : <SkeletonBlock className="h-56" />}
        </Card>
        <Card title="Risco por comportamento" subtitle="Acessos por semana no último mês (≤2 = até 2 acessos/semana)">
          {data ? <GroupRiskChart data={data.por_comportamento} xLabel="Uso" /> : <SkeletonBlock className="h-56" />}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1.5fr_1fr]">
        <Card title="Prioridade de contato" subtitle="Alto risco ordenado por receita em risco (probabilidade × mensalidade)"
          actions={<Link to="/clientes?nivel_risco=Alto&ordenar=receita_em_risco" className="btn-ghost text-xs">Ver todos <ArrowRight size={14} aria-hidden /></Link>}>
          <AttentionTable />
        </Card>
        <Card title="Desempenho do modelo" subtitle={data ? `${data.metricas_modelo.nome} · threshold ${fmtDec(data.metricas_modelo.threshold)}` : undefined}
          actions={<Link to="/modelo" className="btn-ghost text-xs">Detalhes <ArrowRight size={14} aria-hidden /></Link>}>
          {data ? (
            <>
              <dl className="mb-4 grid grid-cols-3 gap-2 text-center">
                {[["ROC-AUC", fmtDec(data.metricas_modelo.roc_auc, 3)], ["PR-AUC", fmtDec(data.metricas_modelo.pr_auc, 3)], ["Recall", fmtPct(data.metricas_modelo.recall, 0)]].map(([l, v]) => (
                  <div key={l} className="rounded-lg bg-surface-2 py-2"><dt className="text-[11px] text-muted">{l}</dt><dd className="tabular font-semibold">{v}</dd></div>
                ))}
              </dl>
              <ConfusionMatrix m={data.metricas_modelo.matriz_confusao} />
            </>
          ) : <SkeletonBlock className="h-64" />}
        </Card>
      </div>

      <div className="mt-6">
        <Disclaimer>
          Os valores exibidos são <b>probabilidades estimadas</b> a partir de padrões históricos, não certezas. Sugestões de ação são apoio à decisão — a decisão final é da equipe responsável. Dados 100% sintéticos.
        </Disclaimer>
      </div>
    </div>
  );
}
