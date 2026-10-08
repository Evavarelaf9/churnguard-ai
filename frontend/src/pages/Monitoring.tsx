import clsx from "clsx";
import { AlertTriangle, CheckCircle2, Info, OctagonAlert, Play } from "lucide-react";

import { PredictedVsRealChart } from "../components/charts/Charts";
import { DriftTimelineChart, ErrorsOverTimeChart, PerformanceOverTimeChart, PredictionVolumeChart } from "../components/charts/MonitoringCharts";
import { Card } from "../components/ui/Card";
import { KpiCard } from "../components/ui/KpiCard";
import { PageHeader } from "../components/ui/PageHeader";
import { Disclaimer, ErrorState, SkeletonBlock } from "../components/ui/States";
import { useMonitoring, useMonitoringRuns, useRunMonitoring } from "../hooks/useApi";
import { fmtDate, fmtDec, fmtInt, fmtPct } from "../lib/format";
import type { DriftStatus, MonitoringAlert, MonitoringStatus } from "../lib/types";

const STATUS: Record<MonitoringStatus, { label: string; bg: string; ink: string; icon: typeof CheckCircle2; text: string }> = {
  ok: { label: "Saudável", bg: "var(--color-risk-low-bg)", ink: "var(--color-risk-low-ink)", icon: CheckCircle2, text: "Desempenho e dados dentro do esperado." },
  "atenção": { label: "Atenção", bg: "var(--color-risk-mid-bg)", ink: "var(--color-risk-mid-ink)", icon: AlertTriangle, text: "Há sinais que merecem acompanhamento. As previsões continuam utilizáveis." },
  "crítico": { label: "Crítico", bg: "var(--color-risk-high-bg)", ink: "var(--color-risk-high-ink)", icon: OctagonAlert, text: "Desempenho ou dados mudaram de forma relevante. Avaliar re-treino antes de usar as previsões." },
};
const ALERT_ICON = { "crítico": OctagonAlert, "atenção": AlertTriangle, info: Info } as const;
const ALERT_STYLE = {
  "crítico": "border-risk-high/40 bg-risk-high-bg/50",
  "atenção": "border-risk-mid/50 bg-risk-mid-bg/60",
  info: "border-brand-200 bg-brand-50/60",
} as const;
const DRIFT_STYLE: Record<DriftStatus, { bg: string; ink: string; color: string }> = {
  "estável": { bg: "var(--color-risk-low-bg)", ink: "var(--color-risk-low-ink)", color: "var(--color-risk-low)" },
  moderado: { bg: "var(--color-risk-mid-bg)", ink: "var(--color-risk-mid-ink)", color: "var(--color-risk-mid)" },
  significativo: { bg: "var(--color-risk-high-bg)", ink: "var(--color-risk-high-ink)", color: "var(--color-risk-high)" },
};

function AlertCard({ a }: { a: MonitoringAlert }) {
  const Icon = ALERT_ICON[a.nivel];
  return (
    <li className={clsx("rounded-xl border p-4", ALERT_STYLE[a.nivel])}>
      <div className="flex items-start gap-3">
        <Icon size={18} className="mt-0.5 shrink-0 text-ink-2" aria-hidden />
        <div>
          <p className="text-sm font-semibold text-ink">
            <span className="mr-2 rounded bg-white/70 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-ink-2">{a.nivel} · {a.tipo}</span>
            {a.titulo}
          </p>
          <p className="mt-1 text-sm text-ink-2">{a.detalhe}</p>
          <p className="mt-2 text-sm text-ink"><b>Recomendação:</b> {a.recomendacao}</p>
        </div>
      </div>
    </li>
  );
}

export default function Monitoring() {
  const { data, isLoading, isError, error, refetch } = useMonitoring();
  const runs = useMonitoringRuns();
  const run = useRunMonitoring();

  if (isError) return <ErrorState error={error} onRetry={() => refetch()} />;
  if (isLoading || !data) {
    return <div className="space-y-4" aria-busy="true"><SkeletonBlock className="h-24" /><SkeletonBlock className="h-40" /><SkeletonBlock className="h-72" /></div>;
  }
  const s = STATUS[data.status];
  const recent = data.desempenho_mensal.slice(-3);
  const avg = (k: "recall" | "pr_auc" | "precision") => recent.reduce((acc, r) => acc + (r[k] ?? 0), 0) / Math.max(recent.length, 1);
  const lastPerf = data.desempenho_mensal[data.desempenho_mensal.length - 1];
  const important = data.drift_atual.atributos.filter((d) => d.importante).slice(0, 4).map((d) => d.feature);
  const timelineFeatures = Array.from(new Set([...important, "tempo_medio_resolucao_h_6m", "valor_mensal"]));
  const labeled = data.desempenho_mensal.reduce((a, r) => a + r.previsoes, 0);

  return (
    <div>
      <PageHeader eyebrow="MLOps" title="Monitoramento do modelo"
        description={<>Um modelo não é treinado uma vez e esquecido. Esta tela acompanha se ele continua acertando e se os dados mudaram desde a validação · modelo v{data.model_version}.</>}
        actions={<button className="btn-primary" onClick={() => run.mutate()} disabled={run.isPending}><Play size={15} aria-hidden /> {run.isPending ? "Executando…" : "Executar monitoramento"}</button>} />

      <section className="card mb-4 flex flex-col gap-4 p-5 sm:flex-row sm:items-center" style={{ background: s.bg }} aria-live="polite">
        <s.icon size={36} style={{ color: s.ink }} aria-hidden />
        <div className="flex-1">
          <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: s.ink }}>Status geral</p>
          <p className="text-2xl font-semibold" style={{ color: s.ink }}>{s.label}</p>
          <p className="text-sm text-ink-2">{s.text}</p>
        </div>
        <p className="text-xs text-ink-2">
          {data.alertas.length} alerta(s) · dados até {fmtDate(data.data_referencia)}
          {run.data?.status && run.isSuccess && <><br />Execução registrada agora.</>}
        </p>
      </section>

      {data.alertas.length > 0 && <ul className="mb-4 space-y-3" aria-label="Alertas">{data.alertas.map((a) => <AlertCard key={a.titulo} a={a} />)}</ul>}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard label="Previsões com resultado real" value={fmtInt(labeled)} hint={`${fmtInt(data.previsoes_aguardando_resultado)} aguardando os 60 dias`} />
        <KpiCard label="Recall (últimos 3 meses)" value={fmtPct(avg("recall"))} hint={`validado: ${fmtPct(data.linha_de_base.recall)}`} accent="var(--color-churn)" />
        <KpiCard label="PR-AUC (últimos 3 meses)" value={fmtDec(avg("pr_auc"), 3)} hint={`validado: ${fmtDec(data.linha_de_base.pr_auc, 3)}`} />
        <KpiCard label="Calibração (último mês)" value={lastPerf ? `${lastPerf.gap_calibracao > 0 ? "+" : ""}${fmtDec(lastPerf.gap_calibracao * 100, 1)} p.p.` : "—"}
          hint={lastPerf && <>previsto {fmtPct(lastPerf.churn_previsto)} × real {fmtPct(lastPerf.churn_real)}</>} accent="var(--color-risk-mid)"
          info="Diferença entre a probabilidade média prevista e o churn que realmente aconteceu. Perto de zero = probabilidades realistas." />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Previsto × real" subtitle="Churn previsto (média das probabilidades) vs. observado em 60 dias">
          <PredictedVsRealChart data={data.desempenho_mensal} />
        </Card>
        <Card title="Precision e recall ao longo do tempo" subtitle="Linhas tracejadas (mesma cor): precision e recall medidos na validação">
          <PerformanceOverTimeChart data={data.desempenho_mensal} baseline={data.linha_de_base} />
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Acertos e erros por mês" subtitle="Falso positivo = alerta sem necessidade · falso negativo = cancelamento não detectado">
          <ErrorsOverTimeChart data={data.desempenho_mensal} />
        </Card>
        <Card title="Drift ao longo do tempo (PSI)" subtitle="Atributos mais importantes e sinais operacionais"
          info="PSI compara a distribuição de cada atributo com a dos dados de validação. Abaixo de 0,10 = estável; 0,10–0,25 = moderado; acima de 0,25 = significativo.">
          <DriftTimelineChart data={data.drift_temporal} features={timelineFeatures} labels={data.rotulos} />
        </Card>
      </div>

      <Card className="mt-4" title="Volume de previsões" subtitle="Quantidade de previsões registradas por data de referência e origem"
        info="Quedas ou picos inesperados no volume indicam falhas no job de pontuação ou mudanças na base de clientes.">
        {data.volume.length ? <PredictionVolumeChart data={data.volume} /> : <p className="py-8 text-center text-sm text-muted">Sem previsões registradas.</p>}
      </Card>

      <Card className="mt-4" title="Drift dos dados atuais" subtitle={`${data.referencia_drift} × clientes em ${fmtDate(data.drift_atual.data_referencia)}`} bodyClassName="!px-0">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-sm">
            <caption className="sr-only">Estabilidade de cada atributo do modelo</caption>
            <thead>
              <tr className="border-y border-line bg-surface-2 text-left text-xs text-ink-2">
                <th scope="col" className="px-5 py-2 font-medium">Atributo</th>
                <th scope="col" className="px-3 py-2 font-medium">Importância</th>
                <th scope="col" className="px-3 py-2 font-medium">PSI</th>
                <th scope="col" className="px-3 py-2 font-medium">Status</th>
                <th scope="col" className="px-3 py-2 text-right font-medium">Referência</th>
                <th scope="col" className="px-5 py-2 text-right font-medium">Atual</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {data.drift_atual.atributos.slice(0, 15).map((d) => {
                const st = DRIFT_STYLE[d.status];
                const fmtVal = (v: number | null) => d.tipo === "categórico" ? fmtPct(v, 0) : fmtDec(v, 2);
                return (
                  <tr key={d.feature}>
                    <td className="px-5 py-2.5">
                      <p className="font-medium text-ink">{d.label}</p>
                      <p className="text-xs text-muted">{d.tipo}{d.categoria_referencia ? ` · % de "${d.categoria_referencia}"` : " · média"}</p>
                    </td>
                    <td className="px-3 py-2.5 text-ink-2">{d.ranking_importancia ? <>#{d.ranking_importancia}{d.importante && <span className="ml-1.5 rounded bg-brand-50 px-1.5 text-[10px] font-semibold text-brand-700">top 10</span>}</> : "—"}</td>
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="tabular w-12 font-semibold">{fmtDec(d.psi, 3)}</span>
                        <span className="h-1.5 w-24 overflow-hidden rounded-full bg-black/[0.06]" aria-hidden>
                          <span className="block h-full rounded-full" style={{ width: `${Math.min(d.psi / 0.5, 1) * 100}%`, background: st.color }} />
                        </span>
                      </div>
                    </td>
                    <td className="px-3 py-2.5"><span className="rounded-full px-2 py-0.5 text-xs font-medium" style={{ background: st.bg, color: st.ink }}>{d.status}</span></td>
                    <td className="tabular px-3 py-2.5 text-right text-ink-2">{fmtVal(d.media_referencia)}</td>
                    <td className="tabular px-5 py-2.5 text-right text-ink">{fmtVal(d.media_atual)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="mt-4 grid gap-4 lg:grid-cols-[1.2fr_1fr]">
        <Card title="Política de monitoramento e re-treino" subtitle="Regras que definem os alertas">
          <ul className="space-y-2 text-sm text-ink-2">
            <li>• <b className="text-ink">Desempenho:</b> alerta crítico se recall ou PR-AUC médios dos últimos {data.politica.janela_desempenho_meses} meses caírem mais de {fmtPct(data.politica.queda_relativa_recall_critica, 0)} em relação à validação.</li>
            <li>• <b className="text-ink">Calibração:</b> atenção se o churn real se afastar mais de {fmtDec(data.politica.gap_calibracao_atencao * 100, 0)} p.p. do previsto.</li>
            <li>• <b className="text-ink">Drift:</b> PSI ≥ {fmtDec(data.politica.psi_significativo)} em atributo do top {data.politica.top_atributos_importantes} de importância = crítico; em atributos menos influentes = atenção.</li>
            <li>• <b className="text-ink">Re-treino:</b> em alerta crítico, ou a cada 6 meses, com nova validação temporal antes de trocar a versão ativa.</li>
            <li>• <b className="text-ink">Resultado real:</b> registrado automaticamente 60 dias após cada previsão (job <code className="text-xs">update_outcomes</code>).</li>
          </ul>
        </Card>
        <Card title="Execuções registradas" subtitle="Histórico do job de monitoramento" bodyClassName="!px-0">
          {runs.data?.length ? (
            <ul className="max-h-60 divide-y divide-line overflow-y-auto">
              {runs.data.map((r) => (
                <li key={r.id} className="flex items-center justify-between px-5 py-2.5 text-sm">
                  <span className="text-ink-2">{new Date(r.executado_em).toLocaleString("pt-BR")}</span>
                  <span className="text-xs text-muted">{r.alertas} alerta(s)</span>
                  <span className="rounded-full px-2 py-0.5 text-xs font-medium" style={{ background: STATUS[r.status].bg, color: STATUS[r.status].ink }}>{STATUS[r.status].label}</span>
                </li>
              ))}
            </ul>
          ) : <p className="px-5 text-sm text-muted">Nenhuma execução registrada. Use “Executar monitoramento”.</p>}
        </Card>
      </div>

      <div className="mt-6">
        <Disclaimer>O desempenho mensal usa as previsões do modelo validado para cada data de referência, comparadas com o que aconteceu nos 60 dias seguintes. Drift não significa, por si só, que o modelo piorou — por isso é cruzado com o desempenho e com a importância de cada atributo.</Disclaimer>
      </div>
    </div>
  );
}
