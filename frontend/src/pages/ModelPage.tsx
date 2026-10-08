import { BookOpen, Scale, ShieldCheck } from "lucide-react";

import { BarList } from "../components/charts/BarList";
import { ThresholdTradeoffChart } from "../components/charts/Charts";
import { ConfusionMatrix } from "../components/charts/ConfusionMatrix";
import { Card } from "../components/ui/Card";
import { PageHeader } from "../components/ui/PageHeader";
import { RiskBadge } from "../components/ui/RiskBadge";
import { ErrorState, SkeletonBlock } from "../components/ui/States";
import { useRiskSettings } from "../context/RiskSettings";
import { useDashboard, useModelInfo, useModelMetrics, useThresholds } from "../hooks/useApi";
import { fmtDate, fmtDec, fmtInt, fmtMoney, fmtPct } from "../lib/format";

const METRICS = [
  { key: "roc_auc", label: "ROC-AUC", fmt: (v: number) => fmtDec(v, 3), text: "Capacidade de ordenar clientes por risco. 0,5 = acaso; 1 = perfeito." },
  { key: "pr_auc", label: "PR-AUC", fmt: (v: number) => fmtDec(v, 3), text: "Métrica principal com classes desbalanceadas: equilíbrio entre achar churns e evitar alarmes." },
  { key: "precision", label: "Precision", fmt: (v: number) => fmtPct(v), text: "Dos clientes sinalizados como alto risco, quantos cancelaram de fato." },
  { key: "recall", label: "Recall", fmt: (v: number) => fmtPct(v), text: "Dos clientes que cancelaram, quantos o modelo sinalizou a tempo." },
  { key: "f1", label: "F1", fmt: (v: number) => fmtDec(v, 3), text: "Média harmônica entre precision e recall." },
  { key: "brier", label: "Brier score", fmt: (v: number) => fmtDec(v, 3), text: "Qualidade das probabilidades (calibração). Menor é melhor." },
] as const;

export default function ModelPage() {
  const info = useModelInfo();
  const metrics = useModelMetrics();
  const tr = useThresholds();
  const dash = useDashboard();
  const { threshold, setThreshold } = useRiskSettings();

  if (info.isError || metrics.isError) return <ErrorState error={info.error ?? metrics.error} onRetry={() => { info.refetch(); metrics.refetch(); }} />;
  const i = info.data, m = metrics.data, t = tr.data;
  const current = threshold ?? t?.threshold_modelo ?? i?.threshold ?? 0.23;
  const rows = t ? [0.3, 0.5, 0.7, t.threshold_modelo].sort((a, b) => a - b)
    .map((x) => ({ t: x, p: t.pontos.find((pp) => Math.abs(pp.threshold - x) < 0.005) })) : [];

  return (
    <div>
      <PageHeader eyebrow="Transparência" title="Modelo de previsão"
        description={i ? <>{i.model_name} · versão {i.version} · treinado em {fmtDate(i.trained_at)} · alvo: {i.target}</> : "Carregando…"} />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        {METRICS.map((x) => (
          <div key={x.key} className="card p-4">
            <p className="text-xs font-medium text-ink-2">{x.label}</p>
            {m ? <p className="tabular mt-1 text-2xl font-semibold">{x.fmt(m[x.key])}</p> : <div className="skeleton mt-2 h-7 w-16" />}
            <p className="mt-1 text-[11.5px] leading-snug text-muted">{x.text}</p>
          </div>
        ))}
      </div>
      {m && (
        <div className="mt-4 rounded-xl border border-line bg-surface p-4 text-sm leading-relaxed text-ink-2">
          <b className="text-ink">Por que não usar acurácia?</b> No período de teste, só {fmtPct(m.base_rate)} dos clientes cancelaram.
          Um modelo que dissesse sempre “ninguém cancela” teria {fmtPct(1 - m.base_rate)} de acurácia e não encontraria nenhum cliente em risco.
          Nosso modelo tem {fmtPct(m.accuracy)} de acurácia — número que, sozinho, não diz nada sobre sua utilidade. Por isso priorizamos PR-AUC, recall e precision.
          <span className="mt-1 block text-xs text-muted">Métricas medidas em {m.periodo_teste}, com dados nunca vistos no treino, threshold {fmtDec(m.threshold)}.</span>
        </div>
      )}

      <Card className="mt-4 scroll-mt-20" title="Escolha do threshold" subtitle="Quanto menor o limite, mais cancelamentos são encontrados — e mais alertas falsos são gerados"
        info="Calculado com previsões cujo resultado real (60 dias depois) já é conhecido. A linha vermelha é o limite em uso; a tracejada é o recomendado pelo modelo.">
        <div id="threshold" className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
          <div>{t ? <ThresholdTradeoffChart data={t.pontos} current={current} model={t.threshold_modelo} /> : <SkeletonBlock className="h-72" />}</div>
          <div>
            <div className="overflow-x-auto"><table className="w-full text-sm">
              <caption className="sr-only">Comparação de thresholds</caption>
              <thead>
                <tr className="border-b border-line text-left text-xs text-ink-2">
                  <th scope="col" className="py-2 font-medium">Threshold</th>
                  <th scope="col" className="py-2 text-right font-medium">Precision</th>
                  <th scope="col" className="py-2 text-right font-medium">Recall</th>
                  <th scope="col" className="py-2 text-right font-medium" title="Previsões históricas acima do limite (clientes que teriam sido abordados)">Abordados</th>
                  <th scope="col" className="py-2 text-right font-medium" title="Cancelamentos reais detectados / total de cancelamentos no período">Churns encontrados</th>
                  <th scope="col" className="py-2 text-right font-medium">Sinalizados hoje</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {rows.map(({ t: x, p }) => (
                  <tr key={x} className={Math.abs(x - current) < 0.005 ? "bg-brand-50/60" : ""}>
                    <td className="py-2">
                      <button className="tabular font-semibold text-brand-700 hover:underline" onClick={() => setThreshold(Math.abs(x - (t?.threshold_modelo ?? 0)) < 0.005 ? null : x)}
                        aria-label={`Usar threshold ${fmtDec(x)}`}>{fmtDec(x)}</button>
                      {Math.abs(x - (t?.threshold_modelo ?? -1)) < 0.005 && <span className="ml-1.5 text-[10px] font-semibold uppercase text-muted">modelo</span>}
                    </td>
                    <td className="tabular py-2 text-right">{fmtPct(p?.precision, 0)}</td>
                    <td className="tabular py-2 text-right">{fmtPct(p?.recall, 0)}</td>
                    <td className="tabular py-2 text-right">{fmtInt(p?.abordados_historico)}</td>
                    <td className="tabular py-2 text-right whitespace-nowrap">{fmtInt(p?.churns_identificados)} <span className="text-xs text-muted">de {fmtInt(p?.churns_totais)}</span></td>
                    <td className="tabular py-2 text-right">{fmtInt(p?.clientes_sinalizados_hoje)}</td>
                  </tr>
                ))}
              </tbody>
            </table></div>
            <ul className="mt-4 space-y-2 text-xs leading-relaxed text-ink-2">
              <li><b className="text-ink">0,30</b> — mais clientes abordados; encontra mais cancelamentos, com mais falsos positivos.</li>
              <li><b className="text-ink">0,50</b> — intermediário no papel, mas aqui já deixa passar a maioria dos cancelamentos.</li>
              <li><b className="text-ink">0,70</b> — muito conservador: poucos alertas, quase todos corretos, pouca cobertura.</li>
              {i && <li><b className="text-ink">{fmtDec(i.threshold)}</b> — recomendado: maior resultado esperado da campanha com as premissas abaixo.</li>}
            </ul>
            <p className="mt-3 text-xs text-muted">Clique em um valor para aplicá-lo em todo o painel (ajuste pessoal, salvo neste navegador).</p>
          </div>
        </div>
      </Card>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Matriz de confusão" subtitle="Período de teste · threshold operacional">
          {m ? <ConfusionMatrix m={m.matriz_confusao} /> : <SkeletonBlock className="h-56" />}
        </Card>
        <Card title="Níveis de risco" subtitle="Faixas de probabilidade estimada (configuráveis)">
          {i ? (
            <>
              <ul className="divide-y divide-line text-sm">
                <li className="flex items-center justify-between py-3"><RiskBadge level="Baixo" /><span className="tabular">abaixo de {fmtDec(i.risk_thresholds.medio)}</span><span className="text-xs text-muted">monitorar</span></li>
                <li className="flex items-center justify-between py-3"><RiskBadge level="Médio" /><span className="tabular">{fmtDec(i.risk_thresholds.medio)} a {fmtDec(i.risk_thresholds.alto)}</span><span className="text-xs text-muted">engajamento em escala</span></li>
                <li className="flex items-center justify-between py-3"><RiskBadge level="Alto" /><span className="tabular">a partir de {fmtDec(i.risk_thresholds.alto)}</span><span className="text-xs text-muted">contato humano</span></li>
              </ul>
              <div className="mt-4 rounded-lg bg-surface-2 p-3 text-xs leading-relaxed text-ink-2">
                <p className="font-semibold text-ink">Premissas de negócio do threshold</p>
                Custo por cliente abordado: <b>{fmtMoney(i.business_assumptions.custo_acao)}</b> · sucesso das ações: <b>{fmtPct(i.business_assumptions.taxa_sucesso_retencao, 0)}</b> ·
                receita preservada: <b>{i.business_assumptions.meses_receita_preservada} meses</b>. Ajuste estas premissas à realidade da empresa e re-treine.
              </div>
            </>
          ) : <SkeletonBlock className="h-56" />}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Fatores mais influentes" subtitle="Importância por permutação no período de teste">
          {dash.data ? (
            <BarList ariaLabel="Importância global" items={dash.data.importancia_global.slice(0, 12).map((f) => ({
              key: f.feature, label: f.label, value: f.permutation_importance, display: fmtDec(f.permutation_importance, 3), color: "var(--color-brand-700)",
            }))} />
          ) : <SkeletonBlock className="h-72" />}
        </Card>
        <Card title="Dados usados e excluídos" subtitle="Transparência sobre o que o modelo considera">
          {i ? (
            <div className="space-y-4 text-sm">
              <p className="text-ink-2">
                O modelo usa <b className="text-ink">{i.features.numeric.length + i.features.binary.length + i.features.categorical.length} atributos</b> de uso,
                pagamentos, suporte, satisfação e contrato — todos calculados apenas com dados disponíveis na data da previsão.
              </p>
              <div>
                <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-2"><ShieldCheck size={14} aria-hidden /> Não entram no modelo</p>
                <ul className="divide-y divide-line rounded-lg border border-line">
                  {Object.entries(i.features.excluded).map(([k, v]) => (
                    <li key={k} className="flex gap-3 px-3 py-2 text-xs"><code className="w-32 shrink-0 font-mono text-ink">{k}</code><span className="text-ink-2">{v}</span></li>
                  ))}
                </ul>
              </div>
            </div>
          ) : <SkeletonBlock className="h-72" />}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div className="card p-5">
          <p className="flex items-center gap-2 font-semibold"><Scale size={17} className="text-brand-600" aria-hidden /> Uso responsável</p>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-ink-2">
            <li>O resultado é uma <b>probabilidade estimada</b>, nunca uma certeza.</li>
            <li>Nenhuma decisão sobre o cliente é tomada automaticamente: a equipe decide e registra a ação.</li>
            <li>Correlação não é causalidade: sugestões devem ser validadas (ex.: grupo de controle).</li>
            <li>Consultas a dados de clientes são registradas (LGPD).</li>
          </ul>
        </div>
        <div className="card p-5">
          <p className="flex items-center gap-2 font-semibold"><BookOpen size={17} className="text-brand-600" aria-hidden /> Limitações</p>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-ink-2">
            <li>Dados sintéticos: os números demonstram o método, não um mercado real.</li>
            <li>A maioria dos sinalizados não cancelaria — o modelo prioriza, não decide.</li>
            <li>Mudanças de contexto (reajustes, suporte) exigem monitoramento e re-treino.</li>
            <li>Estratégia de desbalanceamento: {i?.imbalance_strategy ?? "—"} (probabilidades calibradas).</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
