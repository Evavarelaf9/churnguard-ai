import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import { fmtDate, fmtDec, fmtInt, fmtMonth, fmtPct } from "../../lib/format";
import type { MonitoringSummary, MonthlyPerformance } from "../../lib/types";
import { TooltipBox } from "./ChartTooltip";
import { axisProps, C } from "./theme";

const pctTick = (v: number) => `${(v * 100).toLocaleString("pt-BR", { maximumFractionDigits: 1 })}%`;

export function PerformanceOverTimeChart({ data, baseline }: { data: MonthlyPerformance[]; baseline: { precision: number; recall: number } }) {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="data_referencia" tickFormatter={fmtMonth} {...axisProps} />
        <YAxis tickFormatter={pctTick} {...axisProps} width={44} domain={[0, 0.6]} />
        <ReferenceLine y={baseline.recall} stroke={C.orange} strokeDasharray="4 3" />
        <ReferenceLine y={baseline.precision} stroke={C.blue} strokeDasharray="4 3" />
        <Tooltip cursor={{ stroke: C.axis }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={`Referência ${fmtDate(payload[0].payload.data_referencia)}`} rows={[
            { label: "Precision", value: fmtPct(payload[0].payload.precision), color: C.blue },
            { label: "Recall", value: fmtPct(payload[0].payload.recall), color: C.orange },
            { label: "PR-AUC", value: fmtDec(payload[0].payload.pr_auc, 3) },
            { label: "ROC-AUC", value: fmtDec(payload[0].payload.roc_auc, 3) },
          ]} />) : null} />
        <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
        <Line name="Precision" dataKey="precision" stroke={C.blue} strokeWidth={2} dot={{ r: 3 }} />
        <Line name="Recall" dataKey="recall" stroke={C.orange} strokeWidth={2} dot={{ r: 3 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ErrorsOverTimeChart({ data }: { data: MonthlyPerformance[] }) {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data} margin={{ top: 8, right: 4, left: 0, bottom: 0 }} barGap={2}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="data_referencia" tickFormatter={fmtMonth} {...axisProps} />
        <YAxis {...axisProps} width={44} />
        <Tooltip cursor={{ fill: "rgba(0,0,0,0.04)" }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={`Referência ${fmtDate(payload[0].payload.data_referencia)}`} rows={[
            { label: "Verdadeiros positivos", value: fmtInt(payload[0].payload.vp), color: C.blue },
            { label: "Falsos positivos", value: fmtInt(payload[0].payload.fp), color: C.orange },
            { label: "Falsos negativos", value: fmtInt(payload[0].payload.fn), color: C.violet },
            { label: "Previsões", value: fmtInt(payload[0].payload.previsoes) },
          ]} />) : null} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar name="Verdadeiros positivos" dataKey="vp" fill={C.blue} radius={[4, 4, 0, 0]} maxBarSize={18} />
        <Bar name="Falsos positivos" dataKey="fp" fill={C.orange} radius={[4, 4, 0, 0]} maxBarSize={18} />
        <Bar name="Falsos negativos" dataKey="fn" fill={C.violet} radius={[4, 4, 0, 0]} maxBarSize={18} />
      </BarChart>
    </ResponsiveContainer>
  );
}

const LINE_COLORS = [C.blue, C.orange, C.aqua, "#eda100", "#e87ba4", C.violet];

export function DriftTimelineChart({ data, features, labels }: { data: MonitoringSummary["drift_temporal"]; features: string[]; labels: Record<string, string> }) {
  const rows = data.map((d) => ({ data_referencia: d.data_referencia, ...Object.fromEntries(features.map((f) => [f, d.psi[f] ?? null])) }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={rows} margin={{ top: 8, right: 64, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="data_referencia" tickFormatter={fmtMonth} {...axisProps} />
        <YAxis {...axisProps} width={44} tickFormatter={(v) => fmtDec(v)} />
        <ReferenceLine y={0.1} stroke={C.mid} strokeDasharray="4 3" label={{ value: "moderado", position: "right", fill: C.muted, fontSize: 10 }} />
        <ReferenceLine y={0.25} stroke={C.high} strokeDasharray="4 3" label={{ value: "significativo", position: "right", fill: C.muted, fontSize: 10 }} />
        <Tooltip cursor={{ stroke: C.axis }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={`Referência ${fmtDate(payload[0].payload.data_referencia)}`}
            rows={features.map((f, i) => ({ label: labels[f] ?? f, value: fmtDec(payload[0].payload[f], 3), color: LINE_COLORS[i % LINE_COLORS.length] }))} />
        ) : null} />
        <Legend iconType="plainline" wrapperStyle={{ fontSize: 11 }} />
        {features.map((f, i) => (
          <Line key={f} name={labels[f] ?? f} dataKey={f} stroke={LINE_COLORS[i % LINE_COLORS.length]} strokeWidth={2} dot={{ r: 2.5 }} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

const ORIGIN_LABEL: Record<string, string> = { historico: "Histórico (validação)", batch: "Pontuação em lote", api: "Consultas on-line" };
const ORIGIN_COLOR: Record<string, string> = { historico: C.blue, batch: C.orange, api: C.aqua };

/** Volume de previsões por data de referência, empilhado por origem. */
export function PredictionVolumeChart({ data }: { data: MonitoringSummary["volume"] }) {
  const origins = Array.from(new Set(data.map((d) => d.origem)));
  const rows = Object.values(data.reduce<Record<string, Record<string, number | string>>>((acc, d) => {
    const r = (acc[d.data_referencia] ??= { data_referencia: d.data_referencia });
    r[d.origem] = ((r[d.origem] as number) ?? 0) + d.previsoes;
    return acc;
  }, {})).sort((a, b) => String(a.data_referencia).localeCompare(String(b.data_referencia)));
  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={rows} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="data_referencia" tickFormatter={fmtMonth} {...axisProps} />
        <YAxis {...axisProps} width={48} tickFormatter={(v: number) => fmtInt(v)} />
        <Tooltip cursor={{ fill: "rgba(0,0,0,0.04)" }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={`Referência ${fmtDate(payload[0].payload.data_referencia)}`}
            rows={origins.filter((o) => payload[0].payload[o] != null).map((o) => ({ label: ORIGIN_LABEL[o] ?? o, value: fmtInt(payload[0].payload[o]), color: ORIGIN_COLOR[o] }))} />) : null} />
        <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v: string) => ORIGIN_LABEL[v] ?? v} />
        {origins.map((o) => <Bar key={o} dataKey={o} name={o} stackId="v" fill={ORIGIN_COLOR[o] ?? C.violet} maxBarSize={22} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}
