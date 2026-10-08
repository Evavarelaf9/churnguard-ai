import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import { fmtDate, fmtDec, fmtInt, fmtMoney, fmtMonth, fmtPct } from "../../lib/format";
import { RISK_STYLE } from "../../lib/risk";
import type {
  DashboardSummary, GroupStat, PaymentPoint, PredictionHistoryItem, RiskThresholds, ThresholdPoint, UsagePoint,
} from "../../lib/types";
import { TooltipBox } from "./ChartTooltip";
import { axisProps, C } from "./theme";

const pctTick = (v: number) => `${(v * 100).toLocaleString("pt-BR", { maximumFractionDigits: 1 })}%`;

/* ------------------------------------------------------------------ */
export function ChurnTrendChart({ data }: { data: DashboardSummary["evolucao_churn"] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
        <defs>
          <linearGradient id="trend" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={C.blue} stopOpacity={0.18} />
            <stop offset="100%" stopColor={C.blue} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="mes" tickFormatter={fmtMonth} {...axisProps} interval="preserveStartEnd" minTickGap={24} />
        <YAxis tickFormatter={pctTick} {...axisProps} width={44} />
        <Tooltip cursor={{ stroke: C.axis }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={fmtMonth(payload[0].payload.mes)} rows={[
            { label: "Taxa de cancelamento", value: fmtPct(payload[0].payload.taxa_churn), color: C.blue },
            { label: "Cancelamentos", value: fmtInt(payload[0].payload.cancelamentos) },
            { label: "Clientes ativos no início", value: fmtInt(payload[0].payload.clientes_ativos) },
          ]} />) : null} />
        <Area type="monotone" dataKey="taxa_churn" stroke={C.blue} strokeWidth={2} fill="url(#trend)" dot={false} activeDot={{ r: 4 }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
export function PredictedVsRealChart({ data }: { data: DashboardSummary["previsto_vs_real"] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="data_referencia" tickFormatter={fmtMonth} {...axisProps} />
        <YAxis tickFormatter={pctTick} {...axisProps} width={44} />
        <Tooltip cursor={{ stroke: C.axis }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={`Referência ${fmtDate(payload[0].payload.data_referencia)}`} rows={[
            { label: "Previsto (média)", value: fmtPct(payload[0].payload.churn_previsto), color: C.blue },
            { label: "Real (60 dias)", value: fmtPct(payload[0].payload.churn_real), color: C.orange },
          ]} />) : null} />
        <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
        <Line name="Previsto" dataKey="churn_previsto" stroke={C.blue} strokeWidth={2} dot={{ r: 3 }} />
        <Line name="Real" dataKey="churn_real" stroke={C.orange} strokeWidth={2} dot={{ r: 3 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
export function ProbabilityHistogram({ data }: { data: DashboardSummary["histograma_probabilidade"] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 4, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="faixa" {...axisProps} interval={0} tick={{ ...axisProps.tick, fontSize: 10 }} />
        <YAxis {...axisProps} width={52} tickFormatter={(v) => fmtInt(v)} />
        <Tooltip cursor={{ fill: "rgba(0,0,0,0.04)" }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={`Probabilidade ${payload[0].payload.faixa}`} rows={[{ label: "Clientes", value: fmtInt(payload[0].payload.clientes), color: C.blue }]} />
        ) : null} />
        <Bar dataKey="clientes" fill={C.blue} radius={[4, 4, 0, 0]} maxBarSize={36} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
export function GroupRiskChart({ data, xLabel }: { data: GroupStat[]; xLabel: string }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 16, right: 4, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="grupo" {...axisProps} interval={0} tick={{ ...axisProps.tick, fontSize: 10.5 }} />
        <YAxis tickFormatter={pctTick} {...axisProps} width={44} />
        <Tooltip cursor={{ fill: "rgba(0,0,0,0.04)" }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={`${xLabel}: ${payload[0].payload.grupo}`} rows={[
            { label: "Probabilidade média", value: fmtPct(payload[0].payload.probabilidade_media), color: C.blue },
            { label: "Clientes", value: fmtInt(payload[0].payload.clientes) },
            { label: "Em alto risco", value: fmtInt(payload[0].payload.alto_risco) },
            { label: "Receita mensal em risco", value: fmtMoney(payload[0].payload.receita_em_risco, false) },
          ]} />) : null} />
        <Bar dataKey="probabilidade_media" fill={C.blue} radius={[4, 4, 0, 0]} maxBarSize={48}
          label={{ position: "top", fontSize: 11, fill: C.ink, formatter: (v: unknown) => fmtPct(Number(v), 1) }} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
export function UsageHistoryChart({ data }: { data: UsagePoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={230}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id="usage" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={C.blue} stopOpacity={0.2} />
            <stop offset="100%" stopColor={C.blue} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="mes_referencia" tickFormatter={fmtMonth} {...axisProps} />
        <YAxis {...axisProps} width={44} />
        <Tooltip cursor={{ stroke: C.axis }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={fmtMonth(payload[0].payload.mes_referencia)} rows={[
            { label: "Acessos", value: fmtInt(payload[0].payload.acessos), color: C.blue },
            { label: "Dias ativos", value: fmtInt(payload[0].payload.dias_ativos) },
            { label: "Minutos de uso", value: fmtInt(payload[0].payload.minutos_uso) },
            { label: "Funcionalidades usadas", value: fmtInt(payload[0].payload.funcionalidades_usadas) },
          ]} />) : null} />
        <Area type="monotone" dataKey="acessos" stroke={C.blue} strokeWidth={2} fill="url(#usage)" dot={{ r: 2.5 }} activeDot={{ r: 4 }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
const PAY_COLOR = { "Pago em dia": C.low, "Pago com atraso": C.mid, "Em aberto": C.high } as const;

export function PaymentsChart({ data }: { data: PaymentPoint[] }) {
  const rows = data.map((p) => ({ ...p, atraso: Math.max(p.dias_atraso ?? 0, p.status === "Em aberto" ? 30 : 0), barra: Math.max(p.dias_atraso ?? 0, 0.6) }));
  return (
    <div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={rows} margin={{ top: 8, right: 4, left: -12, bottom: 0 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          <XAxis dataKey="mes_referencia" tickFormatter={fmtMonth} {...axisProps} />
          <YAxis {...axisProps} width={44} allowDecimals={false} label={{ value: "dias de atraso", angle: -90, position: "insideLeft", offset: 18, fill: C.muted, fontSize: 10 }} />
          <Tooltip cursor={{ fill: "rgba(0,0,0,0.04)" }} content={({ active, payload }) => active && payload?.length ? (
            <TooltipBox title={fmtMonth(payload[0].payload.mes_referencia)} rows={[
              { label: "Status", value: payload[0].payload.status, color: PAY_COLOR[payload[0].payload.status as keyof typeof PAY_COLOR] },
              { label: "Valor", value: fmtMoney(payload[0].payload.valor_cobrado) },
              { label: "Vencimento", value: fmtDate(payload[0].payload.data_vencimento) },
              { label: "Pagamento", value: fmtDate(payload[0].payload.data_pagamento) },
              { label: "Dias de atraso", value: payload[0].payload.dias_atraso ?? "—" },
            ]} />) : null} />
          <Bar dataKey="barra" radius={[4, 4, 0, 0]} maxBarSize={28}>
            {rows.map((r) => <Cell key={r.mes_referencia} fill={PAY_COLOR[r.status]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <ul className="mt-2 flex flex-wrap gap-4 text-xs text-ink-2" aria-label="Legenda">
        {Object.entries(PAY_COLOR).map(([k, c]) => (
          <li key={k} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: c }} />{k}</li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ */
export function ProbabilityHistoryChart({ data, thresholds }: { data: PredictionHistoryItem[]; thresholds: RiskThresholds }) {
  const rows = Object.values(
    data.reduce<Record<string, PredictionHistoryItem>>((acc, p) => { acc[p.data_referencia] = p; return acc; }, {}),
  );
  const max = Math.max(...rows.map((r) => r.probabilidade), thresholds.alto) * 1.15;
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={rows} margin={{ top: 8, right: 56, left: -8, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="data_referencia" tickFormatter={fmtMonth} {...axisProps} />
        <YAxis tickFormatter={pctTick} {...axisProps} width={44} domain={[0, Math.min(1, max)]} />
        <ReferenceLine y={thresholds.alto} stroke={C.high} strokeDasharray="4 3" label={{ value: "alto", position: "right", fill: C.muted, fontSize: 10 }} />
        <ReferenceLine y={thresholds.medio} stroke={C.mid} strokeDasharray="4 3" label={{ value: "médio", position: "right", fill: C.muted, fontSize: 10 }} />
        <Tooltip cursor={{ stroke: C.axis }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={`Referência ${fmtDate(payload[0].payload.data_referencia)}`} rows={[
            { label: "Probabilidade", value: fmtPct(payload[0].payload.probabilidade), color: C.blue },
            { label: "Nível", value: payload[0].payload.nivel_risco },
            { label: "Resultado real", value: payload[0].payload.churn_real == null ? "aguardando" : payload[0].payload.churn_real ? "cancelou" : "permaneceu" },
          ]} />) : null} />
        <Line dataKey="probabilidade" stroke={C.blue} strokeWidth={2}
          dot={(p: { cx?: number; cy?: number; payload?: PredictionHistoryItem }) => (
            <circle key={`${p.cx}-${p.cy}`} cx={p.cx} cy={p.cy} r={3.5} fill={p.payload ? RISK_STYLE[p.payload.nivel_risco].color : C.blue} stroke="#fff" strokeWidth={1.5} />
          )} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
export function ThresholdTradeoffChart({ data, current, model }: { data: ThresholdPoint[]; current: number; model: number }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 12, right: 12, left: -8, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="threshold" type="number" domain={[0.05, 0.7]} tickFormatter={(v) => fmtDec(v)} {...axisProps} ticks={[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]} />
        <YAxis tickFormatter={pctTick} {...axisProps} width={44} domain={[0, 1]} />
        <ReferenceLine x={model} stroke={C.muted} strokeDasharray="4 3" label={{ value: "modelo", position: "insideTopRight", fill: C.muted, fontSize: 10 }} />
        <ReferenceLine x={current} stroke={C.high} strokeWidth={1.5} />
        <Tooltip cursor={{ stroke: C.axis }} content={({ active, payload }) => active && payload?.length ? (
          <TooltipBox title={`Threshold ${fmtDec(payload[0].payload.threshold)}`} rows={[
            { label: "Precision", value: fmtPct(payload[0].payload.precision), color: C.blue },
            { label: "Recall", value: fmtPct(payload[0].payload.recall), color: C.orange },
            { label: "F1", value: fmtDec(payload[0].payload.f1), color: C.violet },
            { label: "Sinalizados hoje", value: fmtInt(payload[0].payload.clientes_sinalizados_hoje) },
          ]} />) : null} />
        <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
        <Line name="Precision" dataKey="precision" stroke={C.blue} strokeWidth={2} dot={false} />
        <Line name="Recall" dataKey="recall" stroke={C.orange} strokeWidth={2} dot={false} />
        <Line name="F1" dataKey="f1" stroke={C.violet} strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
