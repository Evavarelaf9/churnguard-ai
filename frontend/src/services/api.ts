import type {
  CustomerDetail, CustomerFilters, CustomerList, DashboardSummary, FilterOptions, Health, ModelInfo,
  ModelMetrics, MonitoringRun, MonitoringSummary, PredictionResult, RetentionAction, SimulationChanges, SimulationResult, ThresholdTradeoff,
} from "../lib/types";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "/api/v1";
const KEY = import.meta.env.VITE_API_KEY as string | undefined;

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

type Params = Record<string, string | number | boolean | string[] | undefined | null>;

function qs(params?: Params) {
  if (!params) return "";
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    if (Array.isArray(v)) v.forEach((x) => sp.append(k, x));
    else sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

async function request<T>(path: string, init?: RequestInit & { params?: Params }): Promise<T> {
  const url = `${path.startsWith("/health") ? "" : BASE}${path}${qs(init?.params)}`;
  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      headers: { "Content-Type": "application/json", ...(KEY ? { "X-API-Key": KEY } : {}), ...init?.headers },
    });
  } catch {
    throw new ApiError(0, "Não foi possível conectar à API. Verifique se o backend está em execução.");
  }
  if (!res.ok) {
    let detail = `Erro ${res.status}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : Array.isArray(body.detail)
        ? body.detail.map((d: { msg: string }) => d.msg).join("; ") : detail;
    } catch { /* corpo não-JSON */ }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  dashboard: (threshold?: number | null, medio?: number | null) =>
    request<DashboardSummary>("/dashboard/summary", { params: { threshold, limite_medio: medio } }),
  filters: () => request<FilterOptions>("/meta/filtros"),
  thresholds: () => request<ThresholdTradeoff>("/meta/thresholds"),
  customers: (f: CustomerFilters, threshold?: number | null, medio?: number | null) =>
    request<CustomerList>("/customers", { params: { ...f, nivel_risco: f.nivel_risco, threshold, limite_medio: medio } as Params }),
  customer: (id: string) => request<CustomerDetail>(`/customers/${encodeURIComponent(id)}`),
  recalculate: (id: string, threshold?: number | null, medio?: number | null) =>
    request<PredictionResult>(`/customers/${encodeURIComponent(id)}/risk`, { params: { threshold, limite_medio: medio } }),
  simulate: (id: string, changes: SimulationChanges, threshold?: number | null, medio?: number | null) =>
    request<SimulationResult>(`/customers/${encodeURIComponent(id)}/simulate`, {
      method: "POST", body: JSON.stringify(changes), params: { threshold, limite_medio: medio },
    }),
  actions: (status?: string) => request<RetentionAction[]>("/actions", { params: { status, limite: 200 } }),
  createAction: (id: string, body: { tipo_acao: string; responsavel?: string; observacoes?: string }) =>
    request<RetentionAction>(`/customers/${encodeURIComponent(id)}/actions`, { method: "POST", body: JSON.stringify(body) }),
  updateAction: (actionId: number, body: Partial<Pick<RetentionAction, "status" | "resultado" | "observacoes" | "responsavel">>) =>
    request<RetentionAction>(`/actions/${actionId}`, { method: "PATCH", body: JSON.stringify(body) }),
  monitoring: () => request<MonitoringSummary>("/monitoring/summary"),
  monitoringRuns: () => request<MonitoringRun[]>("/monitoring/runs"),
  runMonitoring: () => request<MonitoringSummary>("/monitoring/run", { method: "POST" }),
  modelInfo: () => request<ModelInfo>("/model/info"),
  modelMetrics: () => request<ModelMetrics>("/model/metrics"),
};
