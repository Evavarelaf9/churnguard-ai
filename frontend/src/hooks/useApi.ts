import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { useRiskSettings } from "../context/RiskSettings";
import type { CustomerFilters, SimulationChanges } from "../lib/types";
import { api } from "../services/api";

export function useDashboard() {
  const { threshold, medio } = useRiskSettings();
  return useQuery({ queryKey: ["dashboard", threshold, medio], queryFn: () => api.dashboard(threshold, medio), placeholderData: keepPreviousData });
}

export function useCustomers(filters: CustomerFilters) {
  const { threshold, medio } = useRiskSettings();
  return useQuery({
    queryKey: ["customers", filters, threshold, medio],
    queryFn: () => api.customers(filters, threshold, medio),
    placeholderData: keepPreviousData,
  });
}

export function useCustomer(id: string) {
  return useQuery({ queryKey: ["customer", id], queryFn: () => api.customer(id), retry: (n, e) => (e as { status?: number }).status !== 404 && n < 2 });
}

export const useFilters = () => useQuery({ queryKey: ["filters"], queryFn: api.filters, staleTime: Infinity });
export const useThresholds = () => useQuery({ queryKey: ["thresholds"], queryFn: api.thresholds, staleTime: 5 * 60_000 });
export const useModelInfo = () => useQuery({ queryKey: ["model-info"], queryFn: api.modelInfo, staleTime: Infinity });
export const useModelMetrics = () => useQuery({ queryKey: ["model-metrics"], queryFn: api.modelMetrics, staleTime: Infinity });
export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 60_000 });
export const useActions = (status?: string) => useQuery({ queryKey: ["actions", status], queryFn: () => api.actions(status) });

export function useRecalculate(id: string) {
  const qc = useQueryClient();
  const { threshold, medio } = useRiskSettings();
  return useMutation({
    mutationFn: () => api.recalculate(id, threshold, medio),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["customer", id] }),
  });
}

export function useSimulation(id: string) {
  const { threshold, medio } = useRiskSettings();
  return useMutation({ mutationFn: (changes: SimulationChanges) => api.simulate(id, changes, threshold, medio) });
}

export function useCreateAction(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { tipo_acao: string; responsavel?: string; observacoes?: string }) => api.createAction(id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["customer", id] });
      qc.invalidateQueries({ queryKey: ["actions"] });
    },
  });
}

export function useUpdateAction() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: { id: number; status?: string; resultado?: string | null }) =>
      api.updateAction(id, body as never),
    onSuccess: (a) => {
      qc.invalidateQueries({ queryKey: ["actions"] });
      qc.invalidateQueries({ queryKey: ["customer", a.customer_id] });
    },
  });
}

export const useMonitoring = () => useQuery({ queryKey: ["monitoring"], queryFn: api.monitoring, staleTime: 5 * 60_000 });
export const useMonitoringRuns = () => useQuery({ queryKey: ["monitoring-runs"], queryFn: api.monitoringRuns });

export function useRunMonitoring() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.runMonitoring,
    onSuccess: (data) => {
      qc.setQueryData(["monitoring"], data);
      qc.invalidateQueries({ queryKey: ["monitoring-runs"] });
    },
  });
}
