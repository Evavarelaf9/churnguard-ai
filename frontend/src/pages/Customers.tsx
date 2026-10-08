import clsx from "clsx";
import { ChevronLeft, ChevronRight, Search, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { Card } from "../components/ui/Card";
import { PageHeader } from "../components/ui/PageHeader";
import { ProbabilityBar } from "../components/ui/ProbabilityBar";
import { RiskBadge } from "../components/ui/RiskBadge";
import { EmptyState, ErrorState, SkeletonBlock } from "../components/ui/States";
import { useCustomers, useFilters } from "../hooks/useApi";
import { fmtDate, fmtDec, fmtInt, fmtMoney, fmtTenure } from "../lib/format";
import { RISK_ORDER } from "../lib/risk";
import type { CustomerFilters, RiskLevel } from "../lib/types";

const SORTS = [
  { v: "probabilidade", l: "Maior probabilidade" },
  { v: "receita_em_risco", l: "Maior receita em risco" },
  { v: "valor_mensal", l: "Maior mensalidade" },
  { v: "nome", l: "Nome (A–Z)" },
] as const;
const PAGE_SIZE = 20;

function useDebounced<T>(value: T, ms = 350) {
  const [v, setV] = useState(value);
  useEffect(() => { const t = setTimeout(() => setV(value), ms); return () => clearTimeout(t); }, [value, ms]);
  return v;
}

export default function Customers() {
  const [sp, setSp] = useSearchParams();
  const nav = useNavigate();
  const { data: opts } = useFilters();
  const [search, setSearch] = useState(sp.get("busca") ?? "");
  const debounced = useDebounced(search);

  const niveis = sp.getAll("nivel_risco") as RiskLevel[];
  const faixa = sp.get("faixa") ?? "";
  const page = Number(sp.get("pagina") ?? 1);

  const set = (key: string, value: string | string[] | null, resetPage = true) => {
    const next = new URLSearchParams(sp);
    next.delete(key);
    if (Array.isArray(value)) value.forEach((v) => next.append(key, v));
    else if (value) next.set(key, value);
    if (resetPage) next.delete("pagina");
    setSp(next, { replace: true });
  };
  useEffect(() => { if ((sp.get("busca") ?? "") !== debounced) set("busca", debounced || null); // eslint-disable-line react-hooks/exhaustive-deps
  }, [debounced]);

  const range = opts?.faixas_valor.find((f) => f.rotulo === faixa);
  const filters: CustomerFilters = useMemo(() => ({
    nivel_risco: niveis.length ? niveis : undefined,
    plano: sp.get("plano") ?? undefined,
    regiao: sp.get("regiao") ?? undefined,
    valor_min: range?.min || undefined,
    valor_max: range?.max ?? undefined,
    data_referencia: sp.get("data_referencia") ?? undefined,
    busca: sp.get("busca") ?? undefined,
    ordenar: (sp.get("ordenar") as CustomerFilters["ordenar"]) ?? "probabilidade",
    pagina: page,
    tamanho_pagina: PAGE_SIZE,
  }), [sp, range]); // eslint-disable-line react-hooks/exhaustive-deps

  const { data, isLoading, isError, error, refetch, isFetching } = useCustomers(filters);
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;
  const active = niveis.length + ["plano", "regiao", "faixa", "data_referencia", "busca"].filter((k) => sp.get(k)).length;

  const historical = !!sp.get("data_referencia");
  const outcome = (v: number | null | undefined) =>
    v == null ? <span className="text-xs text-muted">aguardando</span>
      : v === 1 ? <span className="rounded-full bg-risk-high-bg px-2 py-0.5 text-xs font-medium text-risk-high-ink">Cancelou</span>
      : <span className="rounded-full bg-risk-low-bg px-2 py-0.5 text-xs font-medium text-risk-low-ink">Permaneceu</span>;

  const toggleLevel = (l: RiskLevel) => set("nivel_risco", niveis.includes(l) ? niveis.filter((x) => x !== l) : [...niveis, l]);

  return (
    <div>
      <PageHeader title="Clientes" eyebrow="Carteira monitorada"
        description="Clientes ativos com a probabilidade estimada de cancelamento mais recente. Clique em um cliente para ver a explicação, o histórico e simular ações." />

      <Card className="mb-4" bodyClassName="!pt-5">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Filtrar por nível de risco">
            <span className="mr-1 text-xs font-medium text-ink-2">Nível de risco:</span>
            {RISK_ORDER.map((l) => (
              <button key={l} onClick={() => toggleLevel(l)} aria-pressed={niveis.includes(l)}
                className={clsx("rounded-full border transition-colors", niveis.includes(l) ? "border-ink/30 ring-2 ring-brand-500/20" : "border-transparent opacity-60 hover:opacity-100")}>
                <RiskBadge level={l} />
              </button>
            ))}
            {active > 0 && (
              <button className="btn-ghost ml-auto py-1 text-xs" onClick={() => { setSearch(""); setSp(new URLSearchParams(), { replace: true }); }}>
                <X size={14} aria-hidden /> Limpar filtros ({active})
              </button>
            )}
          </div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <div className="col-span-2 md:col-span-3 xl:col-span-2">
              <label className="label" htmlFor="busca">Buscar</label>
              <div className="relative">
                <Search size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" aria-hidden />
                <input id="busca" className="input pl-9" placeholder="Nome ou ID do cliente" value={search} onChange={(e) => setSearch(e.target.value)} />
              </div>
            </div>
            <div>
              <label className="label" htmlFor="plano">Plano</label>
              <select id="plano" className="input" value={sp.get("plano") ?? ""} onChange={(e) => set("plano", e.target.value || null)}>
                <option value="">Todos</option>
                {opts?.planos.map((p) => <option key={p}>{p}</option>)}
              </select>
            </div>
            <div>
              <label className="label" htmlFor="regiao">Região</label>
              <select id="regiao" className="input" value={sp.get("regiao") ?? ""} onChange={(e) => set("regiao", e.target.value || null)}>
                <option value="">Todas</option>
                {opts?.regioes.map((r) => <option key={r}>{r}</option>)}
              </select>
            </div>
            <div>
              <label className="label" htmlFor="faixa">Faixa de valor</label>
              <select id="faixa" className="input" value={faixa} onChange={(e) => set("faixa", e.target.value || null)}>
                <option value="">Todas</option>
                {opts?.faixas_valor.map((f) => <option key={f.rotulo}>{f.rotulo}</option>)}
              </select>
            </div>
            <div>
              <label className="label" htmlFor="periodo">Período</label>
              <select id="periodo" className="input" value={sp.get("data_referencia") ?? ""} onChange={(e) => set("data_referencia", e.target.value || null)}>
                <option value="">Mais recente</option>
                {opts?.datas_referencia.map((d) => <option key={d} value={d}>{fmtDate(d)}</option>)}
              </select>
            </div>
          </div>
        </div>
      </Card>

      {historical && (
        <div role="note" className="mb-4 rounded-lg border border-brand-100 bg-brand-50 px-4 py-3 text-sm text-brand-900">
          Você está vendo as previsões feitas em <b>{fmtDate(sp.get("data_referencia"))}</b>, incluindo clientes que cancelaram depois.
          A coluna <b>Resultado em 60 dias</b> compara o que o modelo previu com o que realmente aconteceu.
        </div>
      )}

      <Card
        title={data ? `${fmtInt(data.total)} ${data.total === 1 ? "cliente" : "clientes"}` : "Clientes"}
        subtitle={data && <>Alto risco ≥ {fmtDec(data.faixas_risco.alto)} · médio ≥ {fmtDec(data.faixas_risco.medio)}</>}
        actions={
          <label className="flex items-center gap-2 whitespace-nowrap text-xs text-ink-2">
            <span className="hidden sm:inline">Ordenar por</span>
            <select className="input py-1.5 text-xs" value={filters.ordenar} onChange={(e) => set("ordenar", e.target.value, true)} aria-label="Ordenar por">
              {SORTS.map((s) => <option key={s.v} value={s.v}>{s.l}</option>)}
            </select>
          </label>
        }
        bodyClassName="!px-0 !pb-0"
      >
        {isError ? (
          <div className="p-5"><ErrorState error={error} onRetry={() => refetch()} /></div>
        ) : isLoading ? (
          <div className="space-y-2 p-5">{Array.from({ length: 8 }).map((_, i) => <SkeletonBlock key={i} className="h-11" />)}</div>
        ) : !data?.itens.length ? (
          <EmptyState title="Nenhum cliente encontrado" description="Ajuste ou limpe os filtros para ver mais resultados." />
        ) : (
          <div className={clsx(isFetching && "opacity-60 transition-opacity")}>
            {/* Tabela (telas médias e grandes) */}
            <div className="hidden overflow-x-auto md:block">
              <table className="w-full text-sm">
                <caption className="sr-only">Lista de clientes com probabilidade estimada de cancelamento</caption>
                <thead>
                  <tr className="border-y border-line bg-surface-2 text-left text-xs text-ink-2">
                    <th scope="col" className="px-5 py-2.5 font-medium">Cliente</th>
                    <th scope="col" className="px-3 py-2.5 font-medium">Plano</th>
                    <th scope="col" className="px-3 py-2.5 font-medium">Probabilidade</th>
                    <th scope="col" className="px-3 py-2.5 font-medium">Risco</th>
                    <th scope="col" className="px-3 py-2.5 font-medium">Principal fator</th>
                    {historical && <th scope="col" className="whitespace-nowrap px-3 py-2.5 font-medium">Resultado em 60 dias</th>}
                    <th scope="col" className="px-5 py-2.5 font-medium">Ação sugerida</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {data.itens.map((c) => (
                    <tr key={c.customer_id} className="cursor-pointer align-top hover:bg-surface-2" onClick={() => nav(`/clientes/${c.customer_id}`)}>
                      <td className="px-5 py-3">
                        <Link to={`/clientes/${c.customer_id}`} onClick={(e) => e.stopPropagation()} className="font-medium text-ink hover:text-brand-600">{c.nome}</Link>
                        <p className="text-xs text-muted">{c.customer_id} · {c.regiao ?? "região n/i"} · {fmtTenure(c.tempo_cliente_meses)}</p>
                      </td>
                      <td className="px-3 py-3">
                        <p className="text-ink">{c.plano}</p>
                        <p className="tabular text-xs text-muted">{fmtMoney(c.valor_mensal)}/mês</p>
                      </td>
                      <td className="px-3 py-3"><ProbabilityBar value={c.probabilidade} level={c.nivel_risco} /></td>
                      <td className="px-3 py-3"><RiskBadge level={c.nivel_risco} size="sm" /></td>
                      <td className="max-w-[240px] px-3 py-3 text-ink-2">{c.principal_fator ?? "—"}</td>
                      {historical && <td className="px-3 py-3">{outcome(c.churn_real)}</td>}
                      <td className="max-w-[300px] px-5 py-3 text-xs leading-relaxed text-ink-2"><span className="line-clamp-2">{c.acao_sugerida}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Cartões (celular) */}
            <ul className="divide-y divide-line border-t border-line md:hidden">
              {data.itens.map((c) => (
                <li key={c.customer_id}>
                  <Link to={`/clientes/${c.customer_id}`} className="block px-4 py-3 hover:bg-surface-2">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate font-medium">{c.nome}</p>
                        <p className="text-xs text-muted">{c.customer_id} · {c.plano} · {fmtMoney(c.valor_mensal)}</p>
                      </div>
                      <RiskBadge level={c.nivel_risco} size="sm" />
                    </div>
                    <div className="mt-2"><ProbabilityBar value={c.probabilidade} level={c.nivel_risco} /></div>
                    <p className="mt-1.5 text-xs text-ink-2">{c.principal_fator}</p>
                    {historical && <p className="mt-1.5 text-xs text-ink-2">Resultado em 60 dias: {outcome(c.churn_real)}</p>}
                  </Link>
                </li>
              ))}
            </ul>
            <nav className="flex items-center justify-between border-t border-line px-5 py-3 text-sm" aria-label="Paginação">
              <p className="text-xs text-ink-2">Página {page} de {pages}</p>
              <div className="flex gap-2">
                <button className="btn-secondary py-1.5" disabled={page <= 1} onClick={() => set("pagina", String(page - 1), false)}>
                  <ChevronLeft size={16} aria-hidden /> Anterior
                </button>
                <button className="btn-secondary py-1.5" disabled={page >= pages} onClick={() => set("pagina", String(page + 1), false)}>
                  Próxima <ChevronRight size={16} aria-hidden />
                </button>
              </div>
            </nav>
          </div>
        )}
      </Card>
    </div>
  );
}
