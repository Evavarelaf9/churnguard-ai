import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight, BellRing, BrainCircuit, Database, Gauge, LineChart, Lock, MessageSquareText, ScanSearch, ShieldCheck,
  SlidersHorizontal, Workflow,
} from "lucide-react";
import { Link } from "react-router-dom";

import { Logo } from "../components/layout/Logo";
import { RiskBadge } from "../components/ui/RiskBadge";
import { useDashboard, useModelInfo } from "../hooks/useApi";
import { fmtCompactMoney, fmtInt, fmtPct } from "../lib/format";
import { RISK_STYLE } from "../lib/risk";
import { api } from "../services/api";

const STEPS = [
  { icon: Database, title: "Dados do relacionamento", text: "Uso do produto, pagamentos, chamados e satisfação, mês a mês." },
  { icon: ScanSearch, title: "Sinais de comportamento", text: "Queda de uso, atrasos, aumento de chamados, renovação próxima." },
  { icon: BrainCircuit, title: "Probabilidade de churn", text: "Modelo de Machine Learning estima o risco nos próximos 60 dias." },
  { icon: MessageSquareText, title: "Explicação clara", text: "Os fatores que mais pesaram, em linguagem de negócio." },
  { icon: BellRing, title: "Ação e acompanhamento", text: "Sugestão de retenção, decisão humana e medição do resultado." },
];

const FEATURES = [
  { icon: Gauge, title: "Preventivo", text: "Identifica o risco antes do cancelamento, quando ainda dá tempo de agir." },
  { icon: MessageSquareText, title: "Explicável", text: "Cada previsão mostra por que o cliente está em risco, com SHAP traduzido para o negócio." },
  { icon: SlidersHorizontal, title: "Configurável", text: "O limite de alto risco é ajustável: mais cobertura ou mais precisão, conforme a capacidade do time." },
  { icon: Workflow, title: "Simulação de retenção", text: "Teste cenários “e se” — mais uso, sem atrasos, outro plano — antes de decidir." },
  { icon: LineChart, title: "Mensurável", text: "Previsto × real, falsos positivos e negativos: o modelo é acompanhado, não esquecido." },
  { icon: Lock, title: "Responsável", text: "Sem atributos sensíveis no modelo, acesso auditado e decisão sempre humana (LGPD)." },
];

function PreviewCard() {
  const { data } = useQuery({
    queryKey: ["landing-preview"],
    queryFn: async () => {
      const list = await api.customers({ nivel_risco: ["Alto"], tamanho_pagina: 1 });
      const id = list.itens[0]?.customer_id;
      return id ? api.customer(id) : null;
    },
    retry: 0,
  });
  const p = data?.previsao_atual;
  if (!data || !p) {
    return <div className="h-[340px] w-full max-w-md animate-pulse rounded-2xl bg-white/5" aria-hidden />;
  }
  return (
    <div className="w-full max-w-md rounded-2xl border border-white/10 bg-white p-5 text-ink shadow-2xl shadow-black/30" aria-label="Exemplo de análise de cliente">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs text-muted">{data.perfil.customer_id} · plano {data.perfil.plano}</p>
          <p className="text-[15px] font-semibold">{data.perfil.nome}</p>
        </div>
        <RiskBadge level={p.nivel_risco} />
      </div>
      <div className="mt-4 flex items-end gap-3">
        <p className="text-4xl font-semibold tracking-tight">{fmtPct(p.probabilidade, 0)}</p>
        <p className="pb-1 text-xs leading-snug text-ink-2">probabilidade estimada de<br />cancelamento em 60 dias</p>
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-black/[0.06]">
        <div className="h-full rounded-full" style={{ width: `${p.probabilidade * 100}%`, background: RISK_STYLE[p.nivel_risco].color }} />
      </div>
      <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-muted">Principais fatores</p>
      <ul className="mt-2 space-y-1.5 text-sm">
        {p.fatores_risco.slice(0, 3).map((f) => (
          <li key={f.feature} className="flex items-start gap-2">
            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-risk-high" aria-hidden />
            {f.text}
          </li>
        ))}
      </ul>
      {p.recomendacao && (
        <div className="mt-4 rounded-lg bg-brand-50 p-3 text-[13px] leading-snug text-brand-900">
          <span className="font-semibold">Sugestão: </span>{p.recomendacao.acao_sugerida}
        </div>
      )}
    </div>
  );
}

export default function Landing() {
  const { data: dash } = useDashboard();
  const { data: info } = useModelInfo();
  const k = dash?.kpis;
  const b = info?.business_assumptions;
  const protectedRevenue = k && b ? k.receita_mensal_em_risco * b.meses_receita_preservada * b.taxa_sucesso_retencao : null;

  const stats = [
    { label: "clientes monitorados", value: fmtInt(k?.clientes_monitorados) },
    { label: "previsões realizadas", value: fmtInt(k?.previsoes_realizadas) },
    { label: "clientes em risco (alto + médio)", value: fmtInt(k?.clientes_atencao) },
    { label: "potencial de receita protegida*", value: protectedRevenue == null ? "—" : fmtCompactMoney(protectedRevenue) },
  ];

  return (
    <div className="min-h-screen bg-canvas">
      {/* Hero */}
      <div className="relative overflow-hidden bg-navy-800 text-white">
        <div className="pointer-events-none absolute inset-0 opacity-[0.07]"
          style={{ backgroundImage: "linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)", backgroundSize: "44px 44px" }} aria-hidden />
        <div className="pointer-events-none absolute -right-40 -top-40 h-[520px] w-[520px] rounded-full bg-brand-500/25 blur-3xl" aria-hidden />
        <header className="relative mx-auto flex max-w-7xl items-center justify-between px-4 py-5 sm:px-6">
          <Logo light />
          <nav className="flex items-center gap-1 text-sm sm:gap-6" aria-label="Seções">
            <a href="#como-funciona" className="hidden text-white/70 hover:text-white sm:inline">Como funciona</a>
            <a href="#diferenciais" className="hidden text-white/70 hover:text-white sm:inline">Diferenciais</a>
            <a href="#responsabilidade" className="hidden text-white/70 hover:text-white md:inline">IA responsável</a>
            <Link to="/dashboard" className="btn rounded-lg bg-white/10 text-white hover:bg-white/15">Abrir painel</Link>
          </nav>
        </header>

        <section className="relative mx-auto grid max-w-7xl items-center gap-12 px-4 pb-20 pt-10 sm:px-6 lg:grid-cols-[1.1fr_1fr] lg:pb-28 lg:pt-16">
          <div>
            <p className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3 py-1 text-xs font-medium text-brand-200">
              <ShieldCheck size={14} aria-hidden /> Churn Prediction · IA explicável
            </p>
            <h1 className="mt-5 text-4xl font-semibold leading-[1.08] tracking-tight sm:text-5xl lg:text-[56px]">
              Antecipe o cancelamento.<br />
              <span className="text-brand-300">Retenha seus clientes.</span>
            </h1>
            <p className="mt-5 max-w-xl text-lg leading-relaxed text-white/75">
              Use Inteligência Artificial para identificar sinais de risco antes que o cancelamento aconteça.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/dashboard" className="btn bg-brand-500 px-5 py-3 text-base text-white hover:bg-brand-400">
                Analisar clientes <ArrowRight size={18} aria-hidden />
              </Link>
              <a href="#como-funciona" className="btn border border-white/20 px-5 py-3 text-base text-white hover:bg-white/5">Como funciona</a>
            </div>
          </div>
          <div className="flex justify-center lg:justify-end"><PreviewCard /></div>
        </section>
      </div>

      {/* Números */}
      <section aria-label="Indicadores" className="relative mx-auto -mt-10 max-w-7xl px-4 sm:px-6">
        <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-line bg-line shadow-[var(--shadow-card)] lg:grid-cols-4">
          {stats.map((s) => (
            <div key={s.label} className="bg-surface px-5 py-6">
              <dd className="tabular text-2xl font-semibold tracking-tight text-ink sm:text-3xl">{s.value}</dd>
              <dt className="mt-1 text-sm text-ink-2">{s.label}</dt>
            </div>
          ))}
        </dl>
        <p className="mt-2 text-xs text-muted">
          * Estimativa: receita mensal dos clientes em alto risco × {b?.meses_receita_preservada ?? "—"} meses × {fmtPct(b?.taxa_sucesso_retencao, 0)} de sucesso das ações (premissas configuráveis).
        </p>
      </section>

      {/* Como funciona */}
      <section id="como-funciona" className="mx-auto max-w-7xl scroll-mt-8 px-4 py-20 sm:px-6">
        <p className="text-sm font-semibold uppercase tracking-wider text-brand-600">Como funciona</p>
        <h2 className="mt-2 max-w-2xl text-3xl font-semibold tracking-tight">Dos dados à ação preventiva, com o ser humano no controle</h2>
        <ol className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {STEPS.map((s, i) => (
            <li key={s.title} className="card relative p-5">
              <span className="absolute right-4 top-4 text-xs font-semibold text-line-strong">0{i + 1}</span>
              <s.icon className="text-brand-600" size={22} aria-hidden />
              <h3 className="mt-3 font-semibold">{s.title}</h3>
              <p className="mt-1 text-sm leading-relaxed text-ink-2">{s.text}</p>
            </li>
          ))}
        </ol>
      </section>

      {/* Diferenciais */}
      <section id="diferenciais" className="border-y border-line bg-surface">
        <div className="mx-auto max-w-7xl scroll-mt-8 px-4 py-20 sm:px-6">
          <p className="text-sm font-semibold uppercase tracking-wider text-brand-600">Diferenciais</p>
          <h2 className="mt-2 max-w-2xl text-3xl font-semibold tracking-tight">Mais do que um classificador: um produto de decisão</h2>
          <div className="mt-10 grid gap-x-10 gap-y-8 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((f) => (
              <div key={f.title} className="flex gap-4">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-600"><f.icon size={20} aria-hidden /></span>
                <div>
                  <h3 className="font-semibold">{f.title}</h3>
                  <p className="mt-1 text-sm leading-relaxed text-ink-2">{f.text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Responsabilidade */}
      <section id="responsabilidade" className="mx-auto max-w-7xl scroll-mt-8 px-4 py-20 sm:px-6">
        <div className="grid gap-10 rounded-2xl bg-navy-800 p-8 text-white sm:p-12 lg:grid-cols-[1.2fr_1fr]">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wider text-brand-300">IA responsável</p>
            <blockquote className="mt-3 text-2xl font-medium leading-snug sm:text-3xl">
              “Probabilidade estimada de cancelamento” — nunca “este cliente vai cancelar”.
            </blockquote>
            <p className="mt-4 text-white/70">
              O modelo aponta onde olhar. Quem decide é a equipe. Correlação não é causalidade: sugestões são hipóteses a validar.
            </p>
          </div>
          <ul className="space-y-3 text-sm text-white/80">
            {["Sem idade, região ou dados sensíveis como entrada do modelo",
              "Acesso por chave e registro de quem consultou cada cliente",
              "Explicação individual para toda previsão",
              "Monitoramento contínuo de desempenho e drift",
              "Dados deste projeto são 100% sintéticos"].map((t) => (
              <li key={t} className="flex gap-2"><ShieldCheck size={17} className="mt-0.5 shrink-0 text-brand-300" aria-hidden />{t}</li>
            ))}
          </ul>
        </div>
        <div className="mt-12 flex flex-col items-center text-center">
          <h2 className="text-2xl font-semibold tracking-tight">Pronto para ver quem precisa de atenção hoje?</h2>
          <Link to="/dashboard" className="btn-primary mt-5 px-5 py-3 text-base">Analisar clientes <ArrowRight size={18} aria-hidden /></Link>
        </div>
      </section>

      <footer className="border-t border-line py-8 text-center text-xs text-muted">
        ChurnGuard AI · projeto de portfólio em Engenharia de Software e IA · Python, scikit-learn, SHAP, FastAPI, PostgreSQL, React
      </footer>
    </div>
  );
}
