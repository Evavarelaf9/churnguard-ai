import clsx from "clsx";
import { Activity, BrainCircuit, ClipboardList, Home, LayoutDashboard, Menu, Radar, Users, X } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import { useHealth } from "../../hooks/useApi";
import { fmtDate } from "../../lib/format";
import { Logo } from "./Logo";
import { ThresholdControl } from "./ThresholdControl";

const NAV = [
  { to: "/dashboard", label: "Visão geral", icon: LayoutDashboard },
  { to: "/clientes", label: "Clientes", icon: Users },
  { to: "/acoes", label: "Ações de retenção", icon: ClipboardList },
  { to: "/modelo", label: "Modelo", icon: BrainCircuit },
  { to: "/monitoramento", label: "Monitoramento", icon: Radar },
];

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Navegação principal" className="flex h-full flex-col bg-navy-800 px-3 py-5 text-white">
      <div className="px-2"><Logo light /></div>
      <ul className="mt-8 space-y-1">
        {NAV.map(({ to, label, icon: Icon }) => (
          <li key={to}>
            <NavLink to={to} onClick={onNavigate}
              className={({ isActive }) => clsx(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive ? "bg-white/10 text-white" : "text-white/65 hover:bg-white/5 hover:text-white",
              )}>
              <Icon size={18} aria-hidden /> {label}
            </NavLink>
          </li>
        ))}
      </ul>
      <div className="mt-auto space-y-3 px-2">
        <NavLink to="/" onClick={onNavigate} className="flex items-center gap-2 text-xs text-white/55 hover:text-white">
          <Home size={14} aria-hidden /> Página inicial
        </NavLink>
        <p className="rounded-lg bg-white/5 p-3 text-[11px] leading-relaxed text-white/60">
          Previsões são <b className="text-white/80">probabilidades estimadas</b>, não certezas. A decisão é sempre da equipe.
        </p>
      </div>
    </nav>
  );
}

export function AppLayout() {
  const [open, setOpen] = useState(false);
  const { data: health, isError } = useHealth();
  const loc = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [loc.pathname]);
  const ok = health?.status === "ok" && !isError;

  return (
    <div className="min-h-screen lg:pl-60">
      <a href="#conteudo" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2">
        Pular para o conteúdo
      </a>
      <aside className="fixed inset-y-0 left-0 hidden w-60 lg:block"><Sidebar /></aside>
      {open && (
        <div className="fixed inset-0 z-50 lg:hidden" role="dialog" aria-modal="true" aria-label="Menu">
          <div className="absolute inset-0 bg-navy-900/50" onClick={() => setOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-64">
            <Sidebar onNavigate={() => setOpen(false)} />
            <button className="absolute right-3 top-4 text-white/70" onClick={() => setOpen(false)} aria-label="Fechar menu"><X size={20} /></button>
          </div>
        </div>
      )}
      <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-line bg-surface/90 px-4 backdrop-blur sm:px-6">
        <button className="btn-ghost p-1.5 lg:hidden" onClick={() => setOpen(true)} aria-label="Abrir menu"><Menu size={20} /></button>
        <div className="lg:hidden"><Logo compact /></div>
        <div className="hidden items-center gap-2 text-xs text-ink-2 md:flex">
          <span className="inline-flex items-center gap-1.5">
            <span className={clsx("h-2 w-2 rounded-full", ok ? "bg-risk-low" : isError ? "bg-risk-high" : "bg-line-strong")} aria-hidden />
            {ok ? "API conectada" : isError ? "API indisponível" : "Verificando API…"}
          </span>
          {health && (
            <>
              <span className="text-line-strong" aria-hidden>·</span>
              <span className="inline-flex items-center gap-1"><Activity size={13} aria-hidden /> Modelo v{health.model_version}</span>
              <span className="text-line-strong" aria-hidden>·</span>
              <span>Dados até {fmtDate(health.data_referencia)}</span>
            </>
          )}
        </div>
        <div className="ml-auto"><ThresholdControl /></div>
      </header>
      <main id="conteudo" className="mx-auto max-w-[1400px] px-4 py-6 sm:px-6 lg:py-8">
        <Outlet />
      </main>
    </div>
  );
}
