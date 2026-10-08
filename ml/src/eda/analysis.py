"""Análise exploratória de dados (EDA) do ChurnGuard AI.

Funções puras: recebem DataFrames e devolvem tabelas ou figuras. São usadas
tanto pelo script `run_eda.py` (que gera o relatório em docs/eda.md) quanto
pelo notebook `ml/notebooks/01_eda.ipynb`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from ml.src.viz import style as S

# --------------------------------------------------------------------------- #
# Tabelas de qualidade e perfil
# --------------------------------------------------------------------------- #

def data_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Tipo, % ausente, cardinalidade e exemplo de cada coluna."""
    return pd.DataFrame(
        {
            "tipo": df.dtypes.astype(str),
            "ausentes": df.isna().sum(),
            "ausentes_%": (df.isna().mean() * 100).round(2),
            "valores_unicos": df.nunique(),
            "exemplo": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        }
    )


def raw_quality_summary(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in raw.items():
        rows.append(
            {
                "tabela": name,
                "registros": len(df),
                "colunas": df.shape[1],
                "duplicatas_exatas": int(df.duplicated().sum()),
                "células_ausentes_%": round(df.isna().mean().mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Contagem de outliers pela regra de Tukey (1,5 × IQR)."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"variável": c, "Q1": q1, "Q3": q3, "limite_inferior": lo, "limite_superior": hi,
                     "outliers": n_out, "outliers_%": round(n_out / len(s) * 100, 2), "máximo": s.max()})
    return pd.DataFrame(rows).round(2)


def class_comparison(df: pd.DataFrame, cols: list[str], target: str = "churn") -> pd.DataFrame:
    """Média e mediana por classe + diferença relativa."""
    g = df.groupby(target)[cols]
    mean, med = g.mean().T, g.median().T
    out = pd.DataFrame(
        {
            "média_permaneceu": mean[0],
            "média_cancelou": mean[1],
            "mediana_permaneceu": med[0],
            "mediana_cancelou": med[1],
        }
    )
    out["diferença_%"] = ((out.média_cancelou / out.média_permaneceu - 1) * 100).round(1)
    return out.round(2).sort_values("diferença_%", key=np.abs, ascending=False)


def add_eda_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas auxiliares SÓ para visualização (a engenharia de atributos oficial fica em ml/src/features)."""
    d = df.copy()
    d["acessos_semana"] = d.acessos_m1 / 4.33
    prev = d[["acessos_m4", "acessos_m5", "acessos_m6"]].mean(axis=1)
    recent = d[["acessos_m1", "acessos_m2", "acessos_m3"]].mean(axis=1)
    d["variacao_uso_3m"] = np.where(prev > 0, recent / prev - 1, np.nan)
    d["faixa_preco"] = pd.cut(d.valor_mensal, [0, 60, 120, 250, 10_000],
                              labels=["até R$ 60", "R$ 60–120", "R$ 120–250", "acima de R$ 250"])
    d["faixa_tempo"] = pd.cut(d.tempo_cliente_meses, [0, 3, 6, 12, 24, 999],
                              labels=["até 3 meses", "3–6 meses", "6–12 meses", "1–2 anos", "mais de 2 anos"])
    d["faixa_uso"] = pd.cut(d.acessos_semana, [-0.1, 2, 5, 10, 20, 1e6],
                            labels=["até 2", "2–5", "5–10", "10–20", "mais de 20"])
    d["faixa_reclamacoes"] = pd.cut(d.reclamacoes_6m, [-1, 0, 1, 2, 999], labels=["0", "1", "2", "3 ou mais"])
    d["faixa_atrasos"] = pd.cut(d.faturas_atrasadas_6m, [-1, 0, 1, 2, 999], labels=["0", "1", "2", "3 ou mais"])
    d["faixa_idade"] = pd.cut(d.idade.where(d.idade.between(18, 100)), [17, 25, 35, 45, 60, 100],
                              labels=["18–25", "26–35", "36–45", "46–60", "60+"])
    return d


def churn_rate_table(df: pd.DataFrame, col: str, order=None) -> pd.DataFrame:
    t = df.groupby(col, observed=True).churn.agg(taxa="mean", clientes="size", churns="sum")
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    return t


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

def _wilson(p, n, z=1.96):
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def plot_rate_bars(df: pd.DataFrame, col: str, title: str, subtitle: str,
                   order=None, xlabel: str | None = None, ax=None):
    """Barras horizontais com a taxa de churn por categoria, IC 95% e média geral."""
    t = churn_rate_table(df, col, order)
    overall = df.churn.mean()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(t) + 1.9))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t.taxa, height=0.62, color=S.BLUE, zorder=2)
    lo, hi = _wilson(t.taxa.to_numpy(), t.clientes.to_numpy())
    ax.hlines(y, lo, hi, color=S.INK_2, lw=1, zorder=3)
    ax.axvline(overall, color=S.INK_2, ls=(0, (3, 3)), lw=1, zorder=1)
    ax.text(overall, -0.62, f" média geral {S.pct(overall)}", fontsize=8.3, color=S.INK_2, va="bottom")
    xmax = max(hi.max(), overall) * 1.28
    for yi, (rate, n) in zip(y, t[["taxa", "clientes"]].itertuples(index=False)):
        ax.text(max(rate, hi[list(y).index(yi)]) + xmax * 0.012, yi, f"{S.pct(rate)}  ·  n={S.num(n)}",
                va="center", fontsize=8.8, color=S.INK)
    ax.set_yticks(y, [str(i) for i in t.index])
    ax.set_ylim(-0.7, len(t) - 0.4)
    ax.set_xlim(0, xmax * 1.18)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel or f"Taxa de churn em {60} dias")
    S.title(ax, title, subtitle)
    return ax.figure if own else ax


def plot_target_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    counts = df.churn.value_counts().sort_index()
    ax = axes[0]
    bars = ax.bar([S.CLASS_LABELS[i] for i in counts.index], counts.values,
                  color=[S.CLASS_COLORS[i] for i in counts.index], width=0.55, zorder=2)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{S.num(v)}\n({S.pct(v / counts.sum())})",
                ha="center", va="bottom", fontsize=9.5, color=S.INK)
    ax.set_ylim(0, counts.max() * 1.22)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.0f} mil"))
    ax.grid(axis="x", visible=False)
    S.title(ax, "Distribuição da variável-alvo", "Observações cliente × mês (snapshots rotulados)")

    ax = axes[1]
    r = df.groupby("data_referencia").churn.mean()
    ax.plot(r.index, r.values, color=S.BLUE, marker="o", ms=4.5, zorder=3)
    ax.axhline(df.churn.mean(), color=S.INK_2, ls=(0, (3, 3)), lw=1)
    ax.text(r.index[0], df.churn.mean(), f"média {S.pct(df.churn.mean())}", fontsize=8.3, color=S.INK_2, va="bottom")
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.04))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylim(0, r.max() * 1.25)
    peak = r.idxmax()
    ax.annotate(f"pico {S.pct(r.max())}\n(reajuste de janeiro)", (peak, r.max()), xytext=(10, 8),
                textcoords="offset points", fontsize=8.5, color=S.INK_2)
    S.title(ax, "Taxa de churn em 60 dias por data de referência",
            "Proporção de clientes ativos que cancelaram nos 60 dias seguintes")
    fig.tight_layout(w_pad=3)
    return fig


def plot_monthly_churn(customers: pd.DataFrame, adjustments: pd.DataFrame, start="2023-01-01", end="2026-09-01"):
    """Taxa mensal de cancelamento calculada a partir do cadastro (visão de negócio)."""
    c = customers.drop_duplicates()
    rows = []
    for m in pd.date_range(start, end, freq="MS"):
        nxt = m + pd.offsets.MonthBegin()
        active = ((c.data_inicio < m) & (c.data_cancelamento.isna() | (c.data_cancelamento >= m))).sum()
        churned = ((c.data_cancelamento >= m) & (c.data_cancelamento < nxt)).sum()
        rows.append((m, active, churned))
    t = pd.DataFrame(rows, columns=["mes", "ativos", "cancelamentos"]).set_index("mes")
    t["taxa"] = t.cancelamentos / t.ativos

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    ax1.plot(t.index, t.taxa, color=S.BLUE, marker="o", ms=4, zorder=3)
    for y, pctv in adjustments[["ano", "percentual"]].itertuples(index=False):
        d = pd.Timestamp(f"{y}-01-01")
        if t.index.min() <= d <= t.index.max():
            ax1.axvspan(d, d + pd.DateOffset(months=2), color=S.GRID, alpha=0.6, lw=0, zorder=0)
            ax1.text(d, t.taxa.max() * 1.1, f" reajuste {int(pctv*100)}%", fontsize=8.3, color=S.INK_2, va="top")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
    ax1.set_ylim(0, t.taxa.max() * 1.18)
    S.title(ax1, "Evolução da taxa mensal de cancelamento",
            "Cancelamentos no mês ÷ clientes ativos no início do mês. Faixas cinzas: 2 meses após o reajuste anual")
    ax2.bar(t.index, t.ativos, width=20, color=S.SEQ_BLUE[1], zorder=2)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.1f} mil".replace(".", ",")))
    ax2.set_ylabel("Clientes ativos")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout(h_pad=1.5)
    return fig, t


def plot_usage_trajectory(df: pd.DataFrame):
    """Mediana de acessos nos 6 meses antes da referência, indexada ao mês -6."""
    cols = [f"acessos_m{i}" for i in range(6, 0, -1)]
    sub = df.dropna(subset=cols)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    x = np.arange(-5, 1)
    labels = ["-5", "-4", "-3", "-2", "-1", "mês da\nreferência"]
    for cls in (0, 1):
        g = sub[sub.churn == cls][cols]
        axes[0].plot(x, g.mean().values, color=S.CLASS_COLORS[cls], marker="o", ms=5, label=S.CLASS_LABELS[cls])
        idx = (g.div(g[cols[0]].replace(0, np.nan), axis=0)).median()
        axes[1].plot(x, idx.values * 100, color=S.CLASS_COLORS[cls], marker="o", ms=5, label=S.CLASS_LABELS[cls])
        axes[1].text(0.12, idx.values[-1] * 100, S.CLASS_LABELS[cls], color=S.INK, fontsize=9, va="center")
        axes[0].text(0.12, g.mean().values[-1], S.CLASS_LABELS[cls], color=S.INK, fontsize=9, va="center")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlim(-5.3, 1.1)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("Acessos por mês (média)")
    axes[1].set_ylabel("Índice (mês -5 = 100)")
    axes[1].axhline(100, color=S.AXIS, lw=1)
    S.title(axes[0], "Uso nos meses anteriores à referência", "Média de acessos mensais por grupo")
    S.title(axes[1], "Mudança de comportamento", "Mediana do uso relativo ao início da janela")
    fig.tight_layout(w_pad=3)
    return fig


def plot_numeric_distributions(df: pd.DataFrame, cols: dict[str, str], clip_q: float = 0.99):
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.1 * nrows))
    for ax, (c, label) in zip(axes.flat, cols.items()):
        s = df[[c, "churn"]].dropna()
        hi = s[c].quantile(clip_q)
        s = s[s[c] <= hi]
        discrete = s[c].nunique() <= 20
        for cls in (0, 1):
            v = s[s.churn == cls][c]
            if discrete:
                vc = v.value_counts(normalize=True).sort_index()
                off = -0.2 if cls == 0 else 0.2
                ax.bar(vc.index + off, vc.values, width=0.38, color=S.CLASS_COLORS[cls], label=S.CLASS_LABELS[cls], zorder=2)
                ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
            else:
                ax.hist(v, bins=30, density=True, histtype="stepfilled", alpha=0.35, color=S.CLASS_COLORS[cls], zorder=2)
                ax.hist(v, bins=30, density=True, histtype="step", lw=1.6, color=S.CLASS_COLORS[cls],
                        label=S.CLASS_LABELS[cls], zorder=3)
        ax.set_title(label, fontsize=10.5, pad=8)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper right")
    fig.suptitle("Distribuição das principais variáveis por grupo", x=0.01, y=1.0, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.962, "Densidades normalizadas (cada grupo soma 100%). Valores acima do percentil 99 omitidos para legibilidade.",
             fontsize=9, color=S.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=2)
    return fig


def plot_correlation(df: pd.DataFrame, cols: list[str], labels: dict[str, str] | None = None):
    corr = df[cols].corr(method="spearman")
    if labels:
        corr = corr.rename(index=labels, columns=labels)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    cmap = sns.blend_palette(S.DIVERGING, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7.5}, linewidths=1.5, linecolor=S.SURFACE, cbar_kws={"shrink": 0.6, "label": "ρ de Spearman"}, ax=ax)
    ax.grid(False)
    ax.tick_params(length=0)
    S.title(ax, "Correlação entre variáveis (Spearman)",
            "Spearman captura relações monotônicas e é robusto a outliers. Correlação não implica causalidade.")
    return fig, corr


def plot_outliers(df: pd.DataFrame, cols: dict[str, str], log_cols: tuple[str, ...] = ()):
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 4.2))
    for ax, (c, label) in zip(axes, cols.items()):
        s = df[c].dropna()
        ax.boxplot(s, vert=True, widths=0.5, patch_artist=True, showfliers=True,
                   boxprops=dict(facecolor=S.SEQ_BLUE[0], edgecolor=S.BLUE, lw=1.2),
                   medianprops=dict(color=S.INK, lw=1.6), whiskerprops=dict(color=S.BLUE),
                   capprops=dict(color=S.BLUE),
                   flierprops=dict(marker="o", ms=3, mfc=S.ORANGE, mec="none", alpha=0.45))
        ax.set_xticks([])
        if c in log_cols:
            ax.set_yscale("symlog", linthresh=10)
            label += "\n(escala log)"
        ax.set_title(label, fontsize=10, pad=8)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Identificação de outliers (boxplot, regra de 1,5 × IQR)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_missing(df: pd.DataFrame, labels: dict[str, str] | None = None):
    m = df.isna().mean().sort_values()
    m = m[m > 0]
    if labels:
        m.index = [labels.get(i, i) for i in m.index]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(m) + 1.8))
    ax.barh(m.index, m.values, color=S.BLUE, height=0.6, zorder=2)
    for i, v in enumerate(m.values):
        ax.text(v + 0.004, i, S.pct(v), va="center", fontsize=8.8)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlim(0, m.max() * 1.18)
    ax.grid(axis="y", visible=False)
    S.title(ax, "Valores ausentes na base analítica",
            "Ausência estrutural (sem histórico/sem chamado/sem pesquisa) ≠ erro de coleta")
    return fig


def plot_fairness_check(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    plot_rate_bars(df, "regiao", "Churn por região", "Atributo geográfico — não deve orientar decisões",
                   order=["Sudeste", "Nordeste", "Sul", "Centro-Oeste", "Norte"], ax=axes[0])
    plot_rate_bars(df, "faixa_idade", "Churn por faixa etária", "Atributo pessoal — avaliar necessidade (LGPD)",
                   order=["18–25", "26–35", "36–45", "46–60", "60+"], ax=axes[1])
    fig.tight_layout(w_pad=4)
    return fig


def plot_contract_panel(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
    plot_rate_bars(df, "ciclo_contrato", "Por tipo de contrato", "Fidelidade de 12 meses vs. mensal", ax=axes[0])
    d = df.assign(renov=df.renovacao_automatica.map({1: "Ativada", 0: "Desativada"}))
    plot_rate_bars(d, "renov", "Por renovação automática", "Fricção para cancelar vs. decisão ativa", ax=axes[1])
    plot_rate_bars(df, "canal_aquisicao", "Por canal de aquisição", "Qualidade do lead na entrada", ax=axes[2])
    fig.tight_layout(w_pad=3)
    return fig
