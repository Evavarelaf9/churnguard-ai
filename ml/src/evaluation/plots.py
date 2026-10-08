"""Figuras de avaliação e explicabilidade do modelo (mesma identidade visual da EDA)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from ml.src.viz import style as S

MODEL_COLORS = {"logreg": S.BLUE, "random_forest": S.AQUA, "gradient_boosting": S.ORANGE}
MODEL_SHORT = {"logreg": "Regressão Logística", "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting"}
RISK_COLORS = {"Alto": S.RISK_HIGH, "Médio": S.RISK_MEDIUM, "Baixo": S.RISK_LOW}
INITIAL, TUNED = S.SEQ_BLUE[1], S.SEQ_BLUE[4]


def model_comparison(results: dict):
    keys = list(results["tuned"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    metrics = [("pr_auc", "PR-AUC", "cv"), ("roc_auc", "ROC-AUC", "cv"), ("pr_auc", "PR-AUC no teste", "test")]
    for ax, (m, title, where) in zip(axes, metrics):
        y = np.arange(len(keys))[::-1]
        ini = [results["baseline"][k][where][m] for k in keys]
        tun = [results["tuned"][k][where][m] for k in keys]
        ax.barh(y + 0.19, ini, height=0.36, color=INITIAL, label="Inicial (padrão)", zorder=2)
        ax.barh(y - 0.19, tun, height=0.36, color=TUNED, label="Ajustado", zorder=2)
        for yi, a, b in zip(y, ini, tun):
            ax.text(a + 0.003, yi + 0.19, f"{a:.3f}".replace(".", ","), va="center", fontsize=8.5)
            ax.text(b + 0.003, yi - 0.19, f"{b:.3f}".replace(".", ","), va="center", fontsize=8.5, fontweight="bold")
        ax.set_yticks(y, [MODEL_SHORT[k] for k in keys])
        lo = min(ini + tun)
        ax.set_xlim(lo * 0.85 if m == "roc_auc" else 0, max(ini + tun) * 1.12)
        ax.grid(axis="y", visible=False)
        S.title(ax, title, "Média das 3 dobras temporais" if where == "cv" else "Período de teste (dez/25 → jul/26)")
        if where == "test" and m == "pr_auc":
            ax.axvline(results["split"]["test_churn_rate"], color=S.INK_2, ls=(0, (3, 3)), lw=1)
            ax.text(results["split"]["test_churn_rate"], len(keys) - 0.45, " modelo aleatório", fontsize=8, color=S.INK_2)
        ax.set_ylim(-0.6, len(keys) - 0.3)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.06))
    fig.tight_layout(w_pad=3)
    return fig


def roc_pr_curves(scores: pd.DataFrame, selected: str):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    y = scores.churn.to_numpy()
    for k in MODEL_SHORT:
        lw = 2.6 if k == selected else 1.6
        fpr, tpr, _ = roc_curve(y, scores[k])
        axes[0].plot(fpr, tpr, color=MODEL_COLORS[k], lw=lw, label=MODEL_SHORT[k])
        pr, rc, _ = precision_recall_curve(y, scores[k])
        axes[1].plot(rc, pr, color=MODEL_COLORS[k], lw=lw, label=MODEL_SHORT[k])
    axes[0].plot([0, 1], [0, 1], color=S.AXIS, ls=(0, (3, 3)), lw=1)
    axes[0].set_xlabel("Taxa de falsos positivos")
    axes[0].set_ylabel("Recall (taxa de verdadeiros positivos)")
    axes[1].axhline(y.mean(), color=S.AXIS, ls=(0, (3, 3)), lw=1)
    axes[1].text(0.62, y.mean() + 0.01, f"taxa base {S.pct(y.mean())}", fontsize=8.3, color=S.INK_2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1)
    S.title(axes[0], "Curva ROC", "Capacidade de ordenar clientes por risco")
    S.title(axes[1], "Curva Precision–Recall", "Mais informativa com classes desbalanceadas")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right")
    fig.tight_layout(w_pad=3)
    return fig


def imbalance_plot(imbalance: list[dict], model_name: str):
    d = pd.DataFrame(imbalance)
    d = d[d.modelo == model_name].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    y = np.arange(len(d))[::-1]
    cols = [("pr_auc", "PR-AUC (ordenação)", "maior é melhor"),
            ("brier", "Brier score (calibração)", "menor é melhor"),
            ("prob_media", "Probabilidade média prevista", f"taxa real ≈ {S.pct(d.base_rate.iloc[0])}")]
    for ax, (c, title, sub) in zip(axes, cols):
        ax.hlines(y, 0, d[c], color=S.GRID, lw=2, zorder=1)
        ax.scatter(d[c], y, s=70, color=[S.ORANGE if e == "Nenhuma" else S.BLUE for e in d.estrategia], zorder=3)
        for yi, v in zip(y, d[c]):
            ax.text(v, yi + 0.28, f"{v:.3f}".replace(".", ","), ha="center", fontsize=8.3)
        if c == "prob_media":
            ax.axvline(d.base_rate.iloc[0], color=S.INK_2, ls=(0, (3, 3)), lw=1)
        ax.set_xlim(0, d[c].max() * 1.2)
        ax.set_ylim(-0.5, len(d) - 0.2)
        ax.grid(axis="y", visible=False)
        S.title(ax, title, sub)
    axes[0].set_yticks(y, d.estrategia)
    fig.suptitle(f"Estratégias de desbalanceamento — {model_name} (CV temporal)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", y=1.08)
    fig.tight_layout(w_pad=2)
    return fig


def confusion_matrices(y, proba, thresholds: list[tuple[float, str]]):
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5.4 * len(thresholds), 4.6))
    for ax, (t, label) in zip(np.atleast_1d(axes), thresholds):
        cm = confusion_matrix(y, (proba >= t).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap=plt.matplotlib.colors.ListedColormap([S.SURFACE]), vmin=0, vmax=1)
        names = [["Verdadeiro negativo", "Falso positivo"], ["Falso negativo", "Verdadeiro positivo"]]
        tints = [[S.SEQ_BLUE[0], "#fbe0d4"], ["#fbe0d4", S.SEQ_BLUE[1]]]
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, color=tints[i][j], zorder=1))
                ax.text(j, i - 0.08, S.num(cm[i, j]), ha="center", va="center", fontsize=17, fontweight="bold", zorder=2)
                ax.text(j, i + 0.2, names[i][j], ha="center", va="center", fontsize=8.8, color=S.INK_2, zorder=2)
        ax.set_xticks([0, 1], ["Previsto: permanece", "Previsto: cancela"])
        ax.set_yticks([0, 1], ["Real: permaneceu", "Real: cancelou"])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        S.title(ax, label, f"threshold {t:.2f} · precision {S.pct(prec)} · recall {S.pct(rec)}".replace("0.", "0,"))
    fig.tight_layout(w_pad=3)
    return fig


def threshold_plot(table: pd.DataFrame, t_op: float, t_med: float):
    t = table.sort_values("threshold")
    t = t[t.threshold <= 0.6]  # acima disso há pouquíssimos clientes e a precision fica instável
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    for col, color, label in [("precision", S.BLUE, "Precision"), ("recall", S.ORANGE, "Recall"), ("f1", S.VIOLET, "F1")]:
        ax.plot(t.threshold, t[col], color=color, label=label)
    for x in (0.3, 0.5, 0.7):
        ax.axvline(x, color=S.AXIS, lw=1, ls=(0, (2, 3)))
    ax.axvline(t_op, color=S.RISK_HIGH, lw=1.6)
    ax.text(t_op + 0.008, 0.05, f"operacional\n{t_op:.2f}".replace(".", ","), fontsize=8.5, color=S.INK)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Threshold (probabilidade mínima para abordar)")
    ax.set_xlim(0.03, 0.62)
    ax.set_ylim(0, 1)
    ax.legend(loc="center right")
    S.title(ax, "Precision, Recall e F1 por threshold", "Previsões out-of-fold do treino (o teste não é usado para escolher)")

    ax = axes[1]
    ax.plot(t.threshold, t.percentual_abordados, color=S.BLUE, label="Clientes abordados (% da base)")
    ax.plot(t.threshold, t.churns_identificados / t.churns_totais, color=S.ORANGE, label="Churns identificados (% do total)")
    ax.axvline(t_op, color=S.RISK_HIGH, lw=1.6)
    ax.axvline(t_med, color=S.RISK_MEDIUM, lw=1.4, ls=(0, (4, 2)))
    ax.text(t_med + 0.008, 0.85, f"início do\nrisco médio\n{t_med:.2f}".replace(".", ","), fontsize=8.3, color=S.INK)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlim(0.03, 0.62)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Threshold")
    ax.legend(loc="upper right")
    S.title(ax, "Esforço × cobertura", "Quantos clientes a equipe aborda e quantos cancelamentos encontra")
    fig.tight_layout(w_pad=3)
    return fig


def business_value_plot(table: pd.DataFrame, t_op: float, business: dict):
    t = table.sort_values("threshold")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(t.threshold, t.resultado_liquido / 1000, color=S.BLUE, zorder=3)
    ax.axhline(0, color=S.AXIS, lw=1)
    best = t.loc[t.threshold.sub(t_op).abs().idxmin()]
    ax.scatter([t_op], [best.resultado_liquido / 1000], s=70, color=S.RISK_HIGH, zorder=4)
    ax.annotate(f"máximo: R$ {best.resultado_liquido/1000:,.0f} mil\nthreshold {t_op:.2f}".replace(".", ","),
                (t_op, best.resultado_liquido / 1000), xytext=(20, 18), textcoords="offset points", fontsize=9)
    top = t.resultado_liquido.max() / 1000
    ax.set_ylim(-0.8 * top, top * 1.45)
    ax.text(0.035, -0.75 * top, "thresholds baixos: abordar quase todos custa mais do que preserva ↓", fontsize=8.3, color=S.INK_2)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"R$ {v:,.0f} mil".replace(",", ".")))
    ax.set_xlabel("Threshold")
    S.title(ax, "Resultado líquido esperado da campanha de retenção",
            f"Premissas: custo R$ {business['custo_acao']:.0f}/cliente · sucesso {S.pct(business['taxa_sucesso_retencao'], 0)} · "
            f"{business['meses_receita_preservada']} meses preservados (out-of-fold do treino)")
    fig.tight_layout()
    return fig


def calibration_plot(y, proba):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    frac, mean = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    ax = axes[0]
    ax.plot([0, mean.max() * 1.1], [0, mean.max() * 1.1], color=S.AXIS, ls=(0, (3, 3)), lw=1)
    ax.plot(mean, frac, color=S.BLUE, marker="o", ms=6)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Probabilidade média prevista (decis)")
    ax.set_ylabel("Taxa real de churn")
    S.title(ax, "Calibração", "Pontos sobre a diagonal: 20% previsto ≈ 20% real")
    ax = axes[1]
    ax.hist(proba, bins=40, color=S.SEQ_BLUE[3], zorder=2)
    ax.set_xlabel("Probabilidade prevista")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    S.title(ax, "Distribuição das probabilidades", "Período de teste")
    fig.tight_layout(w_pad=3)
    return fig


def gains_plot(lift: pd.DataFrame):
    d = pd.DataFrame(lift)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.bar(d.decil, d.taxa_real, color=S.BLUE, width=0.7, zorder=2)
    for x, v, l in zip(d.decil, d.taxa_real, d.lift):
        ax.text(x, v + 0.005, f"{S.pct(v)}\n{l:.1f}×".replace(".", ","), ha="center", fontsize=8.3)
    ax.axhline(d.churns.sum() / d.clientes.sum(), color=S.INK_2, ls=(0, (3, 3)), lw=1)
    ax.set_xticks(d.decil)
    ax.set_xlabel("Decil de risco previsto (1 = 10% com maior risco)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylim(0, d.taxa_real.max() * 1.3)
    ax.grid(axis="x", visible=False)
    S.title(ax, "Taxa real de churn por decil de risco", "Lift = quantas vezes acima da média")
    ax = axes[1]
    x = np.r_[0, d.decil / 10]
    ax.plot(x, np.r_[0, d.captura_acumulada], color=S.ORANGE, marker="o", ms=5, label="Modelo")
    ax.plot([0, 1], [0, 1], color=S.AXIS, ls=(0, (3, 3)), lw=1, label="Aleatório")
    k = 2
    ax.annotate(f"abordando 20% da base,\nencontra {S.pct(d.captura_acumulada.iloc[k-1], 0)} dos churns",
                (0.2, d.captura_acumulada.iloc[k - 1]), xytext=(25, -30), textcoords="offset points", fontsize=8.8,
                arrowprops=dict(arrowstyle="-", color=S.INK_2, lw=0.8))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlabel("% de clientes abordados (do maior para o menor risco)")
    ax.set_ylabel("% dos churns encontrados")
    ax.legend(loc="lower right")
    S.title(ax, "Curva de ganho acumulado", "Período de teste")
    fig.tight_layout(w_pad=3)
    return fig


def risk_groups_plot(risk_test: list[dict], scoring: dict, thresholds: dict):
    d = pd.DataFrame(risk_test)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    order = ["Baixo", "Médio", "Alto"]
    d = d.set_index("risco").reindex(order)
    bars = ax.bar(order, d.taxa_real, color=[RISK_COLORS[r] for r in order], width=0.55, zorder=2)
    for b, v, n in zip(bars, d.taxa_real, d.clientes):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{S.pct(v)}\nn={S.num(n)}", ha="center", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylim(0, d.taxa_real.max() * 1.3)
    ax.grid(axis="x", visible=False)
    S.title(ax, "Taxa real de churn por nível de risco", "Período de teste — os níveis separam bem o risco real")
    ax = axes[1]
    dist = scoring["distribuicao_risco"]
    total = sum(dist.values())
    y = np.arange(3)[::-1]
    vals = [dist[r] for r in order]
    ax.barh(y, vals, color=[RISK_COLORS[r] for r in order], height=0.55, zorder=2)
    for yi, v in zip(y, vals):
        ax.text(v + total * 0.01, yi, f"{S.num(v)} ({S.pct(v / total, 0)})", va="center", fontsize=9.5)
    ax.set_yticks(y, order)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Clientes ativos")
    S.title(ax, "Clientes ativos hoje por nível de risco",
            f"Baixo < {thresholds['medio']:.2f} ≤ Médio < {thresholds['alto']:.2f} ≤ Alto".replace(".", ","))
    fig.tight_layout(w_pad=3)
    return fig


def importance_plot(perm: list[dict], top: int = 15):
    d = pd.DataFrame(perm).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * top + 1.6))
    ax.barh(d.label, d.importance, xerr=d["std"], color=S.BLUE, height=0.6, zorder=2,
            error_kw=dict(ecolor=S.INK_2, lw=1, capsize=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Queda na PR-AUC ao embaralhar o atributo")
    S.title(ax, "Importância por permutação", "Quanto o modelo piora sem a informação (teste, 5 repetições)")
    fig.tight_layout()
    return fig


def shap_summary_plot(shap_df: pd.DataFrame, feat_df: pd.DataFrame, labels: dict, top: int = 14):
    cols = shap_df.drop(columns="churn", errors="ignore").abs().mean().sort_values(ascending=False).head(top).index
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [S.SEQ_BLUE[5], "#d9d7d0", S.RISK_HIGH])
    fig, ax = plt.subplots(figsize=(10.5, 0.45 * top + 1.6))
    rng = np.random.default_rng(0)
    for i, c in enumerate(cols[::-1]):
        sv = shap_df[c].to_numpy()
        x = feat_df[c]
        if x.dtype.kind in "biuf":
            xv = x.astype(float)
            ranks = xv.rank(pct=True).fillna(0.5).to_numpy()
            colors = cmap(ranks)
        else:
            colors = S.MUTED
        ax.scatter(sv, i + rng.uniform(-0.3, 0.3, len(sv)), s=5, c=colors, alpha=0.55, linewidths=0, zorder=2)
    ax.axvline(0, color=S.AXIS, lw=1)
    ax.set_yticks(range(len(cols)), [labels.get(c, c) for c in cols[::-1]])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribuição SHAP (log-odds) · ← reduz o risco | aumenta o risco →")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(["valor\nbaixo", "valor\nalto"], fontsize=8)
    cb.outline.set_visible(False)
    S.title(ax, "Como cada fator influencia o risco (SHAP)",
            "Cada ponto é um cliente. Cor = valor do atributo; posição = efeito na previsão. Cinza = categórico")
    fig.tight_layout()
    return fig


def lr_coefficients_plot(coefs: list[dict], labels: dict, top: int = 15):
    d = pd.DataFrame(coefs).head(top)
    d["label"] = [labels.get(f, f.replace("_", " ")) for f in d.feature]
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * top + 1.6))
    ax.barh(d.label, d.coef, color=[S.RISK_HIGH if c > 0 else S.BLUE for c in d.coef], height=0.6, zorder=2)
    ax.axvline(0, color=S.AXIS, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Coeficiente (variáveis padronizadas) · vermelho aumenta o risco, azul reduz")
    S.title(ax, "Regressão Logística: coeficientes", "Baseline interpretável — confirma a direção dos efeitos")
    fig.tight_layout()
    return fig


def individual_explanation_plot(expl: dict, customer: str):
    factors = expl["fatores_risco"] + expl["fatores_protecao"]
    d = pd.DataFrame(factors).sort_values("impact")
    fig, ax = plt.subplots(figsize=(10.5, 0.5 * len(d) + 1.8))
    colors = [S.RISK_HIGH if v > 0 else S.BLUE for v in d.impact]
    ax.barh(range(len(d)), d.impact, color=colors, height=0.6, zorder=2)
    for i, (v, txt) in enumerate(zip(d.impact, d.text)):
        ax.text(v + (0.02 if v > 0 else -0.02), i, txt, va="center", ha="left" if v > 0 else "right", fontsize=8.8)
    ax.axvline(0, color=S.AXIS, lw=1)
    ax.set_yticks([])
    lim = d.impact.abs().max() * 2.3
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribuição para o risco (SHAP, log-odds)")
    S.title(ax, f"Por que o cliente {customer} está em risco {expl['nivel_risco'].lower()}?",
            f"Probabilidade estimada de cancelamento em 60 dias: {S.pct(expl['probabilidade'], 0)}")
    fig.tight_layout()
    return fig


def predicted_vs_actual_plot(pred: pd.DataFrame):
    g = pred.groupby("data_referencia").agg(real=("churn", "mean"), previsto=("probabilidade", "mean"))
    fig, ax = plt.subplots(figsize=(10.5, 4))
    ax.plot(g.index, g.real, color=S.ORANGE, marker="o", ms=5, label="Taxa real")
    ax.plot(g.index, g.previsto, color=S.BLUE, marker="o", ms=5, label="Probabilidade média prevista")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylim(0, max(g.real.max(), g.previsto.max()) * 1.3)
    ax.legend(loc="upper right")
    S.title(ax, "Previsto × real por data de referência (teste)", "Jan–mar/2026: reajuste de 9% gerou mais churn do que o padrão aprendido — exemplo de drift (ver monitoramento)")
    fig.tight_layout()
    return fig
