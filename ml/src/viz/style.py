"""Identidade visual dos gráficos do ChurnGuard AI (Matplotlib/Seaborn).

As mesmas cores são usadas no frontend (Tailwind), para que relatório,
notebook e dashboard pareçam um único produto.

Regras:
- Cor segue a entidade: "Permaneceu" é sempre azul, "Cancelou" sempre laranja.
- Níveis de risco usam a paleta de status (verde/âmbar/vermelho) e sempre vêm
  acompanhados de rótulo — nunca só cor.
- Magnitude (taxa de churn) usa um único tom de azul.
- Eixos e grades discretos; texto nunca na cor da série.
"""
from __future__ import annotations

import matplotlib as mpl

# Superfícies e tinta
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Séries categóricas (ordem fixa)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"

RETAINED = BLUE
CHURNED = ORANGE
CLASS_COLORS = {0: RETAINED, 1: CHURNED}
CLASS_LABELS = {0: "Permaneceu", 1: "Cancelou"}

# Status / risco
RISK_LOW = "#0ca30c"
RISK_MEDIUM = "#fab219"
RISK_HIGH = "#d03b3b"

# Rampa sequencial (azul) e divergente (azul ↔ vermelho, meio cinza)
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32d2d"]


def apply() -> None:
    """Aplica o tema global do projeto."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.labelsize": 9.5,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
        }
    )


def title(ax, text: str, subtitle: str | None = None) -> None:
    """Título à esquerda com subtítulo explicativo em cinza."""
    ax.set_title(text, pad=26 if subtitle else 14)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.2, color=INK_2, va="bottom")


def footnote(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, fontsize=8, color=MUTED, ha="left", va="top")


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def num(n: float) -> str:
    """Inteiro com separador de milhar brasileiro (1.234)."""
    return f"{int(round(n)):,}".replace(",", ".")
