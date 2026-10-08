// Tokens de gráfico (mesma paleta dos relatórios em Python).
export const C = {
  blue: "#2a78d6",
  blueLight: "#9ec5f4",
  orange: "#eb6834",
  violet: "#4a3aa7",
  aqua: "#1baf7a",
  grid: "#e5e4de",
  axis: "#cfcdc4",
  muted: "#7c7d84",
  ink: "#16171a",
  low: "#0ca30c",
  mid: "#fab219",
  high: "#d03b3b",
};

export const axisProps = {
  tick: { fill: C.muted, fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: C.axis },
} as const;
