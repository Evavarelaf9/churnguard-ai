// Formatação no padrão brasileiro.
const nf0 = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 });
const brl = new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL" });
const brl0 = new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL", maximumFractionDigits: 0 });

export const fmtInt = (v: number | null | undefined) => (v == null ? "—" : nf0.format(v));

export const fmtPct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v)
    ? "—"
    : `${new Intl.NumberFormat("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v * 100)}%`;

export const fmtMoney = (v: number | null | undefined, cents = true) =>
  v == null ? "—" : (cents ? brl : brl0).format(v);

export const fmtCompactMoney = (v: number) => {
  if (Math.abs(v) >= 1_000_000) return `R$ ${(v / 1_000_000).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} mi`;
  if (Math.abs(v) >= 1_000) return `R$ ${(v / 1_000).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} mil`;
  return fmtMoney(v, false);
};

export const fmtDec = (v: number | null | undefined, digits = 2) =>
  v == null ? "—" : v.toLocaleString("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits });

/** Datas ISO (yyyy-mm-dd) sem conversão de fuso: evita "voltar um dia" no Brasil. */
export const fmtDate = (iso: string | null | undefined) => {
  if (!iso) return "—";
  const [y, m, d] = iso.slice(0, 10).split("-");
  return `${d}/${m}/${y}`;
};

const MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"];
export const fmtMonth = (iso: string) => {
  const [y, m] = iso.slice(0, 7).split("-");
  return `${MONTHS[Number(m) - 1]}/${y.slice(2)}`;
};

export const fmtTenure = (months: number | null | undefined) => {
  if (months == null) return "—";
  if (months < 12) return `${Math.round(months)} ${Math.round(months) === 1 ? "mês" : "meses"}`;
  const y = Math.floor(months / 12);
  const m = Math.round(months % 12);
  return `${y} ${y === 1 ? "ano" : "anos"}${m ? ` e ${m} ${m === 1 ? "mês" : "meses"}` : ""}`;
};
