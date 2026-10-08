import { fmtDate, fmtMonth, fmtPct, fmtTenure } from "../lib/format";
import { classifyRisk, effectiveThresholds } from "../lib/risk";

describe("formatação pt-BR", () => {
  it("formata percentuais com vírgula", () => {
    expect(fmtPct(0.0784)).toBe("7,8%");
    expect(fmtPct(null)).toBe("—");
  });
  it("não desloca datas ISO por fuso horário", () => {
    expect(fmtDate("2026-09-30")).toBe("30/09/2026");
    expect(fmtMonth("2026-01-01")).toBe("jan/26");
  });
  it("descreve o tempo de relacionamento", () => {
    expect(fmtTenure(1)).toBe("1 mês");
    expect(fmtTenure(26)).toBe("2 anos e 2 meses");
  });
});

describe("classificação de risco (mesma regra do backend)", () => {
  const th = { medio: 0.12, alto: 0.23 };
  it("usa os limites de médio e alto", () => {
    expect(classifyRisk(0.05, th)).toBe("Baixo");
    expect(classifyRisk(0.12, th)).toBe("Médio");
    expect(classifyRisk(0.23, th)).toBe("Alto");
  });
  it("ajusta o limite médio quando o alto fica abaixo dele", () => {
    expect(effectiveThresholds(th, 0.08)).toEqual({ medio: 0.08, alto: 0.08 });
    expect(effectiveThresholds(th, null)).toEqual(th);
  });
});
