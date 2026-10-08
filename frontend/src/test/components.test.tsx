import { render, screen } from "@testing-library/react";

import { ConfusionMatrix } from "../components/charts/ConfusionMatrix";
import { RiskBadge } from "../components/ui/RiskBadge";
import { ErrorState } from "../components/ui/States";

describe("componentes", () => {
  it("nível de risco é comunicado por texto, não só por cor", () => {
    render(<RiskBadge level="Alto" />);
    expect(screen.getByText("Alto risco")).toBeInTheDocument();
  });

  it("matriz de confusão calcula precision e recall", () => {
    render(<ConfusionMatrix m={{ verdadeiro_positivo: 30, falso_positivo: 70, verdadeiro_negativo: 880, falso_negativo: 20 }} />);
    expect(screen.getByText("30,0%")).toBeInTheDocument(); // precision 30/100
    expect(screen.getByText("60,0%")).toBeInTheDocument(); // recall 30/50
  });

  it("estado de erro é anunciado para tecnologias assistivas", () => {
    render(<ErrorState error={new Error("API fora do ar")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("API fora do ar");
  });
});
