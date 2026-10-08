import { act, fireEvent, render, screen } from "@testing-library/react";

import { RiskSettingsProvider, useRiskSettings } from "../context/RiskSettings";
import { ToastProvider, useToast } from "../context/Toast";

function Notifier() {
  const { notify } = useToast();
  return <button onClick={() => notify("Ação registrada: Contato do Customer Success.")}>salvar</button>;
}

function Limits() {
  const { threshold, medio, setLimits } = useRiskSettings();
  return (
    <div>
      <span data-testid="v">{`${threshold ?? "modelo"}|${medio ?? "modelo"}`}</span>
      <button onClick={() => setLimits(0.3, 0.15)}>ajustar</button>
      <button onClick={() => setLimits(null, null)}>padrão</button>
    </div>
  );
}

describe("feedback e configurações", () => {
  it("mostra confirmação acessível após uma ação e some sozinha", () => {
    vi.useFakeTimers();
    render(<ToastProvider><Notifier /></ToastProvider>);
    fireEvent.click(screen.getByText("salvar"));
    expect(screen.getByRole("status")).toHaveTextContent("Ação registrada");
    act(() => { vi.advanceTimersByTime(4100); });
    expect(screen.getByRole("status")).not.toHaveTextContent("Ação registrada");
    vi.useRealTimers();
  });

  it("limites alto e médio são configuráveis e voltam ao padrão do modelo", () => {
    render(<RiskSettingsProvider><Limits /></RiskSettingsProvider>);
    expect(screen.getByTestId("v")).toHaveTextContent("modelo|modelo");
    fireEvent.click(screen.getByText("ajustar"));
    expect(screen.getByTestId("v")).toHaveTextContent("0.3|0.15");
    fireEvent.click(screen.getByText("padrão"));
    expect(screen.getByTestId("v")).toHaveTextContent("modelo|modelo");
  });
});
