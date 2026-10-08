import { useState } from "react";

import { useToast } from "../../context/Toast";
import { useCreateAction } from "../../hooks/useApi";
import { ACTION_TYPES } from "../../lib/types";
import { Modal } from "../ui/Modal";

const SUGGESTED_TYPE: Record<string, (typeof ACTION_TYPES)[number]> = {
  uso: "Treinamento / reonboarding",
  financeiro: "Negociação financeira",
  suporte: "Follow-up de suporte",
  contrato: "Conversa de renovação",
  preco: "Oferta de ajuste de plano",
  perfil: "Campanha de engajamento",
};

export function ActionModal({ open, onClose, customerId, category }: { open: boolean; onClose: () => void; customerId: string; category?: string | null }) {
  const create = useCreateAction(customerId);
  const { notify } = useToast();
  const [tipo, setTipo] = useState<string>(SUGGESTED_TYPE[category ?? ""] ?? "Contato do Customer Success");
  const [resp, setResp] = useState("");
  const [obs, setObs] = useState("");
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    create.mutate({ tipo_acao: tipo, responsavel: resp || undefined, observacoes: obs || undefined }, {
      onSuccess: () => { setObs(""); onClose(); notify(`Ação registrada: ${tipo}.`); },
    });
  };
  return (
    <Modal open={open} onClose={onClose} title="Registrar ação de retenção">
      <form onSubmit={submit} className="space-y-4">
        <p className="text-sm text-ink-2">A ação fica vinculada à previsão atual, para medir depois se funcionou. A decisão é sua — a sugestão do modelo é apenas um ponto de partida.
          Se, após revisar o caso, você decidir <b>não agir</b>, registre “Revisão humana: sem ação” para que a decisão fique documentada.</p>
        <div>
          <label className="label" htmlFor="tipo">Tipo de ação</label>
          <select id="tipo" className="input" value={tipo} onChange={(e) => setTipo(e.target.value)}>
            {ACTION_TYPES.map((t) => <option key={t}>{t}</option>)}
          </select>
        </div>
        <div>
          <label className="label" htmlFor="resp">Responsável</label>
          <input id="resp" className="input" maxLength={80} placeholder="Ex.: Ana (Customer Success)" value={resp} onChange={(e) => setResp(e.target.value)} />
        </div>
        <div>
          <label className="label" htmlFor="obs">Observações</label>
          <textarea id="obs" className="input min-h-24" maxLength={2000} value={obs} onChange={(e) => setObs(e.target.value)} />
        </div>
        {create.isError && <p role="alert" className="text-sm text-risk-high-ink">{(create.error as Error).message}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-secondary" onClick={onClose}>Cancelar</button>
          <button type="submit" className="btn-primary" disabled={create.isPending}>{create.isPending ? "Salvando…" : "Registrar ação"}</button>
        </div>
      </form>
    </Modal>
  );
}
