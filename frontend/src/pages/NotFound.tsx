import { Link } from "react-router-dom";

import { EmptyState } from "../components/ui/States";

export default function NotFound() {
  return (
    <div className="card">
      <EmptyState title="Página não encontrada" description="O endereço acessado não existe."
        action={<Link to="/dashboard" className="btn-primary mt-2">Ir para a visão geral</Link>} />
    </div>
  );
}
