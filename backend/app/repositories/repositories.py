"""Acesso a dados (camada de repositório): consultas SQL isoladas da regra de negócio."""
from __future__ import annotations

from datetime import date

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from backend.app.models.entities import (
    AccessLog,
    Customer,
    ModelVersion,
    Payment,
    Plan,
    Prediction,
    RetentionAction,
    Survey,
    Ticket,
    UsageMonthly,
)


# --------------------------------------------------------------------------- #
# Clientes
# --------------------------------------------------------------------------- #
class CustomerRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, customer_id: str) -> Customer | None:
        return self.db.get(Customer, customer_id)

    def plans(self) -> list[Plan]:
        return list(self.db.scalars(select(Plan).order_by(Plan.plano_id)))

    def usage_history(self, customer_id: str, months: int = 12) -> list[UsageMonthly]:
        q = (select(UsageMonthly).where(UsageMonthly.customer_id == customer_id)
             .order_by(UsageMonthly.mes_referencia.desc()).limit(months))
        return list(reversed(list(self.db.scalars(q))))

    def payment_history(self, customer_id: str, months: int = 12) -> list[Payment]:
        q = (select(Payment).where(Payment.customer_id == customer_id)
             .order_by(Payment.data_vencimento.desc()).limit(months))
        return list(reversed(list(self.db.scalars(q))))

    def tickets(self, customer_id: str, limit: int = 20) -> list[Ticket]:
        q = (select(Ticket).where(Ticket.customer_id == customer_id)
             .order_by(Ticket.data_abertura.desc()).limit(limit))
        return list(self.db.scalars(q))

    def surveys(self, customer_id: str, limit: int = 8) -> list[Survey]:
        q = (select(Survey).where(Survey.customer_id == customer_id)
             .order_by(Survey.data_resposta.desc()).limit(limit))
        return list(self.db.scalars(q))

    def active_ids(self, ref: date) -> list[str]:
        q = select(Customer.customer_id).where(
            Customer.data_inicio <= ref,
            (Customer.data_cancelamento.is_(None)) | (Customer.data_cancelamento > ref),
        )
        return list(self.db.scalars(q))


# --------------------------------------------------------------------------- #
# Previsões
# --------------------------------------------------------------------------- #
SORTS = {
    "probabilidade": "lp.probabilidade DESC",
    "receita_em_risco": "(lp.probabilidade * (lp.entrada->>'valor_mensal')::numeric) DESC",
    "valor_mensal": "(lp.entrada->>'valor_mensal')::numeric DESC",
    "nome": "c.nome ASC",
}


LEVEL_SQL = ("CASE WHEN lp.probabilidade >= :t_alto THEN 'Alto' "
             "WHEN lp.probabilidade >= :t_medio THEN 'Médio' ELSE 'Baixo' END")


class PredictionRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_many(self, rows: list[dict]) -> list[Prediction]:
        objs = [Prediction(**r) for r in rows]
        self.db.add_all(objs)
        self.db.flush()
        return objs

    def latest_for_customer(self, customer_id: str) -> Prediction | None:
        q = (select(Prediction).where(Prediction.customer_id == customer_id, Prediction.origem.in_(["batch", "api"]))
             .order_by(Prediction.data_referencia.desc(), Prediction.created_at.desc()).limit(1))
        return self.db.scalars(q).first()

    def history_for_customer(self, customer_id: str, limit: int = 24) -> list[Prediction]:
        q = (select(Prediction).where(Prediction.customer_id == customer_id)
             .order_by(Prediction.data_referencia.desc(), Prediction.created_at.desc()).limit(limit))
        return list(reversed(list(self.db.scalars(q))))

    def list_latest(self, *, niveis: list[str] | None, plano: str | None, regiao: str | None,
                    valor_min: float | None, valor_max: float | None, data_referencia: date | None,
                    busca: str | None, ordenar: str, limit: int, offset: int,
                    thresholds: dict) -> tuple[list[dict], int]:
        # O nível de risco é recalculado a partir da probabilidade com os limites vigentes,
        # permitindo que o usuário ajuste o threshold sem reprocessar as previsões.
        params = {"t_alto": thresholds["alto"], "t_medio": thresholds["medio"]}
        if data_referencia:
            # Período histórico: previsões feitas naquela data (inclusive de clientes que depois cancelaram),
            # com o resultado real quando já conhecido.
            source = """(SELECT DISTINCT ON (customer_id) * FROM predictions
                         WHERE data_referencia = :ref ORDER BY customer_id, created_at DESC)"""
            where = ["TRUE"]
            params["ref"] = data_referencia
        else:
            source = "v_latest_predictions"
            where = ["c.status = 'Ativo'"]
        if niveis:
            where.append(f"{LEVEL_SQL} = ANY(:niveis)")
            params["niveis"] = niveis
        if plano:
            where.append("p.plano = :plano")
            params["plano"] = plano
        if regiao:
            where.append("c.regiao = :regiao")
            params["regiao"] = regiao
        if valor_min is not None:
            where.append("(lp.entrada->>'valor_mensal')::numeric >= :vmin")
            params["vmin"] = valor_min
        if valor_max is not None:
            where.append("(lp.entrada->>'valor_mensal')::numeric <= :vmax")
            params["vmax"] = valor_max
        if busca:
            where.append("(c.nome ILIKE :busca OR c.customer_id ILIKE :busca)")
            params["busca"] = f"%{busca}%"
        base = f"""
            FROM {source} lp
            JOIN customers c ON c.customer_id = lp.customer_id
            JOIN plans p ON p.plano_id = c.plano_id
            WHERE {' AND '.join(where)}"""
        total = self.db.execute(text(f"SELECT COUNT(*) {base}"), params).scalar_one()
        rows = self.db.execute(text(f"""
            SELECT lp.id AS prediction_id, c.customer_id, c.nome, p.plano, c.regiao, c.tipo_cliente,
                   c.ciclo_contrato, (lp.entrada->>'valor_mensal')::float AS valor_mensal,
                   (lp.entrada->>'tempo_cliente_meses')::float AS tempo_cliente_meses,
                   lp.probabilidade::float AS probabilidade, {LEVEL_SQL} AS nivel_risco, lp.principal_fator,
                   lp.categoria_principal, lp.recomendacao->>'acao_sugerida' AS acao_sugerida,
                   lp.data_referencia, lp.model_version, lp.origem, lp.churn_real
            {base}
            ORDER BY {SORTS.get(ordenar, SORTS['probabilidade'])}, c.customer_id
            LIMIT :limit OFFSET :offset"""), {**params, "limit": limit, "offset": offset}).mappings().all()
        return [dict(r) for r in rows], total

    def latest_frame_rows(self) -> list[dict]:
        """Todas as previsões mais recentes de clientes ativos (para o dashboard)."""
        rows = self.db.execute(text("""
            SELECT lp.customer_id, p.plano, c.regiao, c.ciclo_contrato, lp.probabilidade::float AS probabilidade,
                   lp.nivel_risco, lp.categoria_principal, lp.fatores, lp.entrada, lp.data_referencia,
                   lp.threshold::float AS threshold
            FROM v_latest_predictions lp
            JOIN customers c ON c.customer_id = lp.customer_id
            JOIN plans p ON p.plano_id = c.plano_id
            WHERE c.status = 'Ativo'""")).mappings().all()
        return [dict(r) for r in rows]

    def count(self) -> int:
        return self.db.execute(select(func.count(Prediction.id))).scalar_one()

    def labeled_outcomes(self) -> list[tuple[float, int]]:
        """Probabilidade e resultado real de todas as previsões já observadas."""
        return [(float(p), int(y)) for p, y in self.db.execute(text(
            "SELECT probabilidade, churn_real FROM predictions WHERE churn_real IS NOT NULL")).all()]

    def reference_dates(self) -> list[date]:
        return list(self.db.scalars(text(
            "SELECT DISTINCT data_referencia FROM predictions ORDER BY 1 DESC")))

    def monthly_performance(self) -> list[dict]:
        rows = self.db.execute(text("SELECT * FROM v_monthly_model_performance ORDER BY data_referencia")).mappings().all()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
class ModelRepository:
    def __init__(self, db: Session):
        self.db = db

    def active(self) -> ModelVersion | None:
        return self.db.scalars(select(ModelVersion).where(ModelVersion.is_active)).first()

    def register(self, mv: ModelVersion, activate: bool = True) -> ModelVersion:
        if activate:
            self.db.execute(text("UPDATE model_versions SET is_active = FALSE WHERE is_active"))
        mv.is_active = activate
        self.db.merge(mv)
        self.db.flush()
        return mv


# --------------------------------------------------------------------------- #
# Ações de retenção
# --------------------------------------------------------------------------- #
class ActionRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, action_id: int) -> RetentionAction | None:
        return self.db.get(RetentionAction, action_id)

    def add(self, action: RetentionAction) -> RetentionAction:
        self.db.add(action)
        self.db.flush()
        self.db.refresh(action)
        return action

    def list(self, customer_id: str | None = None, status: str | None = None, limit: int = 100) -> list[RetentionAction]:
        q = select(RetentionAction).order_by(RetentionAction.created_at.desc(), RetentionAction.id.desc()).limit(limit)
        if customer_id:
            q = q.where(RetentionAction.customer_id == customer_id)
        if status:
            q = q.where(RetentionAction.status == status)
        return list(self.db.scalars(q))


# --------------------------------------------------------------------------- #
# Auditoria
# --------------------------------------------------------------------------- #
class AccessLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def log(self, usuario: str, acao: str, customer_id: str | None = None, detalhe: dict | None = None) -> None:
        self.db.add(AccessLog(usuario=usuario, acao=acao, customer_id=customer_id, detalhe=detalhe))
