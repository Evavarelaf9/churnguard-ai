"""Carga inicial do banco a partir do dataset sintético e do modelo treinado.

Etapas:
  1. aplica migrations pendentes;
  2. carrega as tabelas operacionais dos CSVs (ml/data/raw), com limpeza na ingestão:
       - remove duplicatas exatas de clientes;
       - idade fora do domínio 18–100 → NULL (erro de digitação);
  3. registra a versão ativa do modelo (ml/models/model_metadata.json);
  4. gera as previsões históricas do período de teste com o modelo avaliado (origem 'historico') e
     preenche o resultado real → base do monitoramento "previsto × real";
  5. roda o job de pontuação em lote para os clientes ativos hoje;
  6. cria algumas ações de retenção de demonstração.

Uso:
    python -m backend.app.db.seed           # só carrega se o banco estiver vazio
    python -m backend.app.db.seed --reset   # apaga e recarrega tudo
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.app.db.migrate import migrate
from backend.app.db.session import get_engine, get_sessionmaker
from backend.app.jobs import score_customers, update_outcomes
from backend.app.models.entities import ModelVersion, RetentionAction
from backend.app.repositories.repositories import ModelRepository
from ml.src import config as ml_config

log = logging.getLogger("churnguard.seed")

TABLES = ["monitoring_runs", "access_log", "retention_actions", "predictions", "model_versions", "surveys", "tickets",
          "payments", "usage_monthly", "customers", "price_adjustments", "plans"]


def _copy(engine: Engine, table: str, df: pd.DataFrame) -> int:
    """COPY em lote (muito mais rápido que INSERT linha a linha)."""
    cols = list(df.columns)
    data = df.astype(object).where(pd.notna(df), None)
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
                for row in data.itertuples(index=False, name=None):
                    cp.write_row([None if (isinstance(v, float) and math.isnan(v)) else v for v in row])
        raw.commit()
    finally:
        raw.close()
    return len(df)


def load_operational(engine: Engine) -> dict[str, int]:
    r = ml_config.RAW_DIR
    counts = {}
    plans = pd.read_csv(r / "plans.csv")
    counts["plans"] = _copy(engine, "plans", plans)
    counts["price_adjustments"] = _copy(engine, "price_adjustments", pd.read_csv(r / "price_adjustments.csv"))

    raw_customers = pd.read_csv(r / "customers.csv")
    customers = raw_customers.drop_duplicates()
    invalid_age = customers.idade.notna() & ~customers.idade.between(18, 100)
    customers.loc[invalid_age, "idade"] = np.nan
    customers["idade"] = customers.idade.astype("Int64")
    log.info("clientes: %d duplicatas removidas, %d idades fora do domínio → NULL",
             len(raw_customers) - len(customers), int(invalid_age.sum()))
    counts["customers"] = _copy(engine, "customers", customers)

    usage = pd.read_csv(r / "usage_monthly.csv")
    counts["usage_monthly"] = _copy(engine, "usage_monthly", usage)
    counts["payments"] = _copy(engine, "payments", pd.read_csv(r / "payments.csv"))
    tickets = pd.read_csv(r / "tickets.csv")
    tickets["avaliacao"] = tickets.avaliacao.astype("Int64")
    counts["tickets"] = _copy(engine, "tickets", tickets)
    counts["surveys"] = _copy(engine, "surveys", pd.read_csv(r / "surveys.csv"))
    return counts


def register_model(db) -> ModelVersion:
    meta = json.loads(ml_config.MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    mv = ModelVersion(
        version=meta["model_version"],
        model_name=meta["model_name"],
        trained_at=datetime.fromisoformat(meta["trained_at"]),
        threshold=meta["threshold"]["operacional"],
        risk_thresholds=meta["risk_thresholds"],
        metrics=meta["metrics"],
        metadata_=meta,
    )
    ModelRepository(db).register(mv, activate=True)
    db.commit()
    return mv


def load_historical_predictions(engine: Engine, mv: ModelVersion) -> int:
    """Previsões do período de teste, geradas pelo modelo AVALIADO (treinado só até set/2025).

    São o que o modelo teria dito em cada mês do teste; como o resultado real já é conhecido,
    alimentam o monitoramento "previsto × real". Os atributos usados (entrada) também são
    gravados, permitindo acompanhar o drift mês a mês.
    """
    import joblib

    from ml.src.evaluation.metrics import classify_risk
    from ml.src.explainability.explainer import ChurnExplainer
    from ml.src.features.build_features import RAW_INPUT_COLUMNS
    from ml.src.preprocessing.pipeline import DEFAULT_SPLIT, temporal_train_test_split

    bundle = joblib.load(ml_config.MODELS_DIR / "churn_model_eval.joblib")
    base = pd.read_parquet(ml_config.ANALYTICAL_BASE_FILE)
    _, test = temporal_train_test_split(base, DEFAULT_SPLIT)
    test = test.reset_index(drop=True)
    th = bundle["metadata"]["risk_thresholds"]
    # Explicações do que o modelo teria mostrado em cada mês (permite navegar pelo histórico no painel)
    expl = ChurnExplainer(bundle["pipeline"], risk_thresholds=th).explain(test, top_k=4)
    proba = np.array([e["probabilidade"] for e in expl])
    entrada = test[RAW_INPUT_COLUMNS].astype(object).where(test[RAW_INPUT_COLUMNS].notna(), None)
    df = pd.DataFrame({
        "customer_id": test.customer_id,
        "data_referencia": test.data_referencia.dt.date,
        "model_version": mv.version,
        "probabilidade": np.round(proba, 5),
        "nivel_risco": classify_risk(proba, th),
        "threshold": float(th["alto"]),
        "entrada": [json.dumps(r, ensure_ascii=False, default=float) for r in entrada.to_dict("records")],
        "principal_fator": [e["principal_fator"] for e in expl],
        "categoria_principal": [e["categoria_principal"] for e in expl],
        "fatores": [json.dumps({"risco": e["fatores_risco"], "protecao": e["fatores_protecao"], "resumo": e["resumo"]},
                               ensure_ascii=False) for e in expl],
        "recomendacao": [json.dumps(e["recomendacao"], ensure_ascii=False) for e in expl],
        "origem": "historico",
    })
    return _copy(engine, "predictions", df)


def demo_actions(db) -> int:
    """Ações de retenção de demonstração (claramente marcadas) para o dashboard."""
    rows = db.execute(text("""
        SELECT id, customer_id, categoria_principal FROM predictions
        WHERE origem = 'batch' AND nivel_risco = 'Alto' ORDER BY probabilidade DESC LIMIT 24""")).all()
    tipo = {"uso": "Treinamento / reonboarding", "financeiro": "Negociação financeira",
            "suporte": "Follow-up de suporte", "contrato": "Conversa de renovação",
            "preco": "Oferta de ajuste de plano", "perfil": "Campanha de engajamento"}
    statuses = ["Planejada", "Em andamento", "Concluída", "Planejada", "Em andamento", "Concluída"]
    resultados = {2: "Retido", 5: "Sem resposta"}
    responsaveis = ["Ana (CS)", "Bruno (CS)", "Carla (Suporte)", "Diego (Financeiro)"]
    for i, (pid, cid, cat) in enumerate(rows):
        db.add(RetentionAction(
            customer_id=cid, prediction_id=pid, tipo_acao=tipo.get(cat or "perfil", "Contato do Customer Success"),
            status=statuses[i % 6], responsavel=responsaveis[i % 4],
            resultado=resultados.get(i % 6), observacoes="Registro de demonstração gerado pelo seed.",
        ))
    db.commit()
    return len(rows)


def run(reset: bool = False) -> bool:
    """Executa a carga. Retorna False se o banco já tinha dados e reset=False."""
    t0 = time.time()
    engine = get_engine()
    migrate(engine)
    with engine.begin() as conn:
        has_data = conn.execute(text("SELECT EXISTS (SELECT 1 FROM customers)")).scalar()
        if has_data and not reset:
            print("Banco já possui dados — nada a fazer (use --reset para recarregar).")
            return False
        if has_data:
            conn.execute(text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE"))
            log.info("dados anteriores removidos")

    counts = load_operational(engine)
    for t, n in counts.items():
        log.info("  %-18s %9s linhas", t, f"{n:,}".replace(",", "."))
    with get_sessionmaker()() as db:
        mv = register_model(db)
        log.info("modelo registrado: %s %s (ativo)", mv.model_name, mv.version)
        n_hist = load_historical_predictions(engine, mv)
        n_out = update_outcomes.run(db)
        log.info("previsões históricas: %d (resultado real preenchido em %d)", n_hist, n_out)
        n_batch = score_customers.run(db)
        log.info("previsões atuais (lote): %d clientes ativos", n_batch)
        log.info("ações de retenção de demonstração: %d", demo_actions(db))
    print(f"Seed concluído em {time.time() - t0:.0f}s")
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="apaga os dados existentes antes de carregar")
    run(parser.parse_args().reset)


if __name__ == "__main__":
    main()
