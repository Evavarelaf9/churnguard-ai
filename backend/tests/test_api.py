import pandas as pd
import pytest
from sqlalchemy import text

from backend.app.core.config import get_settings

API = "/api/v1"


# --------------------------------------------------------------------------- #
# Sistema e modelo
# --------------------------------------------------------------------------- #
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["model"] == "ok"


def test_model_metrics_and_info(client):
    m = client.get(f"{API}/model/metrics").json()
    assert 0.5 < m["roc_auc"] <= 1 and 0 < m["pr_auc"] <= 1
    cm = m["matriz_confusao"]
    assert sum(cm.values()) > 0
    info = client.get(f"{API}/model/info").json()
    assert "idade" in info["features"]["excluded"] and "regiao" in info["features"]["excluded"]
    assert info["risk_thresholds"]["medio"] < info["risk_thresholds"]["alto"]


def test_openapi_documents_required_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/api/v1/predict", "/api/v1/predict/batch", "/api/v1/customers/{customer_id}/risk",
              "/api/v1/model/metrics", "/health"]:
        assert p in paths


# --------------------------------------------------------------------------- #
# Autenticação
# --------------------------------------------------------------------------- #
def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "api_keys", "segredo123:analista")
    assert client.get(f"{API}/model/metrics").status_code == 401
    assert client.get(f"{API}/model/metrics", headers={"X-API-Key": "errada"}).status_code == 401
    assert client.get(f"{API}/model/metrics", headers={"X-API-Key": "segredo123"}).status_code == 200


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #
def test_predict_returns_probability_and_explanation(client, example_payload):
    r = client.post(f"{API}/predict", json={**example_payload, "customer_id": None})
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["probabilidade"] <= 1
    assert b["nivel_risco"] in {"Baixo", "Médio", "Alto"}
    assert b["fatores_risco"] and b["recomendacao"]["aviso"]
    assert "não é certeza" in b["aviso"].lower()
    assert b["prediction_id"] is None  # sem customer_id → nada é gravado


def test_predict_threshold_is_configurable(client, example_payload):
    low = client.post(f"{API}/predict?threshold=0.05", json=example_payload).json()
    high = client.post(f"{API}/predict?threshold=0.95", json=example_payload).json()
    assert low["probabilidade"] == pytest.approx(high["probabilidade"])
    assert low["acima_do_threshold"] and not high["acima_do_threshold"]
    assert low["threshold"] == 0.05


def test_predict_persists_when_customer_exists(client, example_payload, db):
    before = db.execute(text("SELECT COUNT(*) FROM predictions WHERE origem='api'")).scalar()
    r = client.post(f"{API}/predict", json={**example_payload, "customer_id": "CLI-00042"})
    assert r.json()["prediction_id"]
    after = db.execute(text("SELECT COUNT(*) FROM predictions WHERE origem='api'")).scalar()
    assert after == before + 1


@pytest.mark.parametrize("field,value", [("plano", "Diamante"), ("dias_ativos_m1", 45), ("valor_mensal", -10)])
def test_predict_validates_input(client, example_payload, field, value):
    assert client.post(f"{API}/predict", json={**example_payload, field: value}).status_code == 422


def test_predict_batch(client, example_payload, monkeypatch):
    items = [example_payload, {**example_payload, "acessos_m1": 90, "dias_ativos_m1": 28, "reclamacoes_6m": 0}]
    r = client.post(f"{API}/predict/batch", json={"clientes": items})
    assert r.status_code == 200
    s = r.json()["resumo"]
    assert s["total"] == 2 == s["alto"] + s["medio"] + s["baixo"]
    monkeypatch.setattr(get_settings(), "batch_max_size", 1)
    assert client.post(f"{API}/predict/batch", json={"clientes": items}).status_code == 413


# --------------------------------------------------------------------------- #
# Clientes
# --------------------------------------------------------------------------- #
def test_list_customers_filters_and_sorting(client):
    r = client.get(f"{API}/customers", params={"nivel_risco": "Alto", "tamanho_pagina": 50}).json()
    assert r["total"] > 0
    probs = [i["probabilidade"] for i in r["itens"]]
    assert all(i["nivel_risco"] == "Alto" for i in r["itens"])
    assert probs == sorted(probs, reverse=True)
    r2 = client.get(f"{API}/customers", params={"plano": "Básico", "valor_max": 60}).json()
    assert all(i["plano"] == "Básico" and i["valor_mensal"] <= 60 for i in r2["itens"])


def test_customer_detail_is_logged(client, db):
    r = client.get(f"{API}/customers/CLI-00001")
    assert r.status_code == 200
    d = r.json()
    assert d["perfil"]["customer_id"] == "CLI-00001"
    assert "idade" not in d["perfil"]  # minimização de dados pessoais
    assert d["historico_uso"] and d["historico_pagamentos"]
    n = db.execute(text("SELECT COUNT(*) FROM access_log WHERE customer_id='CLI-00001' "
                        "AND acao='visualizar_cliente'")).scalar()
    assert n >= 1


def test_unknown_customer_returns_404(client):
    assert client.get(f"{API}/customers/CLI-99999").status_code == 404
    assert client.get(f"{API}/customers/CLI-99999/risk").status_code == 404


def test_cancelled_customer_is_not_scored(client, db):
    cid = db.execute(text("SELECT customer_id FROM customers WHERE status='Cancelado' LIMIT 1")).scalar()
    assert client.get(f"{API}/customers/{cid}/risk").status_code == 422


def test_risk_from_db_matches_batch_scoring(client, db):
    """O risco calculado on-line a partir do banco reproduz o job em lote."""
    row = db.execute(text("SELECT customer_id, probabilidade FROM predictions "
                          "WHERE origem='batch' ORDER BY probabilidade DESC LIMIT 1")).one()
    live = client.get(f"{API}/customers/{row.customer_id}/risk").json()
    assert live["probabilidade"] == pytest.approx(float(row.probabilidade), abs=1e-4)


def test_point_in_time_features_match_training_base(client):
    """Atributos montados do banco em uma data passada == base analítica usada no treino (sem vazamento)."""
    from ml.src import config as mlc
    from ml.src.scoring import ChurnScorer

    base = pd.read_parquet(mlc.ANALYTICAL_BASE_FILE)
    snap = base[base.data_referencia == "2026-03-31"].sample(3, random_state=1)
    scorer = ChurnScorer.load()
    for _, row in snap.iterrows():
        expected = float(scorer.predict(row.to_frame().T.infer_objects()).probabilidade.iloc[0])
        got = client.get(f"{API}/customers/{row.customer_id}/risk",
                         params={"data_referencia": "2026-03-31"}).json()["probabilidade"]
        assert got == pytest.approx(expected, abs=1e-6)


def test_simulation_does_not_persist(client, db):
    cid = db.execute(text("SELECT customer_id FROM predictions WHERE origem='batch' AND nivel_risco='Alto' "
                          "ORDER BY probabilidade DESC LIMIT 1")).scalar()
    before = db.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
    r = client.post(f"{API}/customers/{cid}/simulate", json={"aumento_uso_percentual": 60, "remover_atrasos": True})
    assert r.status_code == 200
    b = r.json()
    assert b["simulado"]["probabilidade"] < b["atual"]["probabilidade"]
    assert "hipotética" in b["aviso"]
    assert db.execute(text("SELECT COUNT(*) FROM predictions")).scalar() == before
    assert client.post(f"{API}/customers/{cid}/simulate", json={}).status_code == 422


def test_simulation_plan_change_updates_price(client):
    r = client.post(f"{API}/customers/CLI-00001/simulate", json={"plano": "Empresarial"}).json()
    assert r["alteracoes_aplicadas"] == ["plano Empresarial"]


# --------------------------------------------------------------------------- #
# Ações de retenção
# --------------------------------------------------------------------------- #
def test_retention_action_lifecycle(client):
    r = client.post(f"{API}/customers/CLI-00001/actions",
                    json={"tipo_acao": "Contato do Customer Success", "responsavel": "Teste"})
    assert r.status_code == 201
    aid = r.json()["id"]
    assert r.json()["prediction_id"]  # vinculada à previsão mais recente
    u = client.patch(f"{API}/actions/{aid}", json={"status": "Concluída", "resultado": "Retido"}).json()
    assert u["status"] == "Concluída" and u["resultado"] == "Retido"
    assert any(a["id"] == aid for a in client.get(f"{API}/customers/CLI-00001/actions").json())
    assert client.post(f"{API}/customers/CLI-00001/actions", json={"tipo_acao": "Desconto mágico"}).status_code == 422
    assert client.patch(f"{API}/actions/999999", json={"status": "Cancelada"}).status_code == 404


# --------------------------------------------------------------------------- #
# Dashboard e monitoramento
# --------------------------------------------------------------------------- #
def test_dashboard_summary(client):
    d = client.get(f"{API}/dashboard/summary").json()
    k = d["kpis"]
    assert k["alto_risco"] + k["medio_risco"] + k["baixo_risco"] == k["clientes_monitorados"]
    assert k["receita_mensal_em_risco"] <= k["receita_mensal_total"]
    assert len(d["evolucao_churn"]) == 24
    assert d["previsto_vs_real"] and d["importancia_global"]
    assert all(f["clientes"] > 0 for f in d["principais_fatores"])


def test_outcomes_view_compares_predicted_and_actual(db):
    rows = db.execute(text("SELECT resultado, COUNT(*) FROM v_prediction_outcomes "
                           "WHERE churn_real IS NOT NULL GROUP BY resultado")).all()
    kinds = {r[0] for r in rows}
    assert {"Verdadeiro positivo", "Falso positivo", "Falso negativo", "Verdadeiro negativo"} <= kinds


# --------------------------------------------------------------------------- #
# Threshold configurável e metadados para a interface
# --------------------------------------------------------------------------- #
def test_threshold_changes_risk_levels_in_list_and_dashboard(client):
    default = client.get(f"{API}/customers", params={"nivel_risco": "Alto", "tamanho_pagina": 1}).json()["total"]
    strict = client.get(f"{API}/customers", params={"nivel_risco": "Alto", "tamanho_pagina": 1,
                                                     "threshold": 0.5}).json()
    assert strict["total"] < default and strict["faixas_risco"]["alto"] == 0.5
    d = client.get(f"{API}/dashboard/summary", params={"threshold": 0.5}).json()
    assert d["kpis"]["alto_risco"] == strict["total"]


def test_threshold_tradeoff_curve(client):
    t = client.get(f"{API}/meta/thresholds").json()
    pts = t["pontos"]
    assert t["base_observacoes"] > 0 and len(pts) > 10
    recalls = [p["recall"] for p in pts]
    flagged = [p["clientes_sinalizados_hoje"] for p in pts]
    assert recalls == sorted(recalls, reverse=True)
    assert flagged == sorted(flagged, reverse=True)


def test_filter_options(client):
    f = client.get(f"{API}/meta/filtros").json()
    assert "Empresarial" in f["planos"] and f["datas_referencia"]


# --------------------------------------------------------------------------- #
# Monitoramento
# --------------------------------------------------------------------------- #
def test_monitoring_summary(client):
    m = client.get(f"{API}/monitoring/summary").json()
    assert m["status"] in {"ok", "atenção", "crítico"}
    assert len(m["desempenho_mensal"]) >= 6
    for mes in m["desempenho_mensal"]:
        assert mes["vp"] + mes["fp"] + mes["fn"] + mes["vn"] == mes["previsoes"]
    feats = {d["feature"]: d for d in m["drift_atual"]["atributos"]}
    # Preços subiram (reajuste de 9% em 2026) → drift esperado no valor mensal
    assert feats["valor_mensal"]["status"] == "significativo"
    assert any(a["tipo"] == "calibração" for a in m["alertas"])


def test_monitoring_run_is_stored(client):
    before = len(client.get(f"{API}/monitoring/runs").json())
    r = client.post(f"{API}/monitoring/run").json()
    assert r["execucao"]["id"]
    assert len(client.get(f"{API}/monitoring/runs").json()) == before + 1


# --------------------------------------------------------------------------- #
# Itens reforçados na auditoria de requisitos
# --------------------------------------------------------------------------- #
def test_explanation_has_business_narrative(client, example_payload):
    b = client.post(f"{API}/predict", json=example_payload).json()
    assert b["resumo"].startswith("O fator que mais contribuiu") or b["resumo"].startswith("Nenhum sinal")
    detail = client.get(f"{API}/customers/CLI-00001/risk").json()
    assert detail["resumo"]


def test_medium_threshold_is_configurable(client):
    base = client.get(f"{API}/customers", params={"nivel_risco": "Médio", "tamanho_pagina": 1}).json()
    wider = client.get(f"{API}/customers", params={"nivel_risco": "Médio", "tamanho_pagina": 1,
                                                    "limite_medio": 0.05}).json()
    assert wider["faixas_risco"]["medio"] == 0.05 and wider["total"] > base["total"]
    # o limite médio nunca passa do alto
    clamp = client.get(f"{API}/customers", params={"tamanho_pagina": 1, "limite_medio": 0.6}).json()
    assert clamp["faixas_risco"]["medio"] == clamp["faixas_risco"]["alto"]
    d = client.get(f"{API}/dashboard/summary", params={"limite_medio": 0.05}).json()
    assert d["kpis"]["medio_risco"] == wider["total"]


def test_historical_period_shows_actual_outcome(client):
    datas = client.get(f"{API}/meta/filtros").json()["datas_referencia"]
    hist = [d for d in datas if d < "2026-06-01"]
    assert hist, "datas históricas devem aparecer no filtro de período"
    r = client.get(f"{API}/customers", params={"data_referencia": hist[-1], "nivel_risco": "Alto",
                                               "tamanho_pagina": 100}).json()
    assert r["total"] > 0
    assert all(i["origem"] == "historico" and i["churn_real"] in (0, 1) for i in r["itens"])
    assert any(i["principal_fator"] for i in r["itens"])


def test_simulation_of_support_tickets(client, db):
    cid = db.execute(text("SELECT customer_id FROM predictions WHERE origem='batch' "
                          "AND (entrada->>'tickets_3m')::int >= 2 ORDER BY probabilidade DESC LIMIT 1")).scalar()
    r = client.post(f"{API}/customers/{cid}/simulate", json={"tickets_3m": 0})
    assert r.status_code == 200
    b = r.json()
    assert any("chamados (3 meses)" in a for a in b["alteracoes_aplicadas"])
    assert b["simulado"]["probabilidade"] != b["atual"]["probabilidade"]


def test_human_review_without_action_is_recorded(client):
    r = client.post(f"{API}/customers/CLI-00002/actions",
                    json={"tipo_acao": "Revisão humana: sem ação", "observacoes": "Cliente em negociação comercial"})
    assert r.status_code == 201 and r.json()["tipo_acao"] == "Revisão humana: sem ação"


def test_tradeoff_reports_churns_found(client):
    pts = client.get(f"{API}/meta/thresholds").json()["pontos"]
    for p in pts:
        assert p["churns_identificados"] <= p["churns_totais"]
        assert p["churns_identificados"] <= p["abordados_historico"]
