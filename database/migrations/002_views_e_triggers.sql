-- =============================================================================
-- ChurnGuard AI · 002 — Views analíticas e triggers
-- =============================================================================

-- updated_at automático
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_customers_updated_at BEFORE UPDATE ON customers
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_actions_updated_at BEFORE UPDATE ON retention_actions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Previsão mais recente de cada cliente (a que o dashboard exibe).
CREATE VIEW v_latest_predictions AS
SELECT DISTINCT ON (p.customer_id) p.*
FROM predictions p
WHERE p.origem IN ('batch', 'api')
ORDER BY p.customer_id, p.data_referencia DESC, p.created_at DESC;

-- "O que o modelo previu?" × "O que realmente aconteceu?"
CREATE VIEW v_prediction_outcomes AS
SELECT
    p.id,
    p.customer_id,
    p.data_referencia,
    p.model_version,
    p.probabilidade,
    p.nivel_risco,
    p.threshold,
    (p.probabilidade >= p.threshold)::int AS previsto,
    p.churn_real,
    CASE
        WHEN p.churn_real IS NULL THEN 'Aguardando'
        WHEN p.probabilidade >= p.threshold AND p.churn_real = 1 THEN 'Verdadeiro positivo'
        WHEN p.probabilidade >= p.threshold AND p.churn_real = 0 THEN 'Falso positivo'
        WHEN p.probabilidade <  p.threshold AND p.churn_real = 1 THEN 'Falso negativo'
        ELSE 'Verdadeiro negativo'
    END AS resultado
FROM predictions p;

-- Resumo mensal previsto × real (base do monitoramento).
CREATE VIEW v_monthly_model_performance AS
SELECT
    data_referencia,
    model_version,
    COUNT(*)                                         AS previsoes,
    AVG(probabilidade)                               AS churn_previsto,
    AVG(churn_real)                                  AS churn_real,
    SUM(CASE WHEN resultado = 'Verdadeiro positivo' THEN 1 ELSE 0 END) AS vp,
    SUM(CASE WHEN resultado = 'Falso positivo'      THEN 1 ELSE 0 END) AS fp,
    SUM(CASE WHEN resultado = 'Falso negativo'      THEN 1 ELSE 0 END) AS fn,
    SUM(CASE WHEN resultado = 'Verdadeiro negativo' THEN 1 ELSE 0 END) AS vn
FROM v_prediction_outcomes
WHERE churn_real IS NOT NULL
GROUP BY data_referencia, model_version;
