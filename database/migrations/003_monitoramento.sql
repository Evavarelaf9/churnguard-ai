-- =============================================================================
-- ChurnGuard AI · 003 — Monitoramento do modelo
-- Cada execução do job de monitoramento guarda um retrato completo, permitindo
-- acompanhar a saúde do modelo ao longo do tempo e auditar decisões de re-treino.
-- =============================================================================
CREATE TABLE monitoring_runs (
    id              BIGSERIAL PRIMARY KEY,
    executado_em    TIMESTAMPTZ NOT NULL DEFAULT now(),
    data_referencia DATE        NOT NULL,
    model_version   VARCHAR(20) NOT NULL REFERENCES model_versions(version),
    status          VARCHAR(10) NOT NULL CHECK (status IN ('ok', 'atenção', 'crítico')),
    alertas         JSONB       NOT NULL,
    desempenho      JSONB       NOT NULL,
    drift           JSONB       NOT NULL,
    resumo          JSONB       NOT NULL
);
CREATE INDEX ix_monitoring_runs_data ON monitoring_runs (executado_em DESC);
