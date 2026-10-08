-- =============================================================================
-- ChurnGuard AI · 001 — Schema inicial
-- Entidades operacionais (espelham os sistemas de origem) + entidades de ML.
-- Convenção: tabelas em inglês, colunas em português (mesmos nomes do dataset).
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Catálogos
-- ---------------------------------------------------------------------------
CREATE TABLE plans (
    plano_id                    SMALLINT PRIMARY KEY,
    plano                       VARCHAR(40)   NOT NULL UNIQUE,
    preco_base                  NUMERIC(10,2) NOT NULL CHECK (preco_base > 0),
    funcionalidades_disponiveis SMALLINT      NOT NULL CHECK (funcionalidades_disponiveis > 0),
    usuarios_incluidos          SMALLINT      NOT NULL CHECK (usuarios_incluidos > 0)
);

CREATE TABLE price_adjustments (
    ano        SMALLINT PRIMARY KEY,
    percentual NUMERIC(5,4) NOT NULL CHECK (percentual >= 0),
    descricao  VARCHAR(200)
);

-- ---------------------------------------------------------------------------
-- Clientes (dados pessoais mínimos — ver docs/lgpd.md)
-- ---------------------------------------------------------------------------
CREATE TABLE customers (
    customer_id          VARCHAR(12) PRIMARY KEY,          -- identificador pseudonimizado
    nome                 VARCHAR(120) NOT NULL,
    idade                SMALLINT     CHECK (idade BETWEEN 18 AND 100),  -- só auditoria de viés
    regiao               VARCHAR(20)  CHECK (regiao IN ('Norte','Nordeste','Centro-Oeste','Sudeste','Sul')),
    tipo_cliente         VARCHAR(30)  NOT NULL CHECK (tipo_cliente IN ('Pessoa Física','Pequena Empresa','Média Empresa')),
    plano_id             SMALLINT     NOT NULL REFERENCES plans(plano_id),
    ciclo_contrato       VARCHAR(10)  NOT NULL CHECK (ciclo_contrato IN ('Mensal','Anual')),
    forma_pagamento      VARCHAR(30)  NOT NULL CHECK (forma_pagamento IN ('Cartão de crédito','Boleto','Pix','Débito automático')),
    renovacao_automatica BOOLEAN      NOT NULL,
    canal_aquisicao      VARCHAR(30)  NOT NULL,
    data_inicio          DATE         NOT NULL,
    data_cancelamento    DATE         CHECK (data_cancelamento IS NULL OR data_cancelamento >= data_inicio),
    motivo_cancelamento  VARCHAR(60),                      -- registrado APÓS o evento: nunca é atributo do modelo
    status               VARCHAR(12)  NOT NULL DEFAULT 'Ativo' CHECK (status IN ('Ativo','Cancelado')),
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT ck_status_cancelamento CHECK ((status = 'Cancelado') = (data_cancelamento IS NOT NULL))
);
CREATE INDEX ix_customers_plano  ON customers (plano_id);
CREATE INDEX ix_customers_status ON customers (status);

-- ---------------------------------------------------------------------------
-- Comportamento e relacionamento
-- ---------------------------------------------------------------------------
CREATE TABLE usage_monthly (
    customer_id            VARCHAR(12) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    mes_referencia         DATE        NOT NULL CHECK (EXTRACT(DAY FROM mes_referencia) = 1),
    acessos                INTEGER     NOT NULL CHECK (acessos >= 0),
    sessoes                INTEGER     NOT NULL CHECK (sessoes >= 0),
    minutos_uso            INTEGER     NOT NULL CHECK (minutos_uso >= 0),
    funcionalidades_usadas SMALLINT    NOT NULL CHECK (funcionalidades_usadas >= 0),
    dias_ativos            SMALLINT    NOT NULL CHECK (dias_ativos BETWEEN 0 AND 31),
    ultimo_acesso          DATE,
    PRIMARY KEY (customer_id, mes_referencia)
);

CREATE TABLE payments (
    payment_id      VARCHAR(14) PRIMARY KEY,
    customer_id     VARCHAR(12)   NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    mes_referencia  DATE          NOT NULL,
    valor_cobrado   NUMERIC(10,2) NOT NULL CHECK (valor_cobrado >= 0),
    data_vencimento DATE          NOT NULL,
    data_pagamento  DATE,
    status          VARCHAR(20)   NOT NULL CHECK (status IN ('Pago em dia','Pago com atraso','Em aberto')),
    UNIQUE (customer_id, mes_referencia)
);
CREATE INDEX ix_payments_customer_venc ON payments (customer_id, data_vencimento);

CREATE TABLE tickets (
    ticket_id             VARCHAR(12) PRIMARY KEY,
    customer_id           VARCHAR(12)  NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    data_abertura         DATE         NOT NULL,
    categoria             VARCHAR(20)  NOT NULL CHECK (categoria IN ('Técnico','Cobrança','Dúvida','Reclamação')),
    tempo_resolucao_horas NUMERIC(8,1) CHECK (tempo_resolucao_horas >= 0),
    avaliacao             SMALLINT     CHECK (avaliacao BETWEEN 1 AND 5),
    data_resolucao        DATE
);
CREATE INDEX ix_tickets_customer_abertura ON tickets (customer_id, data_abertura);

CREATE TABLE surveys (
    survey_id       BIGSERIAL PRIMARY KEY,
    customer_id     VARCHAR(12) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    data_resposta   DATE        NOT NULL,
    nota_satisfacao SMALLINT    NOT NULL CHECK (nota_satisfacao BETWEEN 0 AND 10)
);
CREATE INDEX ix_surveys_customer_data ON surveys (customer_id, data_resposta);

-- ---------------------------------------------------------------------------
-- Machine Learning
-- ---------------------------------------------------------------------------
CREATE TABLE model_versions (
    version         VARCHAR(20) PRIMARY KEY,
    model_name      VARCHAR(80)  NOT NULL,
    trained_at      TIMESTAMPTZ  NOT NULL,
    threshold       NUMERIC(4,3) NOT NULL CHECK (threshold BETWEEN 0 AND 1),
    risk_thresholds JSONB        NOT NULL,
    metrics         JSONB        NOT NULL,
    metadata        JSONB        NOT NULL,
    is_active       BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
-- No máximo uma versão ativa por vez.
CREATE UNIQUE INDEX ux_model_versions_active ON model_versions (is_active) WHERE is_active;

CREATE TABLE predictions (
    id                  BIGSERIAL PRIMARY KEY,
    customer_id         VARCHAR(12)  NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    data_referencia     DATE         NOT NULL,
    model_version       VARCHAR(20)  NOT NULL REFERENCES model_versions(version),
    probabilidade       NUMERIC(6,5) NOT NULL CHECK (probabilidade BETWEEN 0 AND 1),
    nivel_risco         VARCHAR(10)  NOT NULL CHECK (nivel_risco IN ('Baixo','Médio','Alto')),
    threshold           NUMERIC(4,3) NOT NULL,
    principal_fator     TEXT,
    categoria_principal VARCHAR(20),
    fatores             JSONB,        -- fatores de risco/proteção (SHAP traduzido)
    recomendacao        JSONB,
    entrada             JSONB,        -- atributos usados na previsão (auditoria e monitoramento de drift)
    origem              VARCHAR(12)  NOT NULL CHECK (origem IN ('batch','api','historico')),
    -- Resultado observado depois do horizonte (preenchido pelo job update_outcomes)
    churn_real          SMALLINT     CHECK (churn_real IN (0,1)),
    resultado_em        DATE,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX ix_predictions_customer_ref ON predictions (customer_id, data_referencia DESC, created_at DESC);
CREATE INDEX ix_predictions_ref_risk     ON predictions (data_referencia, nivel_risco);
CREATE INDEX ix_predictions_pending      ON predictions (data_referencia) WHERE churn_real IS NULL;

-- ---------------------------------------------------------------------------
-- Ações de retenção (decisão humana registrada)
-- ---------------------------------------------------------------------------
CREATE TABLE retention_actions (
    id            BIGSERIAL PRIMARY KEY,
    customer_id   VARCHAR(12) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    prediction_id BIGINT      REFERENCES predictions(id) ON DELETE SET NULL,
    tipo_acao     VARCHAR(30) NOT NULL CHECK (tipo_acao IN (
                      'Contato do Customer Success','Treinamento / reonboarding','Negociação financeira',
                      'Oferta de ajuste de plano','Follow-up de suporte','Conversa de renovação',
                      'Campanha de engajamento','Outra')),
    status        VARCHAR(15) NOT NULL DEFAULT 'Planejada'
                  CHECK (status IN ('Planejada','Em andamento','Concluída','Cancelada')),
    responsavel   VARCHAR(80),
    observacoes   TEXT,
    resultado     VARCHAR(20) CHECK (resultado IN ('Retido','Cancelou','Sem resposta')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_actions_customer ON retention_actions (customer_id);
CREATE INDEX ix_actions_status   ON retention_actions (status);

-- ---------------------------------------------------------------------------
-- Registro de acesso a dados de clientes (LGPD — rastreabilidade)
-- ---------------------------------------------------------------------------
CREATE TABLE access_log (
    id          BIGSERIAL PRIMARY KEY,
    usuario     VARCHAR(80) NOT NULL,
    acao        VARCHAR(40) NOT NULL,
    customer_id VARCHAR(12),
    detalhe     JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_access_log_customer ON access_log (customer_id, created_at DESC);
