-- =============================================================================
-- ChurnGuard AI · 004 — Revisão humana explícita
-- Permite registrar que a equipe analisou a previsão e decidiu NÃO agir
-- (ex.: discorda do risco, cliente já em negociação). A decisão humana fica
-- documentada mesmo quando contraria a sugestão do modelo.
-- =============================================================================
ALTER TABLE retention_actions DROP CONSTRAINT retention_actions_tipo_acao_check;
ALTER TABLE retention_actions ADD CONSTRAINT retention_actions_tipo_acao_check CHECK (tipo_acao IN (
    'Contato do Customer Success','Treinamento / reonboarding','Negociação financeira',
    'Oferta de ajuste de plano','Follow-up de suporte','Conversa de renovação',
    'Campanha de engajamento','Revisão humana: sem ação','Outra'));
