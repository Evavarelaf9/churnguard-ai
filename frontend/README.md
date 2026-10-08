# ChurnGuard AI — Frontend

React 19 + TypeScript + Vite + Tailwind CSS v4 · TanStack Query · Recharts · lucide-react.

```bash
npm install
cp .env.example .env        # VITE_API_URL e VITE_API_KEY (se a API exigir chave)
npm run dev                 # http://localhost:5173 (proxy /api → http://localhost:8000)
npm run test                # vitest
npm run build               # checagem de tipos + build de produção em dist/
```

## Estrutura

```text
src/
├── main.tsx / App.tsx      # providers (React Query, preferências de risco) e rotas (lazy loading)
├── pages/                  # Landing, Dashboard, Customers, CustomerDetail, ModelPage, Actions, NotFound
├── components/
│   ├── layout/             # AppLayout (sidebar + topbar), ThresholdControl, Logo
│   ├── ui/                 # Card, KpiCard, RiskBadge, ProbabilityBar, InfoTip, Modal, estados (loading/vazio/erro)
│   ├── charts/             # gráficos Recharts, BarList, ProbabilityGauge, ConfusionMatrix
│   └── customer/           # fatores (SHAP), simulação, ações de retenção
├── hooks/useApi.ts         # consultas e mutações (TanStack Query)
├── services/api.ts         # cliente HTTP tipado (X-API-Key, erros padronizados)
├── context/RiskSettings.tsx# threshold configurável (preferência local do usuário)
└── lib/                    # tipos da API, formatação pt-BR, regras de risco
```

## Decisões de interface

- **Risco nunca só por cor:** todo nível tem ícone + texto (acessível para daltônicos e leitores de tela).
- **Probabilidade, não certeza:** textos como "probabilidade estimada de cancelamento em 60 dias" e avisos em
  recomendações e simulações.
- **Threshold configurável:** controle global no topo; o backend reclassifica o risco a partir das probabilidades salvas.
- **Estados completos:** skeletons de carregamento, estados vazios, mensagens de erro com "tentar novamente".
- **Acessibilidade:** navegação por teclado, "pular para o conteúdo", `aria-*` em tooltips, diálogos e filtros,
  `prefers-reduced-motion`, tabelas com `caption`/`scope`.
- **Responsivo:** tabela vira lista de cartões no celular; sidebar vira menu.
- **Paleta única:** mesmos tokens de cor dos gráficos do relatório de ML (`ml/src/viz/style.py`).
