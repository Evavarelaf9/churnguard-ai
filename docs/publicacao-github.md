# Publicando o ChurnGuard AI no GitHub

Guia rápido para transformar a pasta do projeto em um repositório público de portfólio.

## 1. Antes de publicar

- [ ] Apague os arquivos `churnguard-etapa*.zip` da pasta (já estão no `.gitignore`, mas não precisam ficar no disco).
- [ ] Confirme que **não existe** um arquivo `.env` com senhas reais (o `.env` é ignorado pelo Git; o `.env.example` é versionado).
- [ ] Ajuste o nome no `LICENSE` (linha `Copyright (c) 2026 ...`), se quiser o nome completo.
- [ ] Opcional: rode `docker compose up --build` uma vez para conferir o ambiente completo na sua máquina.

## 2. Criar o repositório

```bash
cd CHURNGUARDIA
git init -b main
git add .
git status                      # confira: sem ml/data/*.csv, *.parquet, node_modules, .env, *.zip
git commit -m "ChurnGuard AI: previsão de churn com ML explicável, API, dashboard e monitoramento"
```

Crie um repositório vazio no GitHub (sem README, para não conflitar) chamado `churnguard-ai` e depois:

```bash
git remote add origin https://github.com/<seu-usuario>/churnguard-ai.git
git push -u origin main
```

O que é versionado de propósito: código, migrations, **modelo treinado** (`ml/models`, ~300 KB), relatórios
(`docs/`), figuras, GIF e notebooks executados. O que **não** é versionado: o dataset (regenerado em ~30 s pelos
scripts ou pelo Docker), `node_modules` e segredos.

## 3. Configurar a página do repositório

- **Descrição:** `Previsão de churn com IA explicável: ML (scikit-learn + SHAP), FastAPI, PostgreSQL, React e monitoramento de drift.`
- **Topics:** `machine-learning` `churn-prediction` `explainable-ai` `shap` `scikit-learn` `fastapi` `postgresql`
  `react` `typescript` `tailwindcss` `docker` `mlops` `data-science` `lgpd` `portfolio`
- **Fixar** o repositório no seu perfil (*Customize your pins*).
- A aba **Actions** roda o CI (`.github/workflows/ci.yml`) a cada push: lint, testes de ML, testes da API com
  PostgreSQL, testes e build do frontend e build das imagens Docker.

## 4. Roteiro para apresentar o projeto (2 minutos)

1. **Problema:** empresas de assinatura descobrem o cancelamento tarde demais.
2. **Dados:** histórico mês a mês; o sinal mais forte é a *mudança* de comportamento (o uso cai antes do cancelamento).
3. **Rigor:** divisão temporal com intervalo, agregações *point-in-time*, teste usado uma única vez.
4. **Decisão, não só métrica:** PR-AUC em vez de acurácia; threshold escolhido pelo valor da campanha; probabilidades calibradas.
5. **Explicação:** cada cliente com fatores em linguagem de negócio e sugestão de ação — com decisão humana.
6. **Produto:** API, banco auditável, dashboard, simulação "e se".
7. **Operação:** monitoramento detectou o efeito do reajuste de 2026 e o drift de preços — e mostra por que isso não é crítico.
8. **Responsabilidade:** idade e região fora do modelo, auditoria de viés, LGPD.
