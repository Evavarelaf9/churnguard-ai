"""Gera e executa o notebook 01_eda.ipynb a partir das funções de ml/src/eda.

Manter o notebook como *código gerado* evita divergência entre notebook e
pipeline: a lógica vive em módulos testáveis; o notebook só conta a história.

Uso (na raiz do projeto):
    python ml/notebooks/_build_eda_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).parent
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("""# ChurnGuard AI · 01 — Análise Exploratória de Dados

**Pergunta de negócio:** *quais clientes apresentam maior risco de cancelar e quais fatores contribuem para esse risco?*

Este notebook explora a base analítica (cliente × mês) antes da modelagem. Toda a lógica está em
`ml/src/eda/analysis.py`; aqui fica a narrativa. O relatório completo e versionado está em `docs/eda.md`."""),
    code("""import sys
from pathlib import Path
ROOT = Path.cwd().resolve().parents[1] if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from ml.src import config
from ml.src.data.build_analytical_base import load_raw
from ml.src.eda import analysis as A
from ml.src.viz import style as S

S.apply()
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
%matplotlib inline"""),
    md("## 1. Tabelas brutas e qualidade"),
    code("""raw_files = ["customers", "plans", "usage_monthly", "payments", "tickets", "surveys", "price_adjustments"]
raw_full = {n: pd.read_csv(config.RAW_DIR / f"{n}.csv") for n in raw_files}
A.raw_quality_summary(raw_full)"""),
    code("""raw = load_raw()  # já remove as 25 duplicatas exatas do cadastro
raw["customers"].head()"""),
    md("""## 2. Base analítica

Uma linha por **cliente ativo × data de referência** (fim de mês). O rótulo `churn` indica cancelamento nos
**60 dias seguintes**. Todas as agregações respeitam o que era conhecido na data de referência (*point-in-time*)."""),
    code("""base = pd.read_parquet(config.ANALYTICAL_BASE_FILE)
print(f"{len(base):,} registros × {base.shape[1]} variáveis | {base.customer_id.nunique():,} clientes | churn = {base.churn.mean():.2%}")
A.data_profile(base.drop(columns=["nome"]))"""),
    code("""base.drop(columns=["nome"]).describe().T"""),
    code("""d = A.add_eda_helpers(base)
fig = A.plot_target_distribution(d)"""),
    md("""> **Desbalanceamento:** um modelo que sempre responde "não cancela" acerta ~92% — e não encontra nenhum cliente em risco.
> Por isso a acurácia não será a métrica principal."""),
    code("""fig, monthly = A.plot_monthly_churn(raw["customers"], raw_full["price_adjustments"])"""),
    md("## 3. Churn por dimensão de negócio"),
    code("""A.plot_rate_bars(d, "plano", "Churn por tipo de plano", "Taxa em 60 dias · IC 95%",
                 order=["Básico", "Essencial", "Profissional", "Empresarial"]);
A.plot_rate_bars(d, "faixa_preco", "Churn por faixa de valor mensal", "Valor efetivamente cobrado",
                 order=["até R$ 60", "R$ 60–120", "R$ 120–250", "acima de R$ 250"]);"""),
    code("""A.plot_rate_bars(d, "faixa_tempo", "Churn por tempo de relacionamento", "Onboarding e renovação anual",
                 order=["até 3 meses", "3–6 meses", "6–12 meses", "1–2 anos", "mais de 2 anos"]);
A.plot_rate_bars(d, "faixa_uso", "Churn por frequência de utilização", "Acessos por semana",
                 order=["até 2", "2–5", "5–10", "10–20", "mais de 20"]);"""),
    code("""A.plot_rate_bars(d, "faixa_reclamacoes", "Churn por reclamações (6m)", "",
                 order=["0", "1", "2", "3 ou mais"]);
A.plot_rate_bars(d, "forma_pagamento", "Churn por método de pagamento", "",
                 order=["Cartão de crédito", "Débito automático", "Pix", "Boleto"]);
A.plot_rate_bars(d, "faixa_atrasos", "Churn por atraso de pagamento (6m)", "",
                 order=["0", "1", "2", "3 ou mais"]);"""),
    code("""fig = A.plot_contract_panel(d)"""),
    md("""## 4. Mudança de comportamento

O gráfico mais importante da EDA: **quem cancela reduz o uso nos meses anteriores**. Isso guia a engenharia de atributos."""),
    code("""fig = A.plot_usage_trajectory(d)"""),
    md("## 5. Comparação entre grupos e distribuições"),
    code("""A.class_comparison(d, ["acessos_semana", "variacao_uso_3m", "dias_desde_ultimo_acesso", "funcionalidades_usadas_m1",
                        "faturas_atrasadas_6m", "tickets_3m", "reclamacoes_6m", "ultima_nota_satisfacao",
                        "tempo_cliente_meses", "valor_mensal"])"""),
    code("""fig = A.plot_numeric_distributions(d, {
    "tempo_cliente_meses": "Tempo de cliente (meses)", "acessos_semana": "Acessos por semana",
    "variacao_uso_3m": "Variação de uso", "dias_desde_ultimo_acesso": "Dias desde último acesso",
    "funcionalidades_usadas_m1": "Funcionalidades usadas", "faturas_atrasadas_6m": "Faturas atrasadas (6m)",
    "tickets_6m": "Chamados (6m)", "reclamacoes_6m": "Reclamações (6m)", "ultima_nota_satisfacao": "Nota de satisfação"})"""),
    md("## 6. Correlação, outliers e ausentes"),
    code("""cols = ["tempo_cliente_meses", "valor_mensal", "acessos_m1", "variacao_uso_3m", "minutos_uso_m1",
        "funcionalidades_usadas_m1", "dias_desde_ultimo_acesso", "faturas_atrasadas_6m", "tickets_6m",
        "reclamacoes_6m", "avaliacao_media_6m", "ultima_nota_satisfacao", "churn"]
fig, corr = A.plot_correlation(d, cols)"""),
    code("""out_cols = {"minutos_uso_m1": "Minutos de uso/mês", "acessos_m1": "Acessos/mês",
            "tempo_medio_resolucao_h_6m": "Resolução (h)", "valor_mensal": "Valor mensal"}
display(A.outlier_summary(d, list(out_cols)))
fig = A.plot_outliers(d, out_cols, log_cols=("minutos_uso_m1", "acessos_m1"))"""),
    code("""fig = A.plot_missing(base.drop(columns=["nome"]))"""),
    md("## 7. Checagem de atributos sensíveis"),
    code("""fig = A.plot_fairness_check(d)"""),
    md("""Região e idade **não mostram diferença relevante** de churn (intervalos sobrepostos). Como não trazem sinal e têm
potencial discriminatório, ficam **fora do modelo** e são usadas apenas para auditoria de viés.

## 8. Conclusões para a modelagem

1. Base desbalanceada (~11:1) → PR-AUC, Recall, Precision; balanceamento avaliado só no treino.
2. Dinâmica temporal (picos pós-reajuste) → divisão treino/teste por data, com intervalo de 60 dias.
3. Queda de uso é o sinal mais forte → atributos de tendência e recência.
4. Uso depende do plano → normalizar pela capacidade do plano.
5. Atrito financeiro e de suporte somam risco → taxas de atraso, reclamações por mês.
6. Ausência carrega informação → *flags* de ausência.
7. Colunas pós-evento (`data_cancelamento`, `motivo_cancelamento`, `status`) nunca entram como atributo."""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

out = HERE / "01_eda.ipynb"
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, out)
print(f"Notebook executado e salvo em {out}")
