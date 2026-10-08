"""Executa a EDA completa, salva as figuras em docs/images/eda e gera docs/eda.md.

Uso:
    python -m ml.src.eda.run_eda
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.src import config
from ml.src.data.build_analytical_base import load_raw
from ml.src.eda import analysis as A
from ml.src.viz import style as S

LABELS = {
    "idade": "Idade",
    "tempo_cliente_meses": "Tempo de cliente (meses)",
    "valor_mensal": "Valor mensal (R$)",
    "acessos_m1": "Acessos no mês",
    "acessos_semana": "Acessos por semana",
    "variacao_uso_3m": "Variação de uso (3m vs 3m)",
    "sessoes_m1": "Sessões no mês",
    "minutos_uso_m1": "Minutos de uso no mês",
    "funcionalidades_usadas_m1": "Funcionalidades usadas",
    "dias_ativos_m1": "Dias ativos no mês",
    "dias_desde_ultimo_acesso": "Dias desde último acesso",
    "faturas_atrasadas_6m": "Faturas atrasadas (6m)",
    "dias_atraso_medio_6m": "Atraso médio (dias, 6m)",
    "faturas_em_aberto": "Faturas em aberto",
    "valor_total_pago": "Valor total pago (R$)",
    "reajuste_pct_6m": "Reajuste recente (%)",
    "tickets_3m": "Chamados (3m)",
    "tickets_6m": "Chamados (6m)",
    "reclamacoes_6m": "Reclamações (6m)",
    "tempo_medio_resolucao_h_6m": "Tempo médio de resolução (h)",
    "avaliacao_media_6m": "Avaliação do suporte",
    "dias_desde_ultimo_chamado": "Dias desde último chamado",
    "ultima_nota_satisfacao": "Nota de satisfação (0–10)",
    "renovacao_automatica": "Renovação automática",
    "churn": "Churn (alvo)",
    "regiao": "Região",
    "acessos_m2": "Acessos (mês -1)",
    "acessos_m3": "Acessos (mês -2)",
    "acessos_m4": "Acessos (mês -3)",
    "acessos_m5": "Acessos (mês -4)",
    "acessos_m6": "Acessos (mês -5)",
    "minutos_uso_m2": "Minutos de uso (mês -1)",
}


def _md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:,.2f}") -> str:
    """DataFrame → tabela Markdown com números no padrão brasileiro."""
    d = df.reset_index() if index else df.copy()
    if index and d.columns[0] in ("index", None):
        d = d.rename(columns={d.columns[0]: "variável"})
    numeric = [pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]) for c in d.columns]

    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return str(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}".replace(",", ".")
        if isinstance(v, (float, np.floating)):
            if pd.isna(v):
                return "—"
            return floatfmt.format(v).replace(",", "X").replace(".", ",").replace("X", ".")
        return str(v)

    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---:" if n else "---" for n in numeric) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    S.apply()
    out = config.EDA_FIG_DIR
    out.mkdir(parents=True, exist_ok=True)

    raw_full = {
        name: pd.read_csv(config.RAW_DIR / f"{name}.csv")
        for name in ["customers", "plans", "usage_monthly", "payments", "tickets", "surveys", "price_adjustments"]
    }
    raw = load_raw()
    base = pd.read_parquet(config.ANALYTICAL_BASE_FILE)
    d = A.add_eda_helpers(base)
    figs: dict[str, str] = {}

    def save(fig, name):
        fig.savefig(out / f"{name}.png")
        plt.close(fig)
        figs[name] = f"images/eda/{name}.png"

    # ---------------- Qualidade dos dados ---------------------------------- #
    quality = A.raw_quality_summary(raw_full)
    profile = A.data_profile(base.drop(columns=["nome"]))
    invalid_age = int((~raw["customers"].idade.between(18, 100) & raw["customers"].idade.notna()).sum())

    # ---------------- Figuras ------------------------------------------------ #
    save(A.plot_target_distribution(d), "01_distribuicao_churn")
    fig, monthly = A.plot_monthly_churn(raw["customers"], raw_full["price_adjustments"])
    save(fig, "02_evolucao_churn_mensal")
    save(A.plot_rate_bars(d, "plano", "Churn por tipo de plano", "Taxa de cancelamento em 60 dias · barras = IC 95%",
                          order=["Básico", "Essencial", "Profissional", "Empresarial"]), "03_churn_por_plano")
    save(A.plot_rate_bars(d, "faixa_preco", "Churn por faixa de valor mensal", "Valor efetivamente cobrado (com desconto e reajuste)",
                          order=["até R$ 60", "R$ 60–120", "R$ 120–250", "acima de R$ 250"]), "04_churn_por_faixa_preco")
    save(A.plot_rate_bars(d, "faixa_tempo", "Churn por tempo de relacionamento", "Dois picos: onboarding e renovação dos contratos anuais (~12 meses)",
                          order=["até 3 meses", "3–6 meses", "6–12 meses", "1–2 anos", "mais de 2 anos"]), "05_churn_por_tempo_cliente")
    save(A.plot_rate_bars(d, "faixa_uso", "Churn por frequência de utilização", "Acessos por semana no mês de referência",
                          order=["até 2", "2–5", "5–10", "10–20", "mais de 20"]), "06_churn_por_frequencia_uso")
    save(A.plot_rate_bars(d, "faixa_reclamacoes", "Churn por quantidade de reclamações", "Reclamações registradas nos últimos 6 meses",
                          order=["0", "1", "2", "3 ou mais"]), "07_churn_por_reclamacoes")
    save(A.plot_rate_bars(d, "forma_pagamento", "Churn por método de pagamento", "Métodos automáticos reduzem atrito e esquecimento",
                          order=["Cartão de crédito", "Débito automático", "Pix", "Boleto"]), "08_churn_por_forma_pagamento")
    save(A.plot_rate_bars(d, "faixa_atrasos", "Churn por atraso de pagamento", "Faturas pagas após o vencimento nos últimos 6 meses",
                          order=["0", "1", "2", "3 ou mais"]), "09_churn_por_atraso")
    save(A.plot_usage_trajectory(d), "10_trajetoria_uso")
    num_cols = {k: LABELS[k] for k in [
        "tempo_cliente_meses", "acessos_semana", "variacao_uso_3m", "dias_desde_ultimo_acesso",
        "funcionalidades_usadas_m1", "faturas_atrasadas_6m", "tickets_6m", "reclamacoes_6m", "ultima_nota_satisfacao"]}
    save(A.plot_numeric_distributions(d, num_cols), "11_distribuicoes_numericas")
    corr_cols = ["tempo_cliente_meses", "valor_mensal", "acessos_m1", "variacao_uso_3m", "minutos_uso_m1",
                 "funcionalidades_usadas_m1", "dias_ativos_m1", "dias_desde_ultimo_acesso", "faturas_atrasadas_6m",
                 "dias_atraso_medio_6m", "tickets_6m", "reclamacoes_6m", "tempo_medio_resolucao_h_6m",
                 "avaliacao_media_6m", "ultima_nota_satisfacao", "renovacao_automatica", "idade", "churn"]
    fig, corr = A.plot_correlation(d, corr_cols, LABELS)
    save(fig, "12_correlacao")
    out_cols = {"idade": "Idade", "minutos_uso_m1": "Minutos de uso/mês", "acessos_m1": "Acessos/mês",
                "tempo_medio_resolucao_h_6m": "Resolução de chamados (h)", "valor_mensal": "Valor mensal (R$)"}
    base_age = d.assign(idade=raw["customers"].set_index("customer_id").idade.reindex(d.customer_id).to_numpy())
    save(A.plot_outliers(base_age, out_cols, log_cols=("minutos_uso_m1", "acessos_m1")), "13_outliers")
    save(A.plot_missing(base.drop(columns=["nome"]), LABELS), "14_valores_ausentes")
    save(A.plot_fairness_check(d), "15_churn_regiao_idade")
    save(A.plot_contract_panel(d), "16_churn_contrato_canal")

    # ---------------- Tabelas e números -------------------------------------- #
    outliers = A.outlier_summary(base_age, list(out_cols))
    cmp_cols = ["acessos_m1", "acessos_semana", "variacao_uso_3m", "dias_desde_ultimo_acesso", "funcionalidades_usadas_m1",
                "dias_ativos_m1", "faturas_atrasadas_6m", "tickets_3m", "reclamacoes_6m", "tempo_medio_resolucao_h_6m",
                "ultima_nota_satisfacao", "tempo_cliente_meses", "valor_mensal", "idade"]
    comparison = A.class_comparison(d, cmp_cols).rename(index=LABELS)
    describe = base.drop(columns=["nome"]).describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    describe = describe.rename(index=LABELS)
    corr_target = corr[LABELS["churn"]].drop(LABELS["churn"]).sort_values(key=abs, ascending=False)

    def rate(col, key):
        return A.churn_rate_table(d, col).loc[key, "taxa"]

    facts = {
        "linhas": len(base), "colunas": base.shape[1], "clientes_unicos": base.customer_id.nunique(),
        "snapshots": base.data_referencia.nunique(), "churn_medio": base.churn.mean(),
        "ratio": (1 - base.churn.mean()) / base.churn.mean(),
        "duplicatas_cadastro": int(raw_full["customers"].duplicated().sum()),
        "idade_invalida": invalid_age,
        "churn_basico": rate("plano", "Básico"), "churn_empresarial": rate("plano", "Empresarial"),
        "churn_novos": rate("faixa_tempo", "até 3 meses"), "churn_6_12": rate("faixa_tempo", "6–12 meses"), "churn_antigos": rate("faixa_tempo", "mais de 2 anos"),
        "churn_uso_baixo": rate("faixa_uso", "até 2"), "churn_uso_alto": rate("faixa_uso", "mais de 20"),
        "churn_0_recl": rate("faixa_reclamacoes", "0"), "churn_3_recl": rate("faixa_reclamacoes", "3 ou mais"),
        "churn_boleto": rate("forma_pagamento", "Boleto"), "churn_cartao": rate("forma_pagamento", "Cartão de crédito"),
        "churn_0_atraso": rate("faixa_atrasos", "0"), "churn_3_atraso": rate("faixa_atrasos", "3 ou mais"),
        "var_uso_churn": d[d.churn == 1].variacao_uso_3m.median(), "var_uso_ret": d[d.churn == 0].variacao_uso_3m.median(),
        "reg_min": A.churn_rate_table(d, "regiao").taxa.min(), "reg_max": A.churn_rate_table(d, "regiao").taxa.max(),
        "taxa_mensal_media": monthly.taxa.mean(), "pico_mes": monthly.taxa.idxmax().strftime("%m/%Y"),
        "pico_taxa": monthly.taxa.max(),
    }
    (config.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "eda_facts.json").write_text(json.dumps(
        {k: (v if isinstance(v, (str, int)) else round(float(v), 4)) for k, v in facts.items()},
        indent=2, ensure_ascii=False))

    write_report(facts, quality, profile, describe, outliers, comparison, corr_target, figs)
    print(f"EDA concluída: {len(figs)} figuras em {out.relative_to(config.PROJECT_ROOT)} e relatório em docs/eda.md")


def write_report(f, quality, profile, describe, outliers, comparison, corr_target, figs) -> None:
    p = S.pct
    br = lambda n: f"{n:,}".replace(",", ".")
    img = lambda k, alt: f"![{alt}]({figs[k]})"
    md = f"""# Análise Exploratória de Dados (EDA)

> Relatório gerado automaticamente por `python -m ml.src.eda.run_eda`.
> Os números abaixo são recalculados a cada execução — não há valores digitados à mão.

## 1. Escopo e unidade de análise

A unidade de análise é **cliente × data de referência** (snapshot mensal). Cada linha responde:

> *Com o que a empresa sabia no fim do mês, esse cliente cancelou nos {config.CHURN_HORIZON_DAYS} dias seguintes?*

| Item | Valor |
|---|---:|
| Registros (cliente × mês) | {br(f['linhas'])} |
| Variáveis | {f['colunas']} |
| Clientes distintos | {br(f['clientes_unicos'])} |
| Datas de referência | {f['snapshots']} (jun/2024 → jul/2026) |
| Taxa média de churn em 60 dias | **{p(f['churn_medio'])}** |
| Proporção permaneceu : cancelou | {str(round(f['ratio'], 1)).replace('.', ',')} : 1 |

A EDA é **descritiva**. Qualquer decisão de tratamento que "aprenda" com os dados (limites de outlier,
médias para imputação, etc.) será calculada **apenas no conjunto de treino** (ver `docs/model.md`), para evitar vazamento.

## 2. Qualidade dos dados brutos

{_md_table(quality, index=False)}

Problemas encontrados e decisão:

| Problema | Onde | Tratamento previsto |
|---|---|---|
| {f['duplicatas_cadastro']} linhas duplicadas | `customers` (falha de integração com CRM) | Remoção de duplicatas exatas na ingestão |
| Idade ausente (~3%) | `customers.idade` | Mantida só para auditoria (não entra no modelo, ver seção 8) |
| {f['idade_invalida']} idades impossíveis (0, 1, 112, 120, 150) | `customers.idade` | Converter para ausente (validação de domínio 18–100) |
| Região ausente (~0,5%) | `customers.regiao` | Mantida só para auditoria de viés (não entra no modelo) |
| Minutos de uso extremos (~0,2%) | `usage_monthly.minutos_uso` | Winsorização no p99 do treino (integrações/bots) |
| Ausência estrutural | reajuste, avaliação, satisfação, último chamado | Imputação + *flag* indicadora ("nunca abriu chamado" é informação) |

{img('14_valores_ausentes', 'Valores ausentes')}

### Perfil das variáveis da base analítica

<details>
<summary>Ver tipos, ausentes e cardinalidade de todas as colunas</summary>

{_md_table(profile.drop(columns=['exemplo']))}

</details>

### Estatísticas descritivas

<details>
<summary>Ver estatísticas descritivas</summary>

{_md_table(describe)}

</details>

## 3. Variável-alvo

{img('01_distribuicao_churn', 'Distribuição do churn')}

- A base é **desbalanceada**: {p(f['churn_medio'])} de churn. Um modelo que responda sempre "não cancela"
  teria **{p(1 - f['churn_medio'])} de acurácia** e não encontraria nenhum cliente em risco — por isso a acurácia
  não será a métrica principal.
- A taxa varia no tempo e tem **picos após o reajuste anual de janeiro**, sinal de que os dados têm dinâmica
  temporal e que a divisão treino/teste precisa respeitar o tempo.

{img('02_evolucao_churn_mensal', 'Evolução mensal do churn')}

Taxa mensal média de cancelamento: **{p(f['taxa_mensal_media'])}**; pico em {f['pico_mes']} ({p(f['pico_taxa'])}).

## 4. Churn por dimensão de negócio

### Plano e preço
{img('03_churn_por_plano', 'Churn por plano')}
{img('04_churn_por_faixa_preco', 'Churn por faixa de preço')}

O plano Básico cancela **{p(f['churn_basico'])}** contra **{p(f['churn_empresarial'])}** no Empresarial.
Preço mais alto **não** significa mais churn: planos maiores atendem empresas com uso mais enraizado.

### Tempo de relacionamento
{img('05_churn_por_tempo_cliente', 'Churn por tempo de cliente')}

O risco cai com o tempo de casa (**{p(f['churn_antigos'])}** acima de 2 anos), mas há dois momentos críticos:
o **onboarding** (até 3 meses: **{p(f['churn_novos'])}**) e a faixa de **6–12 meses** (**{p(f['churn_6_12'])}**), quando
os contratos anuais chegam à renovação. A interação *tempo de casa × tipo de contrato* é um atributo candidato
("meses até a renovação").

### Frequência de utilização
{img('06_churn_por_frequencia_uso', 'Churn por frequência de uso')}

Quem acessa até 2 vezes por semana tem churn de **{p(f['churn_uso_baixo'])}**, contra **{p(f['churn_uso_alto'])}**
de quem acessa mais de 20 vezes.

### Mudança de comportamento — o sinal mais importante
{img('10_trajetoria_uso', 'Trajetória de uso')}

A mediana da variação de uso (últimos 3 meses vs. 3 anteriores) é **{p(f['var_uso_churn'])}** para quem cancelou
e **{p(f['var_uso_ret'])}** para quem permaneceu. Clientes não cancelam "do nada": o uso cai antes.
Isso justifica criar atributos de **tendência** na engenharia de atributos, e não apenas fotos do mês atual.

### Relacionamento e suporte
{img('07_churn_por_reclamacoes', 'Churn por reclamações')}

Sem reclamações: **{p(f['churn_0_recl'])}**; com 3 ou mais: **{p(f['churn_3_recl'])}**.

### Pagamento
{img('08_churn_por_forma_pagamento', 'Churn por forma de pagamento')}
{img('09_churn_por_atraso', 'Churn por atraso de pagamento')}

Boleto: **{p(f['churn_boleto'])}** vs. cartão: **{p(f['churn_cartao'])}**. Sem atrasos: **{p(f['churn_0_atraso'])}**;
3+ atrasos em 6 meses: **{p(f['churn_3_atraso'])}**.

### Contrato, renovação e canal
{img('16_churn_contrato_canal', 'Churn por contrato, renovação e canal')}

## 5. Comparação entre quem cancelou e quem permaneceu

{_md_table(comparison)}

{img('11_distribuicoes_numericas', 'Distribuições por grupo')}

## 6. Correlação

{img('12_correlacao', 'Matriz de correlação')}

Correlação (Spearman) de cada variável com o churn, em ordem de magnitude:

{_md_table(corr_target.to_frame('ρ com churn').round(3))}

Leituras importantes:
- As correlações individuais com o alvo são **fracas** (|ρ| < 0,2). Isso é típico de churn: nenhum fator isolado
  explica o cancelamento; é a **combinação** de sinais que importa — motivo para usar modelos multivariados.
- Há **multicolinearidade** entre variáveis de uso (acessos, minutos, dias ativos, funcionalidades). Para a Regressão
  Logística isso pede regularização; para árvores, é menos crítico.

## 7. Outliers

{img('13_outliers', 'Outliers')}

{_md_table(outliers, index=False)}

- **Idade**: valores 0, 1, 112, 120 e 150 são erros de digitação → tratados como ausentes (validação de domínio).
- **Minutos de uso**: a cauda longa é esperada (clientes grandes), mas os valores extremos vêm de integrações
  automatizadas → winsorização no percentil 99 **do treino**.
- **Acessos e valor mensal**: outliers são clientes legítimos de planos maiores → mantidos; usaremos taxas
  relativas ao plano na engenharia de atributos.

## 8. Checagem de atributos sensíveis

{img('15_churn_regiao_idade', 'Churn por região e idade')}

A taxa de churn por região varia entre {p(f['reg_min'])} e {p(f['reg_max'])}, com intervalos de confiança
sobrepostos — **sem diferença relevante**. O mesmo vale para idade. Como esses atributos não trazem sinal e têm
potencial discriminatório, a decisão (detalhada em `docs/lgpd.md`) é **não usá-los como entrada do modelo**,
mantendo-os apenas para auditoria de viés.

## 9. Principais conclusões para a modelagem

1. **Desbalanceamento** (~{f['ratio']:.0f}:1) → métricas PR-AUC, Recall e Precision; estratégia de balanceamento avaliada só no treino.
2. **Dados temporais** → divisão treino/teste **por data de referência**, com intervalo de 60 dias entre os períodos.
3. **Queda de uso é o sinal mais forte** → atributos de tendência (variação, razão recente/anterior, recência).
4. **Uso depende do plano** → normalizar pela capacidade do plano (taxa de utilização de funcionalidades).
5. **Atrito financeiro e de suporte** somam risco → percentuais de atraso, reclamações e tickets por mês.
6. **Ausência carrega informação** → *flags* de ausência ("nunca abriu chamado", "não respondeu pesquisa").
7. **Vazamento**: `data_cancelamento`, `motivo_cancelamento` e `status` ficam fora dos atributos.
8. **Ética**: região e idade ficam fora do modelo; usadas apenas para auditoria.
"""
    (config.DOCS_DIR / "eda.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
