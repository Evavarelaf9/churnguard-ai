"""Treinamento, comparação e seleção do modelo de churn.

Etapas (todas reprodutíveis, semente fixa):
  1. Divisão temporal treino/teste com intervalo de 60 dias.
  2. Modelos iniciais (hiperparâmetros padrão) — validação cruzada temporal.
  3. Estudo de desbalanceamento (nenhuma, class_weight, over, under, SMOTE).
  4. Ajuste de hiperparâmetros (RandomizedSearchCV com CV temporal, só no treino).
  5. Seleção do modelo pela PR-AUC média da validação cruzada.
  6. Escolha do threshold e das faixas de risco com previsões *out-of-fold* do treino.
  7. Avaliação final — UMA vez — no conjunto de teste.
  8. Explicabilidade global (permutation importance + SHAP).
  9. Re-treino do modelo escolhido com todos os dados rotulados (modelo de produção)
     e pontuação dos clientes ativos hoje.

Uso:
    python -m ml.src.training.train            # completo (~10 min em 2 CPUs)
    python -m ml.src.training.train --fast     # busca reduzida, para desenvolvimento
    python -m ml.src.training.train --reuse    # reaproveita a busca anterior (etapas 2–4)
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV

from ml.src import config
from ml.src.evaluation.metrics import (
    choose_threshold,
    classification_metrics,
    classify_risk,
    expected_value,
    lift_table,
    threshold_table,
)
from ml.src.explainability.explainer import ChurnExplainer
from ml.src.features.build_features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_LABELS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RAW_INPUT_COLUMNS,
    add_features,
)
from ml.src.preprocessing.pipeline import (
    DEFAULT_SPLIT,
    TemporalGroupSplit,
    build_pipeline,
    temporal_train_test_split,
)
from ml.src.training.models import model_specs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_VERSION = "1.0.0"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def cv_evaluate(pipeline, df: pd.DataFrame, cv: TemporalGroupSplit, threshold: float = 0.5):
    """Métricas médias da validação cruzada temporal + previsões out-of-fold."""
    rows, oof = [], []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(df, groups=df["data_referencia"])):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        p = clone(pipeline).fit(tr, tr["churn"]).predict_proba(va)[:, 1]
        m = classification_metrics(va["churn"], p, threshold)
        m["prob_media"] = float(p.mean())
        rows.append(m)
        oof.append(pd.DataFrame({"idx": va.index, "fold": fold, "proba": p}))
    folds = pd.DataFrame(rows)
    summary = folds[["roc_auc", "pr_auc", "precision", "recall", "f1", "brier", "prob_media", "base_rate"]].mean().to_dict()
    summary["pr_auc_std"] = float(folds.pr_auc.std())
    summary["roc_auc_std"] = float(folds.roc_auc.std())
    summary["pr_auc_folds"] = [round(float(v), 4) for v in folds.pr_auc]
    return summary, folds, pd.concat(oof, ignore_index=True)


def _search(spec, train: pd.DataFrame, cv_splits, fast: bool) -> dict:
    """RandomizedSearchCV com as dobras temporais; otimiza PR-AUC (average precision)."""
    search = RandomizedSearchCV(
        build_pipeline(clone(spec.estimator), spec.scale),
        spec.search_space,
        n_iter=max(3, spec.n_iter // 3) if fast else spec.n_iter,
        scoring="average_precision",
        cv=cv_splits,
        random_state=config.RANDOM_SEED,
        n_jobs=1,
        refit=False,
    )
    search.fit(train, train["churn"])
    return search.best_params_


def main(fast: bool = False, reuse: bool = False) -> dict:
    t0 = time.time()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(config.ANALYTICAL_BASE_FILE)
    train, test = temporal_train_test_split(base, DEFAULT_SPLIT)
    cv = TemporalGroupSplit(n_splits=3, val_size=3, gap=2)
    cv_splits = list(cv.split(train, groups=train["data_referencia"]))
    log(DEFAULT_SPLIT.describe())
    log(f"Treino: {len(train):,} linhas (churn {train.churn.mean():.2%}) | Teste: {len(test):,} linhas (churn {test.churn.mean():.2%})")

    results: dict = {
        "split": {
            "train_start": str(train.data_referencia.min().date()), "train_end": str(train.data_referencia.max().date()),
            "test_start": str(test.data_referencia.min().date()), "test_end": str(test.data_referencia.max().date()),
            "train_rows": len(train), "test_rows": len(test),
            "train_churn_rate": float(train.churn.mean()), "test_churn_rate": float(test.churn.mean()),
            "cv_folds": [
                {"treino_ate": str(train.data_referencia.iloc[a].max().date()),
                 "validacao": f"{train.data_referencia.iloc[v].min().date()} → {train.data_referencia.iloc[v].max().date()}"}
                for a, v in cv_splits
            ],
        }
    }
    specs = model_specs()

    cache_file = config.REPORTS_DIR / "search_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if (reuse and cache_file.exists()) else None

    # ------------------------------------------------------------------ #
    # 2. Modelos iniciais
    # ------------------------------------------------------------------ #
    log("Modelos iniciais (hiperparâmetros padrão)..." + (" [cache]" if cache else ""))
    baseline = cache["baseline"] if cache else {}
    for s in ([] if cache else specs):
        summary, _, _ = cv_evaluate(build_pipeline(clone(s.estimator), s.scale), train, cv)
        fitted = build_pipeline(clone(s.estimator), s.scale).fit(train, train.churn)
        test_m = classification_metrics(test.churn, fitted.predict_proba(test)[:, 1])
        baseline[s.key] = {"name": s.name, "cv": summary, "test": test_m}
        log(f"  {s.name:<42} CV PR-AUC={summary['pr_auc']:.3f}  ROC-AUC={summary['roc_auc']:.3f}")
    results["baseline"] = baseline

    # ------------------------------------------------------------------ #
    # 3. Desbalanceamento
    # ------------------------------------------------------------------ #
    log("Estudo de desbalanceamento...")

    imbalance = cache["imbalance"] if cache else []
    for s in ([] if cache else specs):
        if s.key == "random_forest" and fast:
            continue
        variants = {
            "Nenhuma": (clone(s.estimator), None),
            "class_weight='balanced'": (clone(s.estimator).set_params(class_weight="balanced"), None),
            "Random oversampling": (clone(s.estimator), RandomOverSampler(random_state=config.RANDOM_SEED)),
            "Random undersampling": (clone(s.estimator), RandomUnderSampler(random_state=config.RANDOM_SEED)),
            "SMOTE": (clone(s.estimator), SMOTE(random_state=config.RANDOM_SEED)),
        }
        for strat, (est, sampler) in variants.items():
            summary, _, _ = cv_evaluate(build_pipeline(est, s.scale, sampler), train, cv)
            imbalance.append({"modelo": s.name, "estrategia": strat, **summary})
            log(f"  {s.name[:22]:<22} {strat:<26} PR-AUC={summary['pr_auc']:.3f} recall@0.5={summary['recall']:.2f} "
                f"prob.média={summary['prob_media']:.2f} Brier={summary['brier']:.3f}")
    results["imbalance"] = imbalance

    # ------------------------------------------------------------------ #
    # 4. Ajuste de hiperparâmetros
    # ------------------------------------------------------------------ #
    log("Ajuste de hiperparâmetros (RandomizedSearchCV, CV temporal)...")
    tuned = {}
    for s in specs:
        if cache:
            best_params = {f"model__{k}": v for k, v in cache["best_params"][s.key].items()}
        else:
            best_params = _search(s, train, cv_splits, fast)
        pipe = build_pipeline(clone(s.estimator), s.scale).set_params(**best_params)
        summary, folds, oof = cv_evaluate(pipe, train, cv)
        fitted = clone(pipe).fit(train, train.churn)
        test_m = classification_metrics(test.churn, fitted.predict_proba(test)[:, 1])
        tuned[s.key] = {
            "name": s.name, "why": s.why,
            "best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()},
            "cv": summary, "cv_folds": folds[["roc_auc", "pr_auc", "brier"]].to_dict("records"), "test": test_m,
            "_pipe": pipe, "_fitted": fitted, "_oof": oof,
        }
        log(f"  {s.name:<42} CV PR-AUC {baseline[s.key]['cv']['pr_auc']:.3f} → {summary['pr_auc']:.3f} | params={tuned[s.key]['best_params']}")
    cache_file.write_text(json.dumps({"baseline": baseline, "imbalance": imbalance,
                                      "best_params": {k: v["best_params"] for k, v in tuned.items()}},
                                     default=lambda o: o.item() if hasattr(o, "item") else str(o), ensure_ascii=False), encoding="utf-8")

    pd.DataFrame({"churn": test.churn.to_numpy(), "data_referencia": test.data_referencia.to_numpy(),
                  **{k: v["_fitted"].predict_proba(test)[:, 1] for k, v in tuned.items()}}
                 ).to_parquet(config.REPORTS_DIR / "test_scores_models.parquet", index=False)

    # ------------------------------------------------------------------ #
    # 5. Seleção
    # ------------------------------------------------------------------ #
    best_key = max(tuned, key=lambda k: (round(tuned[k]["cv"]["pr_auc"], 3), -tuned[k]["cv"]["brier"]))
    best = tuned[best_key]
    log(f"Modelo escolhido: {best['name']} (maior PR-AUC média na validação cruzada)")

    # 5b. A estratégia de balanceamento ainda ajuda depois do ajuste? (só no treino, via CV)
    spec = next(s for s in specs if s.key == best_key)
    params = {k: v for k, v in best["_pipe"].get_params().items() if k.startswith("model__")}
    candidates = {
        "Nenhuma": build_pipeline(clone(spec.estimator), spec.scale).set_params(**params),
        "class_weight='balanced'": build_pipeline(clone(spec.estimator), spec.scale).set_params(**{**params, "model__class_weight": "balanced"}),
        "SMOTE": build_pipeline(clone(spec.estimator), spec.scale, SMOTE(random_state=config.RANDOM_SEED)).set_params(**params),
    }
    imbalance_tuned = []
    for strat, pipe in candidates.items():
        summary = best["cv"] if strat == "Nenhuma" else cv_evaluate(pipe, train, cv)[0]
        imbalance_tuned.append({"estrategia": strat, **summary})
        log(f"  {best['name'][:22]} ajustado + {strat:<24} PR-AUC={summary['pr_auc']:.3f} Brier={summary['brier']:.4f} prob.média={summary['prob_media']:.3f}")
    ref = imbalance_tuned[0]
    winner = max(imbalance_tuned, key=lambda r: r["pr_auc"])
    # Só troca se o ganho superar a variação entre dobras E sem piorar a calibração.
    use_strategy = winner["estrategia"] if (winner["pr_auc"] - ref["pr_auc"] > ref["pr_auc_std"]
                                            and winner["brier"] <= ref["brier"] * 1.05) else "Nenhuma"
    if use_strategy != "Nenhuma":
        best["_pipe"] = candidates[use_strategy]
        best["cv"], _, best["_oof"] = cv_evaluate(best["_pipe"], train, cv)
        best["_fitted"] = clone(best["_pipe"]).fit(train, train.churn)
    results["imbalance_tuned"] = imbalance_tuned
    results["imbalance_strategy"] = use_strategy
    log(f"Estratégia de balanceamento adotada: {use_strategy}")

    # ------------------------------------------------------------------ #
    # 6. Threshold e faixas de risco (out-of-fold, sem tocar no teste)
    # ------------------------------------------------------------------ #
    oof = best["_oof"]
    oof_y = train.loc[oof.idx, "churn"].to_numpy()
    oof_v = train.loc[oof.idx, "valor_mensal"].to_numpy()
    th = choose_threshold(oof_y, oof.proba.to_numpy(), oof_v)
    t_op = th["threshold_negocio"]
    base_rate = float(train.churn.mean())
    t_medium = round(min(max(1.5 * base_rate, t_op / 2), t_op - 0.01), 2)
    risk_thresholds = {"medio": t_medium, "alto": t_op}
    log(f"Threshold operacional (máx. resultado esperado, OOF): {t_op:.2f} | F1-ótimo: {th['threshold_f1']:.2f}")
    log(f"Faixas de risco: Baixo < {t_medium:.2f} ≤ Médio < {t_op:.2f} ≤ Alto")

    # ------------------------------------------------------------------ #
    # 7. Avaliação final no teste
    # ------------------------------------------------------------------ #
    eval_model = best["_fitted"]
    p_test = eval_model.predict_proba(test)[:, 1]
    test_metrics_op = classification_metrics(test.churn, p_test, t_op)
    test_metrics_05 = classification_metrics(test.churn, p_test, 0.5)
    thr_table_test = threshold_table(test.churn, p_test, test.valor_mensal,
                                     sorted({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, t_op, t_medium}))
    risk_test = classify_risk(p_test, risk_thresholds)
    risk_summary = (pd.DataFrame({"risco": risk_test, "churn": test.churn})
                    .groupby("risco").churn.agg(clientes="size", churns="sum", taxa_real="mean")
                    .reindex(["Alto", "Médio", "Baixo"]).reset_index())
    lifts = lift_table(test.churn, p_test)
    ev_test = expected_value(test.churn, p_test, test.valor_mensal, t_op)
    log(f"Teste @ {t_op:.2f}: precision={test_metrics_op['precision']:.3f} recall={test_metrics_op['recall']:.3f} "
        f"F1={test_metrics_op['f1']:.3f} | ROC-AUC={test_metrics_op['roc_auc']:.3f} PR-AUC={test_metrics_op['pr_auc']:.3f}")

    test_pred = test[["customer_id", "data_referencia", "plano", "valor_mensal", "churn"]].copy()
    test_pred["probabilidade"] = p_test
    test_pred["nivel_risco"] = risk_test
    test_pred["previsto"] = (p_test >= t_op).astype(int)
    test_pred.to_parquet(config.TEST_PREDICTIONS_FILE, index=False)

    # ------------------------------------------------------------------ #
    # 8. Explicabilidade global
    # ------------------------------------------------------------------ #
    log("Permutation importance (teste)...")
    fe = eval_model.named_steps["features"]
    rest = eval_model[1:]
    X_test_fe = fe.transform(test)
    sample = X_test_fe.sample(min(len(X_test_fe), 12000), random_state=config.RANDOM_SEED)
    perm = permutation_importance(rest, sample, test.churn.loc[sample.index], scoring="average_precision",
                                  n_repeats=3 if fast else 5, random_state=config.RANDOM_SEED, n_jobs=1)
    perm_df = pd.DataFrame({"feature": MODEL_FEATURES, "importance": perm.importances_mean, "std": perm.importances_std})
    perm_df["label"] = perm_df.feature.map(FEATURE_LABELS)
    perm_df = perm_df.sort_values("importance", ascending=False)

    log("SHAP (amostra do teste)...")
    background = train.sample(500, random_state=config.RANDOM_SEED)
    explainer = ChurnExplainer(eval_model, background=background, risk_thresholds=risk_thresholds)
    shap_sample = test.sample(3000, random_state=config.RANDOM_SEED)
    shap_df = explainer.shap_values(shap_sample)
    shap_global = shap_df.abs().mean().sort_values(ascending=False)
    fe_sample = add_features(shap_sample)
    # Direção: correlação entre o valor do atributo e sua contribuição SHAP
    # (+ → valores altos aumentam o risco; − → valores altos reduzem o risco).
    shap_dir = {}
    for f in shap_df.columns:
        x = fe_sample[f]
        if f in CATEGORICAL_FEATURES or x.nunique() < 2 or shap_df[f].std() == 0:
            shap_dir[f] = None
            continue
        shap_dir[f] = float(np.corrcoef(x.astype(float).fillna(x.median()), shap_df[f])[0, 1])
    shap_df.assign(churn=shap_sample.churn.values).to_parquet(config.REPORTS_DIR / "shap_sample.parquet")
    fe_sample[MODEL_FEATURES].to_parquet(config.REPORTS_DIR / "shap_sample_features.parquet")

    # Coeficientes da regressão logística (interpretação de direção)
    lr = tuned["logreg"]["_fitted"]
    coef = pd.DataFrame({"feature": lr.named_steps["preprocess"].get_feature_names_out(),
                         "coef": lr.named_steps["model"].coef_[0]}).sort_values("coef", key=abs, ascending=False)

    # ------------------------------------------------------------------ #
    # 9. Modelo de produção + scoring dos clientes ativos
    # ------------------------------------------------------------------ #
    log("Re-treinando o modelo escolhido com todos os dados rotulados (produção)...")
    labeled = base[base.data_referencia <= config.LAST_LABELED_SNAPSHOT]
    prod_model = clone(best["_pipe"]).fit(labeled, labeled.churn)

    scoring = pd.read_parquet(config.SCORING_BASE_FILE)
    prod_explainer = ChurnExplainer(prod_model, background=labeled.sample(500, random_state=config.RANDOM_SEED),
                                    risk_thresholds=risk_thresholds)
    explanations = prod_explainer.explain(scoring, top_k=4)
    scored = scoring.copy()
    scored["probabilidade"] = [e["probabilidade"] for e in explanations]
    scored["nivel_risco"] = [e["nivel_risco"] for e in explanations]
    scored["principal_fator"] = [e["principal_fator"] for e in explanations]
    scored["categoria_principal"] = [e["categoria_principal"] for e in explanations]
    scored["acao_sugerida"] = [e["recomendacao"]["acao_sugerida"] for e in explanations]
    scored["explicacao_json"] = [json.dumps(e, ensure_ascii=False) for e in explanations]
    scored.to_parquet(config.SCORED_CUSTOMERS_FILE, index=False)
    dist = scored.nivel_risco.value_counts().reindex(["Alto", "Médio", "Baixo"]).fillna(0).astype(int).to_dict()
    receita_risco = float(scored.loc[scored.nivel_risco == "Alto", "valor_mensal"].sum())
    log(f"Clientes ativos pontuados: {len(scored):,} | {dist} | receita mensal em alto risco: R$ {receita_risco:,.2f}")

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_key": best_key,
        "model_name": best["name"],
        "best_params": best["best_params"],
        "imbalance_strategy": use_strategy,
        "target": f"churn em {config.CHURN_HORIZON_DAYS} dias",
        "horizon_days": config.CHURN_HORIZON_DAYS,
        "training_data": {"production_fit": f"{labeled.data_referencia.min().date()} → {labeled.data_referencia.max().date()}",
                          "production_rows": len(labeled), **results["split"]},
        "features": {"categorical": CATEGORICAL_FEATURES, "numeric": NUMERIC_FEATURES, "binary": BINARY_FEATURES,
                     "raw_input_columns": RAW_INPUT_COLUMNS, "excluded": EXCLUDED_COLUMNS},
        "threshold": {"operacional": t_op, "f1_otimo": th["threshold_f1"], "criterio": "máximo resultado líquido esperado (OOF do treino)"},
        "risk_thresholds": risk_thresholds,
        "business_assumptions": config.BUSINESS,
        "global_importance": [
            {"feature": r.feature, "label": r.label, "permutation_importance": round(float(r.importance), 5),
             "mean_abs_shap": round(float(shap_global.get(r.feature, 0.0)), 5),
             "direction": (None if shap_dir.get(r.feature) is None else round(shap_dir[r.feature], 3))}
            for r in perm_df.head(15).itertuples()
        ],
        "metrics": {
            "cv": best["cv"],
            "test_at_operational_threshold": test_metrics_op,
            "test_at_0_5": test_metrics_05,
            "test_expected_value": ev_test,
        },
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    joblib.dump({"pipeline": prod_model, "metadata": metadata, "background": labeled.sample(300, random_state=1)[RAW_INPUT_COLUMNS]},
                config.MODEL_FILE, compress=3)
    joblib.dump({"pipeline": eval_model, "metadata": metadata}, config.MODELS_DIR / "churn_model_eval.joblib", compress=3)
    config.MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    for k in tuned:
        for private in ("_pipe", "_fitted", "_oof"):
            tuned[k].pop(private)
    results.update({
        "tuned": tuned,
        "selected": best_key,
        "threshold_oof_table": th["tabela"].to_dict("records"),
        "threshold_test_table": thr_table_test.to_dict("records"),
        "risk_thresholds": risk_thresholds,
        "risk_summary_test": risk_summary.to_dict("records"),
        "lift_test": lifts.reset_index().astype({"decil": int}).to_dict("records"),
        "test_metrics_operational": test_metrics_op,
        "test_metrics_05": test_metrics_05,
        "test_expected_value": ev_test,
        "permutation_importance": perm_df.to_dict("records"),
        "shap_global": [{"feature": f, "label": FEATURE_LABELS.get(f, f), "mean_abs_shap": float(v), "direction": shap_dir.get(f)}
                        for f, v in shap_global.items()],
        "logreg_coefficients": coef.head(25).to_dict("records"),
        "scoring": {"clientes": len(scored), "distribuicao_risco": dist, "receita_mensal_alto_risco": receita_risco,
                    "receita_mensal_total": float(scored.valor_mensal.sum()),
                    "probabilidade_media": float(scored.probabilidade.mean())},
        "runtime_seconds": round(time.time() - t0, 1),
        "fast_mode": fast,
    })
    (config.REPORTS_DIR / "training_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")
    # Perfil de referência para o monitoramento de drift (dados de treino do modelo validado)
    from ml.src.monitoring.drift import main as build_reference_profile
    build_reference_profile()
    log(f"Concluído em {results['runtime_seconds']:.0f}s → {config.MODEL_FILE.relative_to(config.PROJECT_ROOT)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="busca reduzida de hiperparâmetros")
    parser.add_argument("--reuse", action="store_true", help="reutiliza modelos iniciais/busca da última execução")
    args = parser.parse_args()
    main(fast=args.fast, reuse=args.reuse)
