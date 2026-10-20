"""Track B experiments: split design, ensemble comparison, Optuna tuning, imbalanced-regression weights, SHAP."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from common import RESULTS, SEED, ensemble_regressors, single_regressors
from solubility import (AQSOL_DESC, add_scaffolds, inverse_frequency_weights, load, per_bin_errors,
                        random_split, rdkit_descriptors, reg_scores, scaffold_split)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main() -> None:
    df = add_scaffolds(load())
    X_rd = rdkit_descriptors(df)
    X_aq = df[AQSOL_DESC].astype("float32")
    y = df["logS"].to_numpy()
    tr, te = scaffold_split(df)
    rtr, rte = random_split(len(df))
    print(f"n={len(df)} scaffolds={df.scaffold.nunique()} rdkit_features={X_rd.shape[1]} "
          f"scaffold train/test={len(tr)}/{len(te)}", flush=True)
    meta = {"n": len(df), "n_scaffolds": int(df.scaffold.nunique()),
            "n_rdkit_features": int(X_rd.shape[1]), "n_train": len(tr), "n_test": len(te),
            "test_scaffolds_seen_in_train": int(df.scaffold.iloc[te].isin(df.scaffold.iloc[tr]).sum())}
    groups_tr = df["scaffold"].iloc[tr].to_numpy()
    inner = list(GroupKFold(5).split(tr, groups=groups_tr))
    rows = []

    def run(tag, name, model, X, split="scaffold", label="", sample_weight=None):
        a, b = (tr, te) if split == "scaffold" else (rtr, rte)
        m = clone(model)
        if hasattr(m, "cv") and split == "scaffold":
            m.set_params(cv=inner)
        t = time.time()
        kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        m.fit(X.iloc[a], y[a], **kw)
        p = m.predict(X.iloc[b])
        s = reg_scores(y[b], p)
        rows.append({"experiment": tag, "model": name, "split": split, "variant": label,
                     **s, "seconds": round(time.time() - t, 1)})
        print(f"{tag:12s} {name:13s} {split:8s} {label:18s} RMSE={s['rmse']:.3f} R2={s['r2']:.3f}", flush=True)
        return m, p

    lgbm = ensemble_regressors()["LightGBM"]
    rf = ensemble_regressors()["RandomForest"]

    # S1 - split design and feature set
    for split in ["random", "scaffold"]:
        for fname, X in [("AqSolDB-17", X_aq), ("RDKit-2D", X_rd)]:
            run("S1_split", "LightGBM", lgbm, X, split, fname)
            run("S1_split", "RandomForest", rf, X, split, fname)

    # S2 - single vs ensemble models (scaffold split, RDKit 2D)
    preds = {}
    for name, m in {**single_regressors(), **ensemble_regressors()}.items():
        _, preds[name] = run("S2_models", name, m, X_rd, label="RDKit-2D")

    # S3 - Optuna (TPE, Bayesian) on LightGBM with scaffold-grouped CV on the training set only
    Xtr, ytr = X_rd.iloc[tr], y[tr]

    def objective(trial):
        params = dict(n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
                      learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                      num_leaves=trial.suggest_int("num_leaves", 15, 127),
                      min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
                      subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
                      colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                      reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                      random_state=SEED, verbose=-1, n_jobs=4)
        errs = []
        for a, b in inner:
            m = lgb.LGBMRegressor(**params).fit(Xtr.iloc[a], ytr[a])
            errs.append(np.sqrt(np.mean((m.predict(Xtr.iloc[b]) - ytr[b]) ** 2)))
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t = time.time()
    study.optimize(objective, n_trials=40)
    print(f"optuna best CV RMSE={study.best_value:.3f} in {time.time() - t:.0f}s", flush=True)
    best = lgb.LGBMRegressor(**study.best_params, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=4)
    tuned, p_tuned = run("S3_optuna", "LightGBM-Optuna", best, X_rd, label="40 trials TPE")
    study.trials_dataframe().to_csv(RESULTS / "solubility_optuna_trials.csv", index=False)
    meta["optuna_best_params"] = study.best_params
    meta["optuna_best_cv_rmse"] = study.best_value

    # S4 - imbalanced regression: rare solubility bins get larger sample weights
    w = inverse_frequency_weights(df["sol_bin"].iloc[tr].astype(str))
    _, p_w = run("S4_imbalance", "LightGBM-Optuna", best, X_rd, label="inv-freq weights", sample_weight=w)
    bins = per_bin_errors(df.iloc[te], p_tuned).rename(columns={"mae": "mae_unweighted"})
    bins["mae_weighted"] = per_bin_errors(df.iloc[te], p_w)["mae"]
    bins.to_csv(RESULTS / "solubility_per_bin.csv", index=False)
    print(bins.to_string(), flush=True)

    # SHAP on the tuned model (sample of the test set)
    sample = X_rd.iloc[te].sample(min(1000, len(te)), random_state=SEED)
    sv = shap.TreeExplainer(tuned).shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    imp.head(20).rename("mean_abs_shap").to_csv(RESULTS / "solubility_shap_top20.csv")

    pd.DataFrame({"logS": y[te], "sol_bin": df["sol_bin"].iloc[te].astype(str),
                  **{f"pred_{k}": v for k, v in preds.items()}, "pred_LightGBM-Optuna": p_tuned}
                 ).to_csv(RESULTS / "solubility_test_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "solubility_summary.csv", index=False)
    (RESULTS / "solubility_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
