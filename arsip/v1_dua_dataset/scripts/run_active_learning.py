"""Reinforcement aspect: bandit-driven active learning on both tracks.

    uv run python scripts/run_active_learning.py solubility
    uv run python scripts/run_active_learning.py enose
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error
from xgboost import XGBClassifier

from active_learning import forest_std, proba_entropy, run_active_learning
from common import RESULTS, SEED

POLICIES = ["random", "uncertainty", "diversity", "bandit-eps", "bandit-ucb"]


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro"))


def solubility(seeds=range(5)):
    from solubility import add_scaffolds, load, rdkit_descriptors, scaffold_split
    df = add_scaffolds(load())
    X = rdkit_descriptors(df).to_numpy()
    y = df["logS"].to_numpy()
    tr, te = scaffold_split(df)
    # validation for the reward is a scaffold split *inside* the training set
    sub = df.iloc[tr].reset_index(drop=True)
    ptr, pval = scaffold_split(sub, test_frac=0.15)
    pool, val = tr[ptr], tr[pval]
    X = np.where(np.isnan(X), np.nanmedian(X[pool], axis=0), X)   # pool-only median imputation
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.3,
                                  n_jobs=-1, random_state=SEED)
    full = model.fit(X[pool], y[pool])
    ref = {"full_pool_test": rmse(y[te], full.predict(X[te])), "pool_size": len(pool)}
    print(f"pool={len(pool)} val={len(val)} test={len(te)} full-pool test RMSE={ref['full_pool_test']:.3f}",
          flush=True)
    logs = []
    for seed in seeds:
        for pol in POLICIES:
            t = time.time()
            log = run_active_learning(model, X[pool], y[pool], X[val], y[val], X[te], y[te],
                                      policy=pol, score=rmse, uncertainty=forest_std,
                                      n_init=100, batch=100, rounds=20, seed=seed,
                                      higher_is_better=False)
            logs += [dict(r, policy=pol, seed=seed, **ref) for r in log]
            print(f"seed={seed} {pol:12s} final test RMSE={log[-1]['test']:.3f} ({time.time() - t:.0f}s)",
                  flush=True)
    pd.DataFrame(logs).to_csv(RESULTS / "al_solubility.csv", index=False)


def enose(seeds=range(3)):
    from enose import build_features, load_raw
    df = load_raw()
    X = build_features(df).to_numpy()
    # binary task (fresh = excellent+good vs deteriorated), the label scheme of the thesis;
    # the 4-class cross-series task is too unstable with 3 pool series to read AL curves
    y = (df["y"].to_numpy() >= 2).astype(int)
    series = df["series"].to_numpy()
    names = sorted(np.unique(series))
    model = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=4, random_state=SEED, verbosity=0)
    logs = []
    for i, test_s in enumerate(names):
        val_s = names[(i + 1) % len(names)]
        te = np.flatnonzero(series == test_s)
        val = np.flatnonzero(series == val_s)
        pool = np.flatnonzero((series != test_s) & (series != val_s))
        full = model.fit(X[pool], y[pool])
        ref = {"full_pool_test": macro_f1(y[te], full.predict(X[te])), "pool_size": len(pool),
               "test_series": test_s, "val_series": val_s}
        for seed in seeds:
            for pol in POLICIES:
                log = run_active_learning(model, X[pool], y[pool], X[val], y[val], X[te], y[te],
                                          policy=pol, score=macro_f1, uncertainty=proba_entropy,
                                          n_init=40, batch=40, rounds=15, seed=seed,
                                          stratify_init=True)
                logs += [dict(r, policy=pol, seed=seed, **ref) for r in log]
                print(f"test={test_s} seed={seed} {pol:12s} final F1={log[-1]['test']:.3f} "
                      f"(full pool {ref['full_pool_test']:.3f})", flush=True)
    pd.DataFrame(logs).to_csv(RESULTS / "al_enose.csv", index=False)


if __name__ == "__main__":
    {"solubility": solubility, "enose": enose}[sys.argv[1]]()
