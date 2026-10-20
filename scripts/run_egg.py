"""Track C (Rijal): same pipeline on EfficientNet-B0 embeddings once the yolk data exist.

    uv run python scripts/run_egg.py                 # real data: data/egg/embeddings_efficientnet_b0.csv
    uv run python scripts/run_egg.py --smoke-test    # synthetic data, only checks the code path

Hybrid scenario of the thesis = CNN embeddings + XGBoost; compared against a single linear head.
Imbalance handling (SMOTE) and the bandit active-learning loop are fitted on training data only.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from active_learning import proba_entropy, run_active_learning
from common import RESULTS, SEED

EMB = Path(__file__).resolve().parents[1] / "data" / "egg" / "embeddings_efficientnet_b0.csv"


def load(smoke: bool):
    if smoke:
        X, y = make_classification(n_samples=372, n_features=64, n_informative=12, n_classes=3,
                                   weights=[0.2, 0.5, 0.3], random_state=SEED)
        return X, y
    df = pd.read_csv(EMB)
    return df.filter(regex=r"^f\d+$").to_numpy(), df["y"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    smoke = ap.parse_args().smoke_test
    X, y = load(smoke)
    tag = "smoke" if smoke else "real"
    models = {
        "LogReg (single)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "XGBoost (hybrid)": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.5, n_jobs=4,
                                          random_state=SEED, verbosity=0),
    }
    rows = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
        for name, m in models.items():
            for smote in [False, True]:
                Xtr, ytr = (SMOTE(random_state=SEED).fit_resample(X[tr], y[tr]) if smote
                            else (X[tr], y[tr]))
                p = m.fit(Xtr, ytr).predict(X[te])
                rows.append({"fold": fold, "model": name, "smote": smote,
                             "macro_f1": f1_score(y[te], p, average="macro")})
    res = pd.DataFrame(rows)
    print(res.groupby(["model", "smote"]).macro_f1.agg(["mean", "std"]).round(3))
    res.to_csv(RESULTS / f"egg_{tag}_models.csv", index=False)

    Xp, Xte, yp, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    Xp, Xval, yp, yval = train_test_split(Xp, yp, test_size=0.2, stratify=yp, random_state=SEED)
    logs = []
    for pol in ["random", "uncertainty", "diversity", "bandit-ucb"]:
        log = run_active_learning(models["XGBoost (hybrid)"], Xp, yp, Xval, yval, Xte, yte,
                                  policy=pol, score=lambda a, b: f1_score(a, b, average="macro"),
                                  uncertainty=proba_entropy, n_init=24, batch=12, rounds=8,
                                  seed=SEED, stratify_init=True)
        logs += [dict(r, policy=pol) for r in log]
        print(f"{pol:12s} labels={log[-1]['n_labelled']} test F1={log[-1]['test']:.3f}")
    pd.DataFrame(logs).to_csv(RESULTS / f"egg_{tag}_al.csv", index=False)


if __name__ == "__main__":
    main()
