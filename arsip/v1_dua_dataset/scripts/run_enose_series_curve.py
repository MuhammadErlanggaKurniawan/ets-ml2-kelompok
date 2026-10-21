"""More rows or more batches? Fresh-vs-deteriorated macro-F1 as a function of the number of
training series (every combination), compared with subsampling rows inside the same series.
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

from common import RESULTS, SEED, ensemble_classifiers
from enose import build_features, load_raw


def main() -> None:
    df = load_raw()
    X = build_features(df)
    y = (df["y"] >= 2).astype(int).to_numpy()
    series = df["series"].to_numpy()
    names = sorted(np.unique(series))
    model = ensemble_classifiers()["XGBoost"]
    rng = np.random.default_rng(SEED)
    rows = []
    for test_s in names:
        te = series == test_s
        others = [s for s in names if s != test_s]
        for k in range(1, len(others) + 1):
            for combo in combinations(others, k):
                tr = np.flatnonzero(np.isin(series, combo))
                m = clone(model).fit(X.iloc[tr], y[tr])
                rows.append({"test_series": test_s, "n_series": k, "design": "more series",
                             "n_rows": len(tr), "macro_f1": f1_score(y[te], m.predict(X[te]), average="macro")})
        # same row budget as k series, but all drawn from the 4 training series at once
        all_tr = np.flatnonzero(~te)
        for k in range(1, len(others) + 1):
            n = int(len(all_tr) * k / len(others))
            tr = rng.choice(all_tr, n, replace=False)
            m = clone(model).fit(X.iloc[tr], y[tr])
            rows.append({"test_series": test_s, "n_series": k, "design": "rows from 4 series",
                         "n_rows": n, "macro_f1": f1_score(y[te], m.predict(X[te]), average="macro")})
        # tiny row budgets from all 4 series: does the label count matter at all?
        for n in [40, 160, 640]:
            tr = rng.choice(all_tr, n, replace=False)
            m = clone(model).fit(X.iloc[tr], y[tr])
            rows.append({"test_series": test_s, "n_series": 4, "design": f"{n} rows from 4 series",
                         "n_rows": n, "macro_f1": f1_score(y[te], m.predict(X[te]), average="macro")})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "enose_series_curve.csv", index=False)
    print(res.groupby(["design", "n_series"]).macro_f1.agg(["mean", "std", "count"]).round(3))


if __name__ == "__main__":
    main()
