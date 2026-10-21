"""Which sensors drive the fresh-vs-deteriorated decision? SHAP on XGBoost, aggregated per sensor.

Fitted on each LOSO training fold and explained on its held-out series, so the attribution
reflects what generalises across beef cuts rather than what fits one series.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import LeaveOneGroupOut

from common import RESULTS, ensemble_classifiers
from enose import build_features, load_raw


def main() -> None:
    df = load_raw()
    X = build_features(df)
    y = (df["y"] >= 2).astype(int).to_numpy()
    groups = df["series"].to_numpy()
    model = ensemble_classifiers()["XGBoost"]
    parts = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = model.fit(X.iloc[tr], y[tr])
        sv = shap.TreeExplainer(m).shap_values(X.iloc[te])
        parts.append(pd.Series(np.abs(sv).mean(axis=0), index=X.columns))
    feat = pd.concat(parts, axis=1).mean(axis=1).sort_values(ascending=False)
    feat.rename("mean_abs_shap").to_csv(RESULTS / "enose_shap_features.csv")
    sensor = feat.groupby(feat.index.str.split("_").str[0]).sum().sort_values(ascending=False)
    sensor.rename("mean_abs_shap").to_csv(RESULTS / "enose_shap_sensors.csv")
    print(sensor.round(3).to_string())


if __name__ == "__main__":
    main()
