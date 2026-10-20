"""Track A: food freshness from a low-cost MOS e-nose (proxy for the Raspberry Pi multisensor thesis).

Data: Wijaya et al. (2018) beef e-nose, 5 series (beef cuts) x 2160 minutes, 4 TVC-based classes.
The unit of generalisation is the *series* (like `batch_id` in the thesis), so the honest
validation is Leave-One-Series-Out (LOSO). Temperature is kept as metadata, not a predictor,
mirroring the thesis variable roles.
"""
import glob

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from common import RAW, SEED

CLASSES = ["excellent", "good", "acceptable", "spoiled"]   # ordinal: fresh -> spoiled
GAS = ["MQ135", "MQ136", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ8", "MQ9"]
PREDICTORS_RAW = GAS + ["Humidity"]
BASELINE_MIN = 10   # first minutes of each series act as the reference, like Rs/R0 in the thesis
WINDOW = 15


def load_raw() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(RAW / "enose_beef" / "TS*.csv"))):
        df = pd.read_csv(f)
        df["series"] = f.split("/")[-1].removesuffix(".csv")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["y"] = df["class"].map({c: i for i, c in enumerate(CLASSES)})
    return df


def wrangling_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("series"):
        rows.append({"series": s, "rows": len(g), "missing": int(g[PREDICTORS_RAW].isna().sum().sum()),
                     "duplicated_minutes": int(g["minute"].duplicated().sum()),
                     "temp_range_c": f"{g.Temperature.min():.1f}-{g.Temperature.max():.1f}",
                     "humidity_range": f"{g.Humidity.min():.1f}-{g.Humidity.max():.1f}",
                     **{c: int((g["class"] == c).sum()) for c in CLASSES}})
    return pd.DataFrame(rows)


def engineer(series_df: pd.DataFrame) -> pd.DataFrame:
    """Causal features for one series: baseline ratio, rolling mean and rolling slope (past only)."""
    g = series_df.sort_values("minute").copy()
    out = pd.DataFrame(index=g.index)
    for c in GAS:
        base = g[c].iloc[:BASELINE_MIN].mean()
        ratio = g[c] / base
        out[f"{c}_ratio"] = ratio
        out[f"{c}_ratio_roll"] = ratio.rolling(WINDOW, min_periods=1).mean()
        out[f"{c}_slope"] = ratio.diff(WINDOW).fillna(0) / WINDOW
    out["Humidity"] = g["Humidity"]
    return out


def build_features(df: pd.DataFrame, kind: str = "engineered", with_temperature: bool = False):
    if kind == "raw":
        X = df[PREDICTORS_RAW].copy()
    else:
        X = pd.concat([engineer(g) for _, g in df.groupby("series", sort=False)]).loc[df.index]
    if with_temperature:
        X["Temperature"] = df["Temperature"]
    return X


def augment_series(df: pd.DataFrame, copies: int, jitter: float, scale: float, rng) -> pd.DataFrame:
    """Train-only time-series augmentation on raw signals.

    jitter : Gaussian noise with sd = jitter * per-sensor sd of that series (sensor noise)
    scale  : per-copy, per-sensor gain ~ N(1, scale) (unit-to-unit MOS sensitivity drift)
    """
    out = []
    for s, g in df.groupby("series"):
        for k in range(copies):
            a = g.copy()
            for c in PREDICTORS_RAW:
                gain = rng.normal(1.0, scale) if scale > 0 else 1.0
                noise = rng.normal(0, jitter * g[c].std(), len(g)) if jitter > 0 else 0.0
                a[c] = g[c] * gain + noise
            a["series"] = f"{s}_aug{k}"
            out.append(a)
    return pd.concat(out, ignore_index=True)


def scores(y_true, y_pred) -> dict:
    return {"macro_f1": f1_score(y_true, y_pred, average="macro"),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "recall_excellent": recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0),
            "recall_last": recall_score(y_true, y_pred, labels=[int(np.max(y_true))], average="macro",
                                        zero_division=0)}


def _inner_group_splits(groups):
    """Leave-one-series-out splits inside the training fold, for the stacking meta-learner."""
    return list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups))


def evaluate(model, df: pd.DataFrame, *, split: str = "loso", kind: str = "engineered",
             with_temperature: bool = False, resampler=None, augment=None,
             seed: int = SEED, return_pred: bool = False):
    """Fit/score `model` with every preprocessing step done inside the training fold only."""
    X_all = build_features(df, kind, with_temperature)
    y_all, groups = df["y"].to_numpy(), df["series"].to_numpy()
    if split == "loso":
        folds = list(LeaveOneGroupOut().split(X_all, y_all, groups))
    else:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X_all, y_all))
    rng = np.random.default_rng(seed)
    per_fold, preds = [], np.empty(len(df), dtype=int)
    for tr, te in folds:
        Xtr, ytr, gtr = X_all.iloc[tr], y_all[tr], groups[tr]
        if augment is not None:
            aug = augment_series(df.iloc[tr], rng=rng, **augment)
            Xa = build_features(aug, kind, with_temperature)
            Xtr = pd.concat([Xtr, Xa], ignore_index=True)
            ytr = np.concatenate([ytr, aug["y"].to_numpy()])
            gtr = np.concatenate([gtr, aug["series"].str.split("_aug").str[0].to_numpy()])
        if resampler is not None:
            Xtr, ytr = clone(resampler).fit_resample(Xtr, ytr)
            gtr = None
        m = clone(model)
        if hasattr(m, "cv") and split == "loso" and gtr is not None:
            m.set_params(cv=_inner_group_splits(gtr))
        m.fit(Xtr, ytr)
        p = m.predict(X_all.iloc[te]).astype(int).ravel()
        preds[te] = p
        per_fold.append({"fold": str(groups[te][0]) if split == "loso" else len(per_fold),
                         **scores(y_all[te], p)})
    res = pd.DataFrame(per_fold)
    return (res, preds) if return_pred else res


RESAMPLERS = {"none": None,
              "SMOTE": SMOTE(random_state=SEED),
              "RandomUnder": RandomUnderSampler(random_state=SEED)}
