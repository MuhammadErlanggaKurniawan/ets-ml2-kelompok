"""Track A experiments: leakage check, ensemble comparison, imbalance handling, augmentation, SHAP."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from common import RESULTS, ensemble_classifiers, single_classifiers
from enose import RESAMPLERS, evaluate, load_raw, wrangling_report


def summarise(tag: str, res: pd.DataFrame, **meta) -> dict:
    row = {"experiment": tag, **meta}
    for m in ["macro_f1", "accuracy", "balanced_acc", "recall_excellent", "recall_last"]:
        row[f"{m}_mean"] = res[m].mean()
        row[f"{m}_sd"] = res[m].std(ddof=1)
    return row


def main() -> None:
    df = load_raw()
    wrangling_report(df).to_csv(RESULTS / "enose_wrangling_report.csv", index=False)
    binary = df.assign(y=(df["y"] >= 2).astype(int))   # fresh (excellent+good) vs deteriorated
    rows, folds = [], []

    def run(tag, model_name, model, data=df, label="", **kw):
        t = time.time()
        res = evaluate(model, data, **kw)
        meta = {"model": model_name, "task": "binary" if data is binary else "4-class",
                "variant": label, "split": kw.get("split", "loso"), "features": kw.get("kind", "engineered"),
                "with_temperature": kw.get("with_temperature", False)}
        rows.append(summarise(tag, res, **meta, seconds=round(time.time() - t, 1)))
        folds.append(res.assign(experiment=tag, **meta))
        print(f"{tag:12s} {model_name:14s} {meta['task']:7s} {meta['split']}/{meta['features']} {label} "
              f"F1={res.macro_f1.mean():.3f}±{res.macro_f1.std(ddof=1):.3f}", flush=True)

    lgbm = ensemble_classifiers()["LightGBM"]
    rf = ensemble_classifiers()["RandomForest"]

    # E1 - validation design: random row split vs leave-one-series-out
    for split in ["random", "loso"]:
        for kind in ["raw", "engineered"]:
            run("E1_split", "LightGBM", lgbm, split=split, kind=kind)
            run("E1_split", "RandomForest", rf, split=split, kind=kind)
    run("E1_temp", "LightGBM", lgbm, kind="engineered", with_temperature=True, label="+Temperature")

    # E2 - single models vs ensembles (LOSO, engineered), both tasks
    for data in [df, binary]:
        for name, m in {**single_classifiers(), **ensemble_classifiers()}.items():
            run("E2_models", name, m, data=data)

    # E3 - imbalance handling (fit on training fold only)
    for name in ["LogReg", "LightGBM"]:
        for cw in [None, "balanced"]:
            zoo = {**single_classifiers(cw), **ensemble_classifiers(cw)}
            run("E3_imbalance", name, zoo[name], label=f"class_weight={cw}")
        zoo = {**single_classifiers(), **ensemble_classifiers()}
        for rname in ["SMOTE", "RandomUnder"]:
            run("E3_imbalance", name, zoo[name], resampler=RESAMPLERS[rname], label=rname)

    # E4 - train-only time-series augmentation
    augs = {"jitter": dict(copies=2, jitter=0.05, scale=0.0),
            "scaling": dict(copies=2, jitter=0.0, scale=0.10),
            "jitter+scaling": dict(copies=2, jitter=0.05, scale=0.10),
            "jitter+scaling x4": dict(copies=4, jitter=0.05, scale=0.10)}
    for name in ["LogReg", "LightGBM"]:
        m = {**single_classifiers(), **ensemble_classifiers()}[name]
        run("E4_augment", name, m, label="none")
        for aname, a in augs.items():
            run("E4_augment", name, m, augment=a, label=aname)

    pd.DataFrame(rows).to_csv(RESULTS / "enose_summary.csv", index=False)
    pd.concat(folds).to_csv(RESULTS / "enose_folds.csv", index=False)


if __name__ == "__main__":
    main()
