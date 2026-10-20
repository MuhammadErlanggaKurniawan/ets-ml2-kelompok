"""Collect every number shown in the slides into deck/data.json (no number is typed by hand)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from common import RESULTS, ROOT

R = lambda x, n=3: float(round(float(x), n))  # noqa: E731


def al_summary(path: Path, metric_higher_better: bool) -> dict:
    al = pd.read_csv(path)
    keys = ["policy", "round", "n_labelled"]
    curve = al.groupby(keys).test.agg(["mean", "std"]).reset_index()
    out = {"n_labelled": sorted(curve.n_labelled.unique().tolist()), "curves": {}, "final": {}}
    for pol, g in curve.groupby("policy"):
        g = g.sort_values("n_labelled")
        out["curves"][pol] = [R(v) for v in g["mean"]]
        out["final"][pol] = {"mean": R(g["mean"].iloc[-1]), "sd": R(g["std"].iloc[-1])}
    # area under the learning curve (mean test score over rounds) - one number per policy
    out["aulc"] = {p: R(np.mean(v)) for p, v in out["curves"].items()}
    # labels each policy needs to match random sampling's final score
    target = out["final"]["random"]["mean"]
    need = {}
    for pol, v in out["curves"].items():
        hit = [n for n, s in zip(out["n_labelled"], v) if (s >= target if metric_higher_better else s <= target)]
        need[pol] = int(hit[0]) if hit else None
    out["labels_to_match_random"] = need
    out["target_random_final"] = target
    arms = al[al.policy.str.startswith("bandit") & (al.arm != "init")]
    out["arm_share"] = {p: {a: R(c, 3) for a, c in g.arm.value_counts(normalize=True).items()}
                        for p, g in arms.groupby("policy")}
    out["full_pool_test"] = R(al["full_pool_test"].mean())
    out["pool_size"] = int(al["pool_size"].iloc[0])
    return out


def main() -> None:
    d = {}
    es = pd.read_csv(RESULTS / "enose_summary.csv")
    pick = lambda q: es.query(q).iloc[0]  # noqa: E731
    d["enose_split"] = [
        {"label": f"{r.split}/{r.features}", "model": r.model, "f1": R(r.macro_f1_mean), "sd": R(r.macro_f1_sd)}
        for r in es[es.experiment == "E1_split"].itertuples()]
    d["enose_temp"] = {"without": R(pick("experiment=='E2_models' and model=='LightGBM' and task=='4-class'").macro_f1_mean),
                       "with": R(pick("experiment=='E1_temp'").macro_f1_mean)}
    for task in ["4-class", "binary"]:
        sub = es[(es.experiment == "E2_models") & (es.task == task)]
        d[f"enose_models_{task}"] = [{"model": r.model, "f1": R(r.macro_f1_mean), "sd": R(r.macro_f1_sd)}
                                     for r in sub.itertuples()]
    d["enose_imbalance"] = [{"model": r.model, "variant": r.variant, "f1": R(r.macro_f1_mean),
                             "sd": R(r.macro_f1_sd), "recall_excellent": R(r.recall_excellent_mean)}
                            for r in es[es.experiment == "E3_imbalance"].itertuples()]
    d["enose_augment"] = [{"model": r.model, "variant": r.variant, "f1": R(r.macro_f1_mean), "sd": R(r.macro_f1_sd)}
                          for r in es[es.experiment == "E4_augment"].itertuples()]
    wr = pd.read_csv(RESULTS / "enose_wrangling_report.csv")
    d["enose_classes"] = {c: int(wr[c].sum()) for c in ["excellent", "good", "acceptable", "spoiled"]}
    d["enose_rows"] = int(wr.rows.sum())
    d["enose_shap_sensors"] = {k: R(v) for k, v in
                               pd.read_csv(RESULTS / "enose_shap_sensors.csv", index_col=0).iloc[:, 0].items()}

    ss = pd.read_csv(RESULTS / "solubility_summary.csv")
    d["sol_split"] = [{"model": r.model, "split": r.split, "features": r.variant, "rmse": R(r.rmse), "r2": R(r.r2)}
                      for r in ss[ss.experiment == "S1_split"].itertuples()]
    d["sol_models"] = [{"model": r.model, "rmse": R(r.rmse), "r2": R(r.r2), "mae": R(r.mae)}
                       for r in ss[ss.experiment.isin(["S2_models", "S3_optuna"])].itertuples()]
    w = ss[ss.experiment == "S4_imbalance"].iloc[0]
    d["sol_weighted_overall"] = {"rmse": R(w.rmse), "r2": R(w.r2), "mae": R(w.mae)}
    pb = pd.read_csv(RESULTS / "solubility_per_bin.csv")
    d["sol_per_bin"] = [{"bin": r.sol_bin, "n": int(r.n), "mae_unweighted": R(r.mae_unweighted),
                         "mae_weighted": R(r.mae_weighted)} for r in pb.itertuples()]
    d["sol_shap_top"] = {k: R(v) for k, v in
                         pd.read_csv(RESULTS / "solubility_shap_top20.csv", index_col=0).iloc[:10, 0].items()}
    d["sol_meta"] = json.loads((RESULTS / "solubility_meta.json").read_text())

    d["al_solubility"] = al_summary(RESULTS / "al_solubility.csv", metric_higher_better=False)
    d["al_enose"] = al_summary(RESULTS / "al_enose.csv", metric_higher_better=True)

    (ROOT / "deck" / "data.json").write_text(json.dumps(d, indent=2))
    print(json.dumps({k: d[k] for k in ["al_solubility", "al_enose"]}, indent=1)[:4000])


if __name__ == "__main__":
    main()
