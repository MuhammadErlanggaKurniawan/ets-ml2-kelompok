"""Generate the four self-contained Colab notebooks in notebooks/ from the cell lists below."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "MuhammadErlanggaKurniawan/ets-ml2-kelompok"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n")}


def badge(name):
    return md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
              f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name})")


def write(name, cells):
    nb = {"cells": copy.deepcopy([badge(name)] + cells), "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}, "colab": {"provenance": []}}}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
        lines = c["source"].split("\n")
        c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    (ROOT / "notebooks" / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", name, len(nb["cells"]), "cells")


# ------------------------------------------------------------------ shared cells
INSTALL = code("""
# Versi dikunci sama dengan uv.lock di repo supaya angka di Colab = angka di laporan
%pip install -q scikit-learn==1.9.1 xgboost==3.4.1 lightgbm==4.7.0 optuna==5.0.0 \\
    imbalanced-learn==0.14.2 rdkit==2026.9.1 shap==0.54.1
""")

COMMON = code('''
import warnings, hashlib, json, time, urllib.request, zipfile, glob
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import clone
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")
SEED = 42
RAW, RESULTS = Path("data/raw"), Path("results")
RAW.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, dest):
    import ssl, certifi                       # certifi: CA bundle yang sama di Colab dan macOS
    dest = Path(dest)
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "ets-ml2"})
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, context=ctx) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest

def mean_sd(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}"
''')

ENOSE_DATA = code('''
# Akuisisi: e-nose daging sapi (Wijaya dkk., 2018, Data in Brief 21) - Mendeley Data, CC BY 4.0
ENOSE_URL = ("https://data.mendeley.com/public-files/datasets/mwmhh766fc/files/"
             "2e65305f-5e53-4395-9b28-91dfc73ff490/file_downloaded")
ENOSE_SHA256 = "772bc2c75f2ac4b785538ea38311bdefaee65059b3391b080b4da5ea619963e3"
z = fetch(ENOSE_URL, RAW / "enose_beef.zip")
assert sha256(z) == ENOSE_SHA256, "checksum tidak cocok - file berubah/korup"
zipfile.ZipFile(z).extractall(RAW / "enose_beef")

CLASSES = ["excellent", "good", "acceptable", "spoiled"]      # ordinal: segar -> busuk
GAS = ["MQ135", "MQ136", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ8", "MQ9"]
PREDICTORS_RAW = GAS + ["Humidity"]
BASELINE_MIN, WINDOW = 10, 15

def load_enose():
    frames = []
    for f in sorted(glob.glob(str(RAW / "enose_beef" / "TS*.csv"))):
        d = pd.read_csv(f); d["series"] = Path(f).stem; frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["y"] = d["class"].map({c: i for i, c in enumerate(CLASSES)})
    return d

enose = load_enose()
print(enose.shape); enose.head()
''')

ENOSE_FEATURES = code('''
def engineer(g):
    """Fitur kausal satu seri: rasio terhadap baseline 10 menit pertama, rata-rata bergulir,
    dan kemiringan 15 menit. Semua hanya memakai data masa lalu (bisa dihitung real time)."""
    g = g.sort_values("minute")
    out = pd.DataFrame(index=g.index)
    for c in GAS:
        ratio = g[c] / g[c].iloc[:BASELINE_MIN].mean()
        out[f"{c}_ratio"] = ratio
        out[f"{c}_ratio_roll"] = ratio.rolling(WINDOW, min_periods=1).mean()
        out[f"{c}_slope"] = ratio.diff(WINDOW).fillna(0) / WINDOW
    out["Humidity"] = g["Humidity"]
    return out

def build_features(d, kind="engineered", with_temperature=False):
    X = d[PREDICTORS_RAW].copy() if kind == "raw" else \\
        pd.concat([engineer(g) for _, g in d.groupby("series", sort=False)]).loc[d.index]
    if with_temperature:
        X["Temperature"] = d["Temperature"]
    return X
''')

# ------------------------------------------------------------------ notebook 01
nb1 = [
    md("""
# 01 · Track A — Kesegaran Pangan dari *E-nose* Berbiaya Rendah

**Proyek ETS Machine Learning II — "Belajar dengan Label Mahal"**

Track ini menjadi proksi untuk skripsi Erlangga (MQ-135 + DHT22 + pH pada Raspberry Pi). Data pilot
Erlangga baru **6 baris dari 1 batch**, belum cukup untuk validasi. Karena itu kami memakai dataset
publik *e-nose* daging sapi (Wijaya dkk., 2018). Dataset ini juga memuat **MQ135**, kelembapan, dan suhu,
dengan 5 seri (5 potongan daging) yang berperan seperti `batch_id`.

| Aspek ML II | Bagian notebook |
|---|---|
| Data wrangling | §2 laporan kualitas data, §3 fitur kausal |
| Validasi yang jujur | §4 split acak vs *Leave-One-Series-Out* |
| Ensemble learning | §5 model tunggal vs bagging/boosting/stacking |
| Imbalanced data | §6 class weight, SMOTE, undersampling |
| Data augmentation | §7 jitter dan *gain scaling* sensor (khusus data latih) |
| Interpretabilitas | §8 SHAP per sensor |
| Data tambahan | §9 lebih banyak batch vs lebih banyak baris |
"""),
    md("## 0. Instalasi & utilitas"), INSTALL, COMMON,
    md("## 1. Akuisisi data (unduh + verifikasi SHA-256)"), ENOSE_DATA,
    md("""
## 2. Data wrangling — laporan kualitas per seri
Kami memeriksa nilai hilang, menit ganda, rentang suhu/kelembapan, dan distribusi kelas.
Label berasal dari jumlah mikroba (TVC). Kelasnya **tidak seimbang**: *spoiled* 39%, *excellent* 16%.
"""),
    code('''
report = []
for s, g in enose.groupby("series"):
    report.append({"series": s, "rows": len(g), "missing": int(g[PREDICTORS_RAW].isna().sum().sum()),
                   "duplicated_minutes": int(g["minute"].duplicated().sum()),
                   "temp_range_c": f"{g.Temperature.min():.1f}-{g.Temperature.max():.1f}",
                   **{c: int((g["class"] == c).sum()) for c in CLASSES}})
report = pd.DataFrame(report); display(report)
print(enose["class"].value_counts(normalize=True).round(3))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
for s, g in enose.groupby("series"):
    ax[0].plot(g.minute, g.MQ135, lw=.8, label=s); ax[1].plot(g.minute, g.MQ4, lw=.8, label=s)
ax[0].set_title("MQ135 per seri"); ax[1].set_title("MQ4 per seri"); ax[0].legend(); plt.show()
'''),
    md("""
## 3. Rekayasa fitur kausal
Suhu disimpan sebagai **metadata**, bukan prediktor, sama seperti peran variabel di skripsi Erlangga.
Pengaruh menambahkan suhu diuji terpisah di §4.
"""),
    ENOSE_FEATURES,
    code('X_demo = build_features(enose); print(X_demo.shape); X_demo.describe().T.head(8)'),
    md("""
## 4. Validasi: split acak vs *Leave-One-Series-Out* (LOSO)
Menit yang bertetangga hampir identik. Split baris acak membuat model "menghafal" seri yang sama.
LOSO melatih model pada 4 seri dan menguji pada 1 seri yang belum pernah dilihat. Ini sama dengan
`GroupKFold` per `batch_id` di skripsi Erlangga.
"""),
    code('''
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

def single_classifiers(cw=None):
    return {"LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw)),
            "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight=cw, random_state=SEED)),
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
            "DecisionTree": DecisionTreeClassifier(max_depth=8, class_weight=cw, random_state=SEED)}

def ensemble_classifiers(cw=None):
    return {"RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                                   class_weight=cw, random_state=SEED),
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, n_jobs=4, random_state=SEED, verbosity=0),
            "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                           class_weight=cw, random_state=SEED, verbose=-1, n_jobs=4),
            "Stacking": StackingClassifier(
                estimators=[("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw))),
                            ("svm", make_pipeline(StandardScaler(), SVC(C=10, probability=True, class_weight=cw,
                                                                         random_state=SEED))),
                            ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                          class_weight=cw, random_state=SEED))],
                final_estimator=LogisticRegression(max_iter=2000), cv=5, n_jobs=-1)}

def augment_series(d, copies, jitter, scale, rng):
    """Augmentasi deret waktu pada sinyal mentah: jitter = derau sensor,
    scale = variasi sensitivitas antar unit sensor MOS (gain ~ N(1, scale))."""
    out = []
    for s, g in d.groupby("series"):
        for k in range(copies):
            a = g.copy()
            for c in PREDICTORS_RAW:
                gain = rng.normal(1.0, scale) if scale > 0 else 1.0
                noise = rng.normal(0, jitter * g[c].std(), len(g)) if jitter > 0 else 0.0
                a[c] = g[c] * gain + noise
            a["series"] = f"{s}_aug{k}"; out.append(a)
    return pd.concat(out, ignore_index=True)

def scores(y, p):
    return {"macro_f1": f1_score(y, p, average="macro"), "accuracy": accuracy_score(y, p),
            "balanced_acc": balanced_accuracy_score(y, p),
            "recall_excellent": recall_score(y, p, labels=[0], average="macro", zero_division=0)}

def evaluate(model, d, split="loso", kind="engineered", with_temperature=False,
             resampler=None, augment=None, seed=SEED):
    """Seluruh prapemrosesan (augmentasi, resampling) di-fit HANYA pada fold latih."""
    X_all = build_features(d, kind, with_temperature)
    y_all, groups = d["y"].to_numpy(), d["series"].to_numpy()
    folds = (list(LeaveOneGroupOut().split(X_all, y_all, groups)) if split == "loso"
             else list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X_all, y_all)))
    rng, rows = np.random.default_rng(seed), []
    for tr, te in folds:
        Xtr, ytr, gtr = X_all.iloc[tr], y_all[tr], groups[tr]
        if augment is not None:
            aug = augment_series(d.iloc[tr], rng=rng, **augment)
            Xtr = pd.concat([Xtr, build_features(aug, kind, with_temperature)], ignore_index=True)
            ytr = np.concatenate([ytr, aug["y"].to_numpy()])
            gtr = np.concatenate([gtr, aug["series"].str.split("_aug").str[0].to_numpy()])
        if resampler is not None:
            Xtr, ytr = clone(resampler).fit_resample(Xtr, ytr); gtr = None
        m = clone(model)
        if hasattr(m, "cv") and split == "loso" and gtr is not None:   # meta-learner stacking juga LOSO
            m.set_params(cv=list(LeaveOneGroupOut().split(np.zeros(len(gtr)), groups=gtr)))
        p = m.fit(Xtr, ytr).predict(X_all.iloc[te]).astype(int).ravel()
        rows.append({"fold": groups[te][0] if split == "loso" else len(rows), **scores(y_all[te], p)})
    return pd.DataFrame(rows)

SUMMARY = []
def run(tag, name, model, d=enose, label="", **kw):
    r = evaluate(model, d, **kw)
    SUMMARY.append({"experiment": tag, "model": name, "task": "binary" if d is not enose else "4-class",
                    "variant": label, "split": kw.get("split", "loso"), "features": kw.get("kind", "engineered"),
                    "macro_f1_mean": r.macro_f1.mean(), "macro_f1_sd": r.macro_f1.std(ddof=1)})
    print(f"{tag:12s} {name:13s} {SUMMARY[-1]['task']:7s} {SUMMARY[-1]['split']}/{SUMMARY[-1]['features']:10s} "
          f"{label:18s} macro-F1 = {mean_sd(r.macro_f1)}")
    return r
'''),
    code('''
lgbm, rf = ensemble_classifiers()["LightGBM"], ensemble_classifiers()["RandomForest"]
for split in ["random", "loso"]:
    for kind in ["raw", "engineered"]:
        run("E1_split", "LightGBM", lgbm, split=split, kind=kind)
        run("E1_split", "RandomForest", rf, split=split, kind=kind)
run("E1_temp", "LightGBM", lgbm, with_temperature=True, label="+Temperature")
'''),
    md("""
**Bacaan:** split acak memberi macro-F1 ≈ 0,99, sedangkan LOSO hanya ≈ 0,5–0,6. Selisih ini adalah
*data leakage*. Angka ≈ 99% hanya bisa kami reproduksi dengan split baris acak.
"""),
    md("""
## 5. Ensemble learning — model tunggal vs ensemble (LOSO)
Dua tugas: 4 kelas TVC dan **biner segar (excellent+good) vs tidak segar (acceptable+spoiled)**.
Tugas biner sama dengan label *Segar / Tidak segar* di skripsi Erlangga.
"""),
    code('''
binary = enose.assign(y=(enose["y"] >= 2).astype(int))
for d in [enose, binary]:
    for name, m in {**single_classifiers(), **ensemble_classifiers()}.items():
        run("E2_models", name, m, d=d)
'''),
    md("## 6. Imbalanced data — class weight, SMOTE, undersampling (hanya pada fold latih)"),
    code('''
RESAMPLERS = {"SMOTE": SMOTE(random_state=SEED), "RandomUnder": RandomUnderSampler(random_state=SEED)}
for name in ["LogReg", "LightGBM"]:
    for cw in [None, "balanced"]:
        run("E3_imbalance", name, {**single_classifiers(cw), **ensemble_classifiers(cw)}[name],
            label=f"class_weight={cw}")
    for rname, rs in RESAMPLERS.items():
        run("E3_imbalance", name, {**single_classifiers(), **ensemble_classifiers()}[name],
            resampler=rs, label=rname)
'''),
    md("## 7. Data augmentation deret waktu (hanya data latih, data uji tetap asli)"),
    code('''
AUGS = {"jitter": dict(copies=2, jitter=0.05, scale=0.0),
        "scaling": dict(copies=2, jitter=0.0, scale=0.10),
        "jitter+scaling": dict(copies=2, jitter=0.05, scale=0.10),
        "jitter+scaling x4": dict(copies=4, jitter=0.05, scale=0.10)}
for name in ["LogReg", "LightGBM"]:
    m = {**single_classifiers(), **ensemble_classifiers()}[name]
    run("E4_augment", name, m, label="none")
    for aname, a in AUGS.items():
        run("E4_augment", name, m, augment=a, label=aname)
summary = pd.DataFrame(SUMMARY); summary.to_csv(RESULTS / "enose_summary.csv", index=False)
'''),
    code('''
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
s = summary[(summary.experiment == "E2_models") & (summary.task == "binary")]
ax[0].barh(s.model, s.macro_f1_mean, xerr=s.macro_f1_sd, color=["#9aa5a6"] * 4 + ["#0f5257"] * 4)
ax[0].set_title("Biner segar vs tidak segar - LOSO macro-F1"); ax[0].set_xlim(0, 1)
a = summary[summary.experiment == "E4_augment"].pivot(index="variant", columns="model", values="macro_f1_mean")
a.plot.barh(ax=ax[1]); ax[1].set_title("Augmentasi - LOSO macro-F1 (4 kelas)"); plt.tight_layout(); plt.show()
'''),
    md("## 8. Interpretabilitas — sensor mana yang menentukan keputusan segar/tidak segar?"),
    code('''
import shap
X = build_features(enose); yb = binary["y"].to_numpy(); groups = enose["series"].to_numpy()
parts = []
for tr, te in LeaveOneGroupOut().split(X, yb, groups):
    m = clone(ensemble_classifiers()["XGBoost"]).fit(X.iloc[tr], yb[tr])
    parts.append(pd.Series(np.abs(shap.TreeExplainer(m).shap_values(X.iloc[te])).mean(0), index=X.columns))
feat = pd.concat(parts, axis=1).mean(1)
sensor = feat.groupby(feat.index.str.split("_").str[0]).sum().sort_values()
sensor.plot.barh(title="Mean |SHAP| per sensor (XGBoost, dijelaskan pada seri uji LOSO)", figsize=(7, 4)); plt.show()
sensor.sort_values(ascending=False).round(3)
'''),
    md("""
## 9. Data tambahan: lebih banyak *batch* atau lebih banyak baris?
Pertanyaan ini menjawab bagian rubrik *"bagaimana tambahan data meningkatkan performa"*,
dan langsung relevan dengan rancangan 16 batch di skripsi Erlangga.
"""),
    code('''
from itertools import combinations
rng, rows, names = np.random.default_rng(SEED), [], sorted(enose.series.unique())
xgb = ensemble_classifiers()["XGBoost"]; series = enose.series.to_numpy()
for test_s in names:
    te = series == test_s; others = [s for s in names if s != test_s]; all_tr = np.flatnonzero(~te)
    for k in range(1, 5):
        for combo in combinations(others, k):
            tr = np.flatnonzero(np.isin(series, combo))
            p = clone(xgb).fit(X.iloc[tr], yb[tr]).predict(X[te])
            rows.append({"design": "lebih banyak seri", "n_series": k, "n_rows": len(tr),
                         "macro_f1": f1_score(yb[te], p, average="macro")})
    for n in [40, 160, 640]:
        tr = rng.choice(all_tr, n, replace=False)
        p = clone(xgb).fit(X.iloc[tr], yb[tr]).predict(X[te])
        rows.append({"design": f"{n} baris dari 4 seri", "n_series": 4, "n_rows": n,
                     "macro_f1": f1_score(yb[te], p, average="macro")})
curve = pd.DataFrame(rows)
curve.groupby(["design", "n_series"]).macro_f1.agg(["mean", "std", "count"]).round(3)
'''),
    md("""
**Bacaan:** macro-F1 naik dari ≈0,72 (1 seri) ke ≈0,91 (4 seri). Sebaliknya, 160 baris yang diambil
dari 4 seri saja sudah ≈0,87. **Menambah batch lebih berharga daripada menambah baris.**
"""),
]

# ------------------------------------------------------------------ notebook 02
nb2 = [
    md("""
# 02 · Track B — Kelarutan Air Senyawa Obat (AqSolDB)

Track ini sesuai dengan topik skripsi Michael: gradient tree-boosting, Optuna, dan SHAP pada AqSolDB
(9.982 senyawa), dengan **scaffold-based splitting**.

| Aspek | Bagian |
|---|---|
| Wrangling | §1 deduplikasi, scaffold Bemis–Murcko, 207 deskriptor RDKit-2D |
| Validasi | §2 split acak vs scaffold |
| Ensemble | §3 model tunggal vs RF/XGBoost/LightGBM/Stacking |
| Optimasi Bayesian | §4 Optuna (TPE) dengan CV berkelompok scaffold |
| Imbalanced regression | §5 bobot frekuensi-invers per kategori kelarutan |
| Interpretabilitas | §6 SHAP |
"""),
    md("## 0. Instalasi & utilitas"), INSTALL, COMMON,
    md("## 1. Akuisisi & wrangling"),
    code('''
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
RDLogger.DisableLog("rdApp.*")

# AqSolDB (Sorkun dkk., 2019, Sci. Data 6:143) - mirror Hugging Face maomlab/AqSolDB (MIT)
SHA = {"train.csv": "b9572e57235f1cfb826f899d640e71617c1ad6014c3e38450c21282ca1962463",
       "test.csv": "e20826b231945ea9e02a7e515c0f56e2f0f98b9c97110c2ed49911bf21550f32"}
parts = []
for f, h in SHA.items():
    p = fetch(f"https://huggingface.co/datasets/maomlab/AqSolDB/resolve/main/AqSolDB/{f}", RAW / f"aqsoldb_{f}")
    assert sha256(p) == h, f"checksum {f} tidak cocok"
    parts.append(pd.read_csv(p))
df = pd.concat(parts, ignore_index=True).drop_duplicates("InChIKey").rename(columns={"Y": "logS"}).reset_index(drop=True)
BINS = [-np.inf, -6, -4, -2, 0, np.inf]
BIN_LABELS = ["very_insoluble(<-6)", "insoluble(-6..-4)", "slightly(-4..-2)", "soluble(-2..0)", "highly(>=0)"]
df["sol_bin"] = pd.cut(df["logS"], BINS, labels=BIN_LABELS, right=False)

def scaffold_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return f"INVALID:{smi}"
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception: return f"INVALID:{smi}"
df["scaffold"] = df["SMILES"].map(scaffold_of)
empty = df["scaffold"] == ""                     # molekul asiklik: masing-masing grup sendiri
df.loc[empty, "scaffold"] = "ACYCLIC:" + df.loc[empty, "InChIKey"]

rows = []
for smi in df["SMILES"]:
    mol = Chem.MolFromSmiles(smi)
    rows.append([fn(mol) if mol is not None else np.nan for _, fn in Descriptors.descList])
X_rd = pd.DataFrame(rows, columns=[n for n, _ in Descriptors.descList]).replace([np.inf, -np.inf], np.nan)
X_rd = X_rd.loc[:, X_rd.isna().mean() < 0.01].clip(-1e6, 1e6)
X_rd = X_rd.loc[:, X_rd.std() > 0].astype("float32")
AQSOL_DESC = ["MolWt", "MolLogP", "MolMR", "HeavyAtomCount", "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
              "NumRotatableBonds", "NumValenceElectrons", "NumAromaticRings", "NumSaturatedRings",
              "NumAliphaticRings", "RingCount", "TPSA", "LabuteASA", "BalabanJ", "BertzCT"]
X_aq = df[AQSOL_DESC].astype("float32"); y = df["logS"].to_numpy()
print(f"n={len(df)}  scaffold unik={df.scaffold.nunique()}  fitur RDKit-2D={X_rd.shape[1]}")
df.sol_bin.value_counts().sort_index()
'''),
    md("""
## 2. Scaffold split vs split acak
Molekul dengan scaffold yang sama mirip bagi model. Split acak membocorkan "kimia" dari data latih ke
data uji. Pada scaffold split, grup scaffold besar masuk ke data latih dan scaffold langka mengisi data uji,
sehingga data uji berisi **kimia baru**. Ini skenario yang paling sulit sekaligus paling realistis.
"""),
    code('''
def scaffold_split(d, test_frac=0.2, seed=SEED):
    groups = defaultdict(list)
    for i, s in enumerate(d["scaffold"]): groups[s].append(i)
    keys = list(groups); np.random.default_rng(seed).shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)
    n_train, train, test = int(round((1 - test_frac) * len(d))), [], []
    for k in keys:
        (train if len(train) + len(groups[k]) <= n_train else test).extend(groups[k])
    return np.array(sorted(train)), np.array(sorted(test))

tr, te = scaffold_split(df)
perm = np.random.default_rng(SEED).permutation(len(df)); k = int(round(.2 * len(df)))
rtr, rte = np.sort(perm[k:]), np.sort(perm[:k])
print("scaffold uji yang juga ada di latih:", df.scaffold.iloc[te].isin(df.scaffold.iloc[tr]).sum())
inner = list(GroupKFold(5).split(tr, groups=df["scaffold"].iloc[tr].to_numpy()))

def reg_scores(t, p):
    return {"rmse": float(np.sqrt(mean_squared_error(t, p))), "mae": float(mean_absolute_error(t, p)),
            "r2": float(r2_score(t, p))}

SUMMARY = []
def run(tag, name, model, X, split="scaffold", label="", sample_weight=None):
    a, b = (tr, te) if split == "scaffold" else (rtr, rte)
    m = clone(model)
    if hasattr(m, "cv") and split == "scaffold": m.set_params(cv=inner)
    kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
    p = m.fit(X.iloc[a], y[a], **kw).predict(X.iloc[b]); s = reg_scores(y[b], p)
    SUMMARY.append({"experiment": tag, "model": name, "split": split, "variant": label, **s})
    print(f"{tag:12s} {name:16s} {split:8s} {label:18s} RMSE={s['rmse']:.3f}  R2={s['r2']:.3f}")
    return m, p

imp = lambda: SimpleImputer(strategy="median")
SINGLE = {"Ridge": make_pipeline(imp(), StandardScaler(), Ridge(alpha=1.0)),
          "KNN": make_pipeline(imp(), StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
          "SVR-RBF": make_pipeline(imp(), StandardScaler(), SVR(C=10, gamma="scale")),
          "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=SEED)}
ENSEMBLE = {"RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
            "XGBoost": XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, n_jobs=4, random_state=SEED, verbosity=0),
            "LightGBM": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                          random_state=SEED, verbose=-1, n_jobs=4),
            "Stacking": StackingRegressor(
                estimators=[("ridge", make_pipeline(imp(), StandardScaler(), Ridge(alpha=1.0))),
                            ("svr", make_pipeline(imp(), StandardScaler(), SVR(C=10, gamma="scale"))),
                            ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                         random_state=SEED)),
                            ("lgbm", lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                                                       random_state=SEED, verbose=-1, n_jobs=4))],
                final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)}

for split in ["random", "scaffold"]:
    for fname, X in [("AqSolDB-17", X_aq), ("RDKit-2D", X_rd)]:
        run("S1_split", "LightGBM", ENSEMBLE["LightGBM"], X, split, fname)
        run("S1_split", "RandomForest", ENSEMBLE["RandomForest"], X, split, fname)
'''),
    md("## 3. Ensemble learning — model tunggal vs ensemble (scaffold split, RDKit-2D)"),
    code('''
preds = {}
for name, m in {**SINGLE, **ENSEMBLE}.items():
    _, preds[name] = run("S2_models", name, m, X_rd, label="RDKit-2D")
s = pd.DataFrame(SUMMARY).query("experiment == 'S2_models'")
s.plot.barh(x="model", y="rmse", legend=False, title="RMSE uji scaffold (lebih kecil lebih baik)",
            color=["#9aa5a6"] * 4 + ["#0f5257"] * 4, figsize=(7, 4)); plt.show()
'''),
    md("""
## 4. Optimasi Bayesian dengan Optuna (TPE)
CV di dalam data latih dikelompokkan per scaffold. Data uji tidak disentuh selama tuning.
`N_TRIALS = 40` sama dengan laporan; turunkan kalau waktu Colab terbatas.
"""),
    code('''
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = 40
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
    errs = [np.sqrt(np.mean((lgb.LGBMRegressor(**params).fit(Xtr.iloc[a], ytr[a]).predict(Xtr.iloc[b]) - ytr[b]) ** 2))
            for a, b in inner]
    return float(np.mean(errs))
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print("CV RMSE terbaik:", round(study.best_value, 3), study.best_params)
best = lgb.LGBMRegressor(**study.best_params, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=4)
tuned, p_tuned = run("S3_optuna", "LightGBM-Optuna", best, X_rd, label=f"{N_TRIALS} trials TPE")
'''),
    md("""
**Bacaan:** CV RMSE hasil tuning (≈1,06) jauh lebih baik daripada RMSE pada data uji (≈1,26).
Data uji berisi scaffold langka, jadi distribusinya bergeser dari fold CV. Tuning tidak bisa menutup
*distribution shift*.
"""),
    md("## 5. Imbalanced regression — bobot frekuensi-invers per kategori kelarutan"),
    code('''
bins_tr = df["sol_bin"].iloc[tr].astype(str)
w = 1.0 / bins_tr.map(bins_tr.value_counts(normalize=True)).astype(float).to_numpy(); w = w / w.mean()
_, p_w = run("S4_imbalance", "LightGBM-Optuna", best, X_rd, label="inv-freq weights", sample_weight=w)
t = df.iloc[te].assign(err_u=np.abs(y[te] - p_tuned), err_w=np.abs(y[te] - p_w))
per_bin = t.groupby("sol_bin", observed=False).agg(n=("err_u", "size"), mae_unweighted=("err_u", "mean"),
                                                   mae_weighted=("err_w", "mean"))
display(per_bin.round(3)); per_bin[["mae_unweighted", "mae_weighted"]].plot.bar(title="MAE per kategori"); plt.show()
pd.DataFrame(SUMMARY).to_csv(RESULTS / "solubility_summary.csv", index=False)
'''),
    md("## 6. Interpretabilitas — SHAP pada model hasil tuning"),
    code('''
import shap
sample = X_rd.iloc[te].sample(1000, random_state=SEED)
sv = shap.TreeExplainer(tuned).shap_values(sample)
shap.summary_plot(sv, sample, max_display=12, show=True)
'''),
]

# ------------------------------------------------------------------ notebook 03
nb3 = [
    md("""
# 03 · Aspek *Reinforcement* — *Active Learning* yang Dikendalikan *Multi-Armed Bandit*

Label mahal di ketiga topik: uji lab kelarutan, telur yang harus dipecah, serta panel sensori dan
penyimpanan 48 jam. Setiap ronde, **agen** memilih satu *arm* (strategi kueri): `random`,
`uncertainty`, atau `diversity`. *Oracle* memberi label pada satu batch, model dilatih ulang, dan
**reward** = perbaikan relatif skor **validasi**. Data uji hanya dipakai untuk pelaporan.

- **ε-greedy + estimasi Monte Carlo:** Q(a) ← Q(a) + (r − Q(a)) / n(a)
- **UCB1:** pilih argmax Q(a) + c·√(2 ln N / n(a))

Gagasan ini mengikuti *Active Learning by Learning* (Hsu & Lin, AAAI 2015), dengan kebijakan bandit dari materi kuliah.
"""),
    md("## 0. Instalasi & utilitas"), INSTALL, COMMON,
    md("## 1. Agen bandit & strategi kueri"),
    code('''
from sklearn.metrics import f1_score, mean_squared_error
ARMS = ["random", "uncertainty", "diversity"]

def q_random(ctx, k, rng): return rng.choice(ctx["pool"], size=k, replace=False)
def q_uncertainty(ctx, k, rng):
    u = ctx["uncertainty"](ctx["model"], ctx["X"][ctx["pool"]]); return ctx["pool"][np.argsort(-u)[:k]]
def q_diversity(ctx, k, rng):
    """k-center greedy: ambil titik pool yang paling jauh dari semua titik berlabel."""
    Z, pool = ctx["Z"], ctx["pool"]; d = ctx["min_dist"][pool].copy(); chosen = []
    for _ in range(k):
        j = int(np.argmax(d)); chosen.append(pool[j])
        d = np.minimum(d, np.linalg.norm(Z[pool] - Z[pool[j]], axis=1))
    return np.array(chosen)
QUERY = {"random": q_random, "uncertainty": q_uncertainty, "diversity": q_diversity}

def forest_std(model, X): return np.stack([t.predict(X) for t in model.estimators_]).std(axis=0)
def proba_entropy(model, X):
    p = np.clip(model.predict_proba(X), 1e-12, 1); return -(p * np.log(p)).sum(axis=1)

class Fixed:
    def __init__(self, arm): self.arm = arm
    def select(self, t, rng): return self.arm
    def update(self, arm, r): pass
class EpsilonGreedy:
    def __init__(self, arms, epsilon=0.2):
        self.arms, self.eps = arms, epsilon; self.n = {a: 0 for a in arms}; self.q = {a: 0.0 for a in arms}
    def select(self, t, rng):
        untried = [a for a in self.arms if self.n[a] == 0]
        if untried: return untried[0]
        if rng.random() < self.eps: return self.arms[rng.integers(len(self.arms))]
        return max(self.arms, key=self.q.get)
    def update(self, arm, r):
        self.n[arm] += 1; self.q[arm] += (r - self.q[arm]) / self.n[arm]
class UCB1(EpsilonGreedy):
    def __init__(self, arms, c=1.0): super().__init__(arms); self.c = c
    def select(self, t, rng):
        untried = [a for a in self.arms if self.n[a] == 0]
        if untried: return untried[0]
        N = sum(self.n.values())
        return max(self.arms, key=lambda a: self.q[a] + self.c * np.sqrt(2 * np.log(N) / self.n[a]))
def make_policy(name):
    return Fixed(name) if name in ARMS else EpsilonGreedy(ARMS, 0.2) if name == "bandit-eps" else UCB1(ARMS, 1.0)

def run_active_learning(model, Xp, yp, Xv, yv, Xt, yt, *, policy, score, uncertainty, n_init, batch,
                        rounds, seed, higher_is_better=True, stratify_init=False):
    rng = np.random.default_rng(seed); Xp = np.asarray(Xp); Z = StandardScaler().fit_transform(Xp); n = len(Xp)
    if stratify_init:
        lab = [rng.choice(np.flatnonzero(yp == c)) for c in np.unique(yp)]
        lab += list(rng.choice(np.setdiff1d(np.arange(n), lab), n_init - len(lab), replace=False)); lab = np.array(lab)
    else:
        lab = rng.choice(n, n_init, replace=False)
    mask = np.ones(n, bool); mask[lab] = False
    min_dist = np.full(n, np.inf)
    for i in lab: min_dist = np.minimum(min_dist, np.linalg.norm(Z - Z[i], axis=1))
    pol, sign = make_policy(policy), (1 if higher_is_better else -1)
    def fit_eval(idx):
        m = clone(model).fit(Xp[idx], yp[idx]); return m, score(yv, m.predict(Xv)), score(yt, m.predict(Xt))
    m, v, te_s = fit_eval(lab)
    log = [{"round": 0, "n_labelled": len(lab), "arm": "init", "val": v, "test": te_s, "reward": 0.0}]
    for t in range(1, rounds + 1):
        arm = pol.select(t, rng)
        ctx = {"pool": np.flatnonzero(mask), "X": Xp, "Z": Z, "model": m, "uncertainty": uncertainty, "min_dist": min_dist}
        new = QUERY[arm](ctx, min(batch, mask.sum()), rng); mask[new] = False
        for i in new: min_dist = np.minimum(min_dist, np.linalg.norm(Z - Z[i], axis=1))
        lab = np.concatenate([lab, new]); m, v_new, te_s = fit_eval(lab)
        r = sign * (v_new - v) / (abs(v) + 1e-9); pol.update(arm, r); v = v_new
        log.append({"round": t, "n_labelled": len(lab), "arm": arm, "val": v, "test": te_s, "reward": r})
    return log
POLICIES = ["random", "uncertainty", "diversity", "bandit-eps", "bandit-ucb"]
'''),
    md("""
## 2. AL pada kelarutan (AqSolDB)
*Pool* = data latih scaffold, validasi = scaffold split 15% di dalam data latih, uji = scaffold uji.
Mulai dari 100 label, lalu +100 per ronde selama 20 ronde (total 2.100 label), dengan 5 seed.
"""),
    code('''
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
parts = [pd.read_csv(fetch(f"https://huggingface.co/datasets/maomlab/AqSolDB/resolve/main/AqSolDB/{f}", RAW / f"aqsoldb_{f}"))
         for f in ["train.csv", "test.csv"]]
df = pd.concat(parts, ignore_index=True).drop_duplicates("InChIKey").rename(columns={"Y": "logS"}).reset_index(drop=True)
def scaffold_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return f"INVALID:{smi}"
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception: return f"INVALID:{smi}"
df["scaffold"] = df["SMILES"].map(scaffold_of)
empty = df["scaffold"] == ""; df.loc[empty, "scaffold"] = "ACYCLIC:" + df.loc[empty, "InChIKey"]
rows = []
for smi in df["SMILES"]:
    mol = Chem.MolFromSmiles(smi); rows.append([fn(mol) if mol is not None else np.nan for _, fn in Descriptors.descList])
X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
X = X.loc[:, X.isna().mean() < 0.01].clip(-1e6, 1e6); X = X.loc[:, X.std() > 0].astype("float32").to_numpy()
y = df["logS"].to_numpy()
def scaffold_split(d, test_frac=0.2, seed=SEED):
    groups = defaultdict(list)
    for i, s in enumerate(d["scaffold"]): groups[s].append(i)
    keys = list(groups); np.random.default_rng(seed).shuffle(keys); keys.sort(key=lambda k: len(groups[k]), reverse=True)
    n_train, a, b = int(round((1 - test_frac) * len(d))), [], []
    for k in keys: (a if len(a) + len(groups[k]) <= n_train else b).extend(groups[k])
    return np.array(sorted(a)), np.array(sorted(b))
tr, te = scaffold_split(df)
ptr, pval = scaffold_split(df.iloc[tr].reset_index(drop=True), test_frac=0.15); pool, val = tr[ptr], tr[pval]
X = np.where(np.isnan(X), np.nanmedian(X[pool], axis=0), X)          # imputasi median dari pool saja
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=SEED)
full = rmse(y[te], clone(rf).fit(X[pool], y[pool]).predict(X[te]))
print(f"pool={len(pool)} val={len(val)} test={len(te)}  RMSE jika SELURUH pool dilabeli = {full:.3f}")
logs = []
for seed in range(5):
    for pol in POLICIES:
        log = run_active_learning(rf, X[pool], y[pool], X[val], y[val], X[te], y[te], policy=pol, score=rmse,
                                  uncertainty=forest_std, n_init=100, batch=100, rounds=20, seed=seed,
                                  higher_is_better=False)
        logs += [dict(r, policy=pol, seed=seed) for r in log]
        print(f"seed={seed} {pol:12s} RMSE akhir={log[-1]['test']:.3f}")
al_sol = pd.DataFrame(logs); al_sol.to_csv(RESULTS / "al_solubility.csv", index=False)
'''),
    code('''
def plot_curves(al, ylabel, title, ref=None):
    c = al.groupby(["policy", "n_labelled"]).test.mean().unstack(0)
    ax = c.plot(figsize=(8, 4.5), marker="o", ms=3, title=title); ax.set_xlabel("jumlah label"); ax.set_ylabel(ylabel)
    if ref is not None: ax.axhline(ref, ls="--", c="k", lw=.8, label="seluruh pool")
    ax.legend(); plt.show(); return c
c = plot_curves(al_sol, "RMSE uji (scaffold)", "Kurva belajar AL - kelarutan", full)
arms = al_sol[al_sol.policy.str.startswith("bandit") & (al_sol.arm != "init")]
display(arms.groupby("policy").arm.value_counts(normalize=True).unstack().round(2))
target = c["random"].iloc[-1]
print("label yang dibutuhkan untuk menyamai RMSE akhir random:",
      {p: next((int(n) for n, s in c[p].items() if s <= target), None) for p in c})
'''),
    md("""
## 3. AL pada *e-nose* (biner segar vs tidak segar)
Untuk setiap seri uji, satu seri lain menjadi validasi dan tiga sisanya menjadi *pool*.
Mulai dari 40 label, lalu +40 per ronde selama 15 ronde, dengan 3 seed.
"""),
    ENOSE_DATA, ENOSE_FEATURES,
    code('''
Xe = build_features(enose).to_numpy(); ye = (enose["y"].to_numpy() >= 2).astype(int)
series = enose.series.to_numpy(); names = sorted(np.unique(series))
xgb = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                    n_jobs=4, random_state=SEED, verbosity=0)
f1 = lambda a, b: float(f1_score(a, b, average="macro"))
logs = []
for i, test_s in enumerate(names):
    val_s = names[(i + 1) % 5]
    te_i, val_i = np.flatnonzero(series == test_s), np.flatnonzero(series == val_s)
    pool_i = np.flatnonzero((series != test_s) & (series != val_s))
    full = f1(ye[te_i], clone(xgb).fit(Xe[pool_i], ye[pool_i]).predict(Xe[te_i]))
    for seed in range(3):
        for pol in POLICIES:
            log = run_active_learning(xgb, Xe[pool_i], ye[pool_i], Xe[val_i], ye[val_i], Xe[te_i], ye[te_i],
                                      policy=pol, score=f1, uncertainty=proba_entropy, n_init=40, batch=40,
                                      rounds=15, seed=seed, stratify_init=True)
            logs += [dict(r, policy=pol, seed=seed, test_series=test_s, full_pool_test=full) for r in log]
    print(test_s, "selesai; F1 seluruh pool =", round(full, 3))
al_en = pd.DataFrame(logs); al_en.to_csv(RESULTS / "al_enose.csv", index=False)
plot_curves(al_en, "macro-F1 uji (seri baru)", "Kurva belajar AL - e-nose biner", al_en.full_pool_test.mean());
'''),
    md("""
**Bacaan:** pada *e-nose*, 40 label sudah memberi F1 ≈ 0,83 dan kurvanya hampir datar. Kendalanya
**bukan jumlah label, melainkan pergeseran domain antar seri** (lihat notebook 01 §9). Strategi pintar
yang memilih titik "aneh" dari seri yang sama tidak membantu, dan *random* justru paling aman. Bandit
dapat membatasi kerugian dengan memilih arm yang terbukti baik pada validasi, tetapi tidak dapat
menciptakan informasi yang tidak ada di dalam *pool*.
"""),
]

# ------------------------------------------------------------------ notebook 04
nb4 = [
    md("""
# 04 · Track C — Mutu Internal Kuning Telur Puyuh (*slot*, menunggu data)

Topik skripsi Rijal: EfficientNet-B0 sebagai *feature extractor* + XGBoost (hybrid) dengan label
**Yolk Index** (tinggi/diameter kuning telur). Setiap telur adalah **sampel destruktif**. Rencananya
2 suhu × 31 hari × 6 butir = 372 butir.

**Status:** citra belum tersedia di repo. Notebook ini sudah memuat seluruh alur:
1. ekstraksi *embedding* EfficientNet-B0 (1.280 dimensi),
2. model tunggal vs hybrid XGBoost, dengan dan tanpa SMOTE,
3. *active learning* bandit yang sama seperti notebook 03.

Selama data belum ada, `SMOKE_TEST = True` menjalankan alur dengan **data sintetis**. Angka dari mode itu
**bukan hasil penelitian** dan tidak boleh dikutip.

Format data yang dibutuhkan (taruh di Google Drive lalu ubah `EGG_DIR`):
```
EGG_DIR/labels.csv     image,yolk_index,storage_day,temperature
EGG_DIR/images/<image>
```
Ambang kelas YI harus ditetapkan dalam protokol Rijal **sebelum** pemodelan.
"""),
    md("## 0. Instalasi & utilitas"), INSTALL, COMMON,
    code('''
SMOKE_TEST = True                     # ubah ke False setelah data Rijal tersedia
EGG_DIR = Path("/content/drive/MyDrive/egg_yolk")
YI_THRESHOLDS = None                  # contoh: (lo, hi) dari protokol Rijal - JANGAN ditebak
'''),
    md("## 1. Ekstraksi embedding EfficientNet-B0 (hanya jika data nyata tersedia)"),
    code('''
if not SMOKE_TEST:
    assert YI_THRESHOLDS is not None, "isi ambang Yolk Index dari protokol penelitian"
    import torch
    from PIL import Image
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    w = EfficientNet_B0_Weights.IMAGENET1K_V1; net = efficientnet_b0(weights=w)
    net.classifier = torch.nn.Identity(); net.eval(); prep = w.transforms()
    labels = pd.read_csv(EGG_DIR / "labels.csv"); lo, hi = YI_THRESHOLDS
    labels["y"] = np.select([labels.yolk_index < lo, labels.yolk_index < hi], [0, 1], 2)
    with torch.no_grad():
        F = np.vstack([net(prep(Image.open(EGG_DIR / "images" / n).convert("RGB")).unsqueeze(0)).squeeze(0).numpy()
                       for n in labels["image"]])
    X, y = F, labels["y"].to_numpy()
else:
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=372, n_features=64, n_informative=12, n_classes=3,
                               weights=[0.2, 0.5, 0.3], random_state=SEED)
print(X.shape, np.bincount(y))
'''),
    md("## 2. Model tunggal vs hybrid XGBoost, ± SMOTE (Stratified 5-fold)"),
    code('''
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
models = {"LogReg (single)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
          "XGBoost (hybrid)": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                            colsample_bytree=0.5, n_jobs=4, random_state=SEED, verbosity=0)}
rows = []
for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
    for name, m in models.items():
        for smote in [False, True]:
            Xtr, ytr = SMOTE(random_state=SEED).fit_resample(X[tr], y[tr]) if smote else (X[tr], y[tr])
            rows.append({"model": name, "smote": smote,
                         "macro_f1": f1_score(y[te], clone(m).fit(Xtr, ytr).predict(X[te]), average="macro")})
pd.DataFrame(rows).groupby(["model", "smote"]).macro_f1.agg(["mean", "std"]).round(3)
'''),
    md("""
## 3. *Active learning* bandit
Fungsi `run_active_learning` sama dengan notebook 03. Salin sel §1 dari notebook 03 ke sini setelah data
tersedia. Untuk telur, setiap label = satu butir yang dipecah, sehingga penghematan label berarti
penghematan sampel destruktif.
"""),
]

for name, cells in [("01_track_A_enose_kesegaran.ipynb", nb1), ("02_track_B_kelarutan_aqsoldb.ipynb", nb2),
                    ("03_reinforcement_active_learning.ipynb", nb3), ("04_track_C_telur_puyuh_slot.ipynb", nb4)]:
    write(name, cells)
