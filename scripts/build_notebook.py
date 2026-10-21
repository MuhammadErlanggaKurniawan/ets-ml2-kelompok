"""Generate notebooks/ETS_ML2_Pipeline_Lengkap.ipynb, the single end-to-end Colab notebook.

Edit the cells here, then run:  python3 scripts/build_notebook.py
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "MuhammadErlanggaKurniawan/ets-ml2-kelompok"
NAME = "ETS_ML2_Pipeline_Lengkap.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n")})


# =====================================================================================
md(rf"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{NAME})

# Prediksi Kesegaran Daging dan Kelarutan Senyawa dengan Machine Learning
### Proyek ETS Machine Learning II — satu notebook, dari awal sampai akhir

**Kelompok:**
- Muhammad Erlangga Kurniawan (23083010045)
- Michael Valencio Erasmo Bagas Ganendra (23083010042)
- Rijal Apuila Khufi (23083010033)

---

## Cerita proyek ini dalam 4 kalimat

1. Ketiga skripsi kami punya masalah yang sama: **data berlabel lama dan mahal didapat**.
   - Kesegaran makanan harus dinilai panel orang setelah disimpan sampai 48 jam.
   - Kelarutan senyawa obat harus diukur di laboratorium.
   - Mutu kuning telur hanya bisa diukur dengan memecah telurnya.
2. Jadi pertanyaan proyeknya: **bagaimana mendapat model yang jujur dan bagus dengan label sesedikit mungkin?**
3. Kami menjawabnya dengan satu pipeline yang sama, dijalankan pada dua dataset publik:
   - **e-nose daging sapi**, proksi topik Erlangga
   - **AqSolDB**, topik Michael
4. Pipeline ini menyentuh **4 aspek ML II**:
   - wrangling & augmentasi
   - imbalanced data
   - ensemble learning
   - reinforcement (agen *multi-armed bandit* yang memilih data mana yang perlu dilabeli)

## Peta notebook

| Bagian | Isi | Aspek ML II / rubrik |
|---|---|---|
| 1 | Persiapan: instal paket, pengaturan | — |
| 2 | **Akuisisi** data + cek keaslian file | pipeline: akuisisi |
| 3 | Kenali datanya (EDA) | latar belakang |
| 4 | **Data wrangling** | wrangling (15%) |
| 5 | **Validasi yang jujur**: temuan *data leakage* | analisis (25%) |
| 6 | **Ensemble learning** + Optuna | ensemble (15%) |
| 7 | **Imbalanced data** | aspek imbalance |
| 8 | **Data augmentation** | augmentasi (15%) |
| 9 | **Reinforcement**: *active learning* dengan agen bandit | reinforcement (15%) |
| 10 | Tambah *batch* vs tambah baris | reinforcement / analisis |
| 11 | Interpretasi model (SHAP) | analisis |
| 12 | (Opsional) slot topik Rijal: citra telur | — |
| 13 | **Ringkasan semua angka** + kesimpulan | presentasi (30%) |

## Cara menjalankan
Pilih menu **Runtime → Run all**. Semua sel berjalan berurutan dari atas ke bawah.
Mode `"lengkap"` memakai pengaturan yang sama dengan slide dan butuh waktu ±60–90 menit di CPU Colab.
Untuk demo cepat, ubah `MODE = "cepat"` di Bagian 1 (±20–30 menit). Angkanya akan sedikit berbeda dari slide.
""")

md(r"""
## Istilah yang sering muncul

| Istilah | Artinya (versi sederhana) |
|---|---|
| **Seri / batch** | Satu rangkaian pengukuran dari satu sampel. Di e-nose: satu potongan daging diukur selama 36 jam. Di skripsi Erlangga: `batch_id` |
| **Scaffold** | "Kerangka" molekul (cincin + penghubung). Molekul satu scaffold mirip satu sama lain |
| **Data leakage** | Informasi dari data uji "bocor" ke data latih, sehingga skor terlihat bagus padahal palsu |
| **LOSO** | *Leave-One-Series-Out*: latih di 4 seri, uji di 1 seri yang belum pernah dilihat, ulangi 5×. Kasus khusus GroupKFold |
| **Macro-F1** | Skor 0–1 untuk klasifikasi; rata-rata F1 per kelas, adil untuk kelas yang tidak seimbang. Makin besar makin baik |
| **RMSE** | Rata-rata besar galat prediksi untuk regresi (satuan log mol/L). Makin kecil makin baik |
| **Ensemble** | Menggabungkan banyak model. *Bagging* (Random Forest), *boosting* (XGBoost, LightGBM), *stacking* (meta-model) |
| **Active learning** | Model memilih sendiri data mana yang paling berguna untuk dilabeli berikutnya |
| **Multi-armed bandit** | Bentuk RL paling sederhana: agen memilih satu dari beberapa "tuas" lalu belajar dari reward-nya |
| **SHAP** | Cara mengukur seberapa besar tiap fitur berkontribusi pada prediksi |
""")

# =====================================================================================
md(r"""
---
# Bagian 1 — Persiapan

**Apa:** memasang paket dan menyiapkan fungsi bantu.

**Kenapa versi paketnya dikunci:** supaya angka yang keluar di Colab sama persis dengan angka di laporan dan slide.
Kalau versi scikit-learn atau XGBoost berbeda, hasilnya bisa bergeser sedikit.
""")
code(r"""
%pip install -q scikit-learn==1.9.1 xgboost==3.4.1 lightgbm==4.7.0 optuna==5.0.0 \
    imbalanced-learn==0.14.2 rdkit==2026.9.1 shap==0.54.1
""")
code(r'''
# ---------------- PENGATURAN ----------------
MODE = "lengkap"        # "lengkap" = sama dengan slide | "cepat" = demo, lebih sedikit trial/seed
JALANKAN_TRACK_C = False  # Bagian 12 (data telur Rijal) - aktifkan setelah datanya ada

CFG = {"lengkap": dict(optuna_trials=40, al_seeds_sol=5, al_seeds_enose=3),
       "cepat":   dict(optuna_trials=10, al_seeds_sol=2, al_seeds_enose=1)}[MODE]
print("Mode:", MODE, CFG)

# ---------------- IMPORT ----------------
import warnings, hashlib, json, time, urllib.request, zipfile, glob, ssl
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import certifi
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import clone
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, recall_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
SEED = 42
RAW, RESULTS = Path("data/raw"), Path("results")
RAW.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(exist_ok=True)
TEAL, GREY, AMBER, RED = "#0f5257", "#9aa5a6", "#e3a33b", "#c4553b"

# ---------------- FUNGSI BANTU ----------------
def sha256(path):
    """Sidik jari file: kalau isinya berubah 1 byte saja, hasilnya berbeda total."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, dest):
    """Unduh file sekali saja (kalau sudah ada, tidak diunduh ulang)."""
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

HASIL = {}   # semua angka penting dikumpulkan di sini, lalu diringkas di Bagian 13
''')

# =====================================================================================
md(r"""
---
# Bagian 2 — Akuisisi data

**Apa:** mengunduh dua dataset publik, lalu memastikan filenya asli dengan checksum SHA-256.

**Kenapa memakai dataset publik, bukan data skripsi kami sendiri:**
- **Erlangga:** data sensornya baru **pilot 6 baris dari 1 batch**. Validasi per batch mustahil dengan 1 batch,
  jadi angka akurasi dari sana tidak sah. Kami memakai **dataset e-nose daging sapi** (Wijaya dkk., 2018)
  sebagai proksi karena strukturnya sama: sensor gas MOS **termasuk MQ135**, kelembapan, suhu, dan
  **5 seri pengukuran** yang berperan seperti `batch_id`.
- **Michael:** **AqSolDB** (Sorkun dkk., 2019) memang data skripsinya, berisi 9.982 senyawa.
- **Rijal:** citra kuning telurnya belum diambil (rencana 372 butir). Kodenya sudah disiapkan di Bagian 12.

> 💡 **Intinya:** kami tidak mau melaporkan angka dari data yang belum cukup. Dataset proksi disebut
> terus terang sebagai proksi.
""")
code(r'''
# --- Dataset A: e-nose daging sapi (Mendeley Data, CC BY 4.0) ---
ENOSE_URL = ("https://data.mendeley.com/public-files/datasets/mwmhh766fc/files/"
             "2e65305f-5e53-4395-9b28-91dfc73ff490/file_downloaded")
z = fetch(ENOSE_URL, RAW / "enose_beef.zip")
assert sha256(z) == "772bc2c75f2ac4b785538ea38311bdefaee65059b3391b080b4da5ea619963e3", "file e-nose berubah/korup"
zipfile.ZipFile(z).extractall(RAW / "enose_beef")

# --- Dataset B: AqSolDB (mirror Hugging Face maomlab/AqSolDB, MIT) ---
AQ_SHA = {"train.csv": "b9572e57235f1cfb826f899d640e71617c1ad6014c3e38450c21282ca1962463",
          "test.csv":  "e20826b231945ea9e02a7e515c0f56e2f0f98b9c97110c2ed49911bf21550f32"}
for f, h in AQ_SHA.items():
    p = fetch(f"https://huggingface.co/datasets/maomlab/AqSolDB/resolve/main/AqSolDB/{f}", RAW / f"aqsoldb_{f}")
    assert sha256(p) == h, f"checksum {f} tidak cocok"
print("✅ Kedua dataset terunduh dan checksum-nya cocok.")
''')

# =====================================================================================
md(r"""
---
# Bagian 3 — Kenali datanya

## 3a. E-nose daging sapi
- Lima potongan daging (seri `TS1`–`TS5`) dipantau tiap menit selama 36 jam, jadi 2.160 menit per seri.
- Ada 9 sensor gas (MQ135, MQ136, MQ2, MQ3, MQ4, MQ5, MQ6, MQ8, MQ9), kelembapan, dan suhu.
- **Labelnya** berasal dari hitungan mikroba (TVC). Ada 4 kelas berurutan:
  `excellent` → `good` → `acceptable` → `spoiled`.
""")
code(r'''
CLASSES = ["excellent", "good", "acceptable", "spoiled"]
GAS = ["MQ135", "MQ136", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ8", "MQ9"]
PREDICTORS_RAW = GAS + ["Humidity"]

frames = []
for f in sorted(glob.glob(str(RAW / "enose_beef" / "TS*.csv"))):
    d = pd.read_csv(f); d["series"] = Path(f).stem; frames.append(d)
enose = pd.concat(frames, ignore_index=True)
enose["y"] = enose["class"].map({c: i for i, c in enumerate(CLASSES)})
print(f"{len(enose):,} baris · {enose.series.nunique()} seri"); display(enose.head(3))

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for s, g in enose.groupby("series"):
    ax[0].plot(g.minute / 60, g.MQ135, lw=.8, label=s); ax[1].plot(g.minute / 60, g.MQ4, lw=.8, label=s)
ax[0].set_title("Sensor MQ135 per seri"); ax[1].set_title("Sensor MQ4 per seri")
for a in ax[:2]: a.set_xlabel("jam ke-")
ax[0].legend(fontsize=8)
share = enose["class"].value_counts(normalize=True).reindex(CLASSES)
ax[2].bar(CLASSES, share, color=[TEAL, "#2e6f95", AMBER, RED]); ax[2].set_title("Proporsi kelas")
for i, v in enumerate(share): ax[2].text(i, v + .01, f"{v:.0%}", ha="center")
plt.tight_layout(); plt.show()
''')
md(r"""
**Cara membaca:**
- Tiap seri punya "level" sensor yang berbeda, karena potongan daging dan suhu ruangnya berbeda. Inilah
  sumber kesulitan utama nanti: model harus bisa menebak **seri yang belum pernah dilihat**.
- Kelas **tidak seimbang**: `spoiled` paling banyak, `excellent` paling sedikit. Ini ditangani di Bagian 7.

## 3b. AqSolDB
Setiap baris adalah satu senyawa: rumus SMILES dan nilai kelarutan **LogS** (log mol/L).
Makin negatif LogS, makin sulit larut.
""")
code(r'''
aq = pd.concat([pd.read_csv(RAW / f"aqsoldb_{f}") for f in AQ_SHA], ignore_index=True)
aq = aq.drop_duplicates("InChIKey").rename(columns={"Y": "logS"}).reset_index(drop=True)
BINS = [-np.inf, -6, -4, -2, 0, np.inf]
BIN_LABELS = ["very_insoluble(<-6)", "insoluble(-6..-4)", "slightly(-4..-2)", "soluble(-2..0)", "highly(>=0)"]
aq["sol_bin"] = pd.cut(aq["logS"], BINS, labels=BIN_LABELS, right=False)
print(f"{len(aq):,} senyawa unik"); display(aq[["Name", "SMILES", "logS"]].head(3))
ax = aq.logS.plot.hist(bins=60, color=TEAL, figsize=(8, 3.2), title="Sebaran LogS")
for b in [-6, -4, -2, 0]: ax.axvline(b, ls="--", c=GREY, lw=.8)
plt.show(); display(aq.sol_bin.value_counts().sort_index().rename("jumlah"))
''')
md(r"""
**Cara membaca:** kebanyakan senyawa berada di tengah (LogS −4 sampai 0). Senyawa yang **sangat tidak larut**
dan **sangat larut** lebih jarang. Ini "imbalance" versi regresi, dibahas di Bagian 7.
""")

# =====================================================================================
md(r"""
---
# Bagian 4 — Data wrangling

## 4a. E-nose: cek kualitas lalu buat fitur
**Cek kualitas:** apakah ada nilai hilang, menit ganda, dan berapa rentang suhunya.

**Fitur yang kita buat.** Semuanya **kausal**, artinya hanya memakai data masa lalu, jadi bisa dihitung
langsung di alat:

| Fitur | Rumus | Kenapa |
|---|---|---|
| `_ratio` | nilai sensor ÷ rata-rata 10 menit pertama | Menghapus perbedaan "level" antar sensor/seri; mirip Rs/R0 MQ-135 di skripsi Erlangga |
| `_ratio_roll` | rata-rata bergulir 15 menit dari rasio | Meredam derau |
| `_slope` | perubahan rasio dalam 15 menit | Menangkap *seberapa cepat* bau berubah |

**Suhu tidak dipakai sebagai input**, sama seperti peran variabel di skripsi Erlangga (suhu = metadata).
Pengaruhnya diuji terpisah.
""")
code(r'''
quality = pd.DataFrame([{"seri": s, "baris": len(g), "nilai_hilang": int(g[PREDICTORS_RAW].isna().sum().sum()),
                         "menit_ganda": int(g.minute.duplicated().sum()),
                         "suhu_°C": f"{g.Temperature.min():.1f}–{g.Temperature.max():.1f}",
                         **{c: int((g["class"] == c).sum()) for c in CLASSES}}
                        for s, g in enose.groupby("series")])
display(quality)

BASELINE_MIN, WINDOW = 10, 15
def engineer(g):
    g = g.sort_values("minute"); out = pd.DataFrame(index=g.index)
    for c in GAS:
        ratio = g[c] / g[c].iloc[:BASELINE_MIN].mean()
        out[f"{c}_ratio"] = ratio
        out[f"{c}_ratio_roll"] = ratio.rolling(WINDOW, min_periods=1).mean()
        out[f"{c}_slope"] = ratio.diff(WINDOW).fillna(0) / WINDOW
    out["Humidity"] = g["Humidity"]
    return out

def build_features(d, kind="engineered", with_temperature=False):
    X = d[PREDICTORS_RAW].copy() if kind == "raw" else \
        pd.concat([engineer(g) for _, g in d.groupby("series", sort=False)]).loc[d.index]
    if with_temperature: X["Temperature"] = d["Temperature"]
    return X

X_en = build_features(enose)
binary = enose.assign(y=(enose["y"] >= 2).astype(int))   # 0 = segar (excellent+good), 1 = tidak segar
print(f"Fitur e-nose: {X_en.shape[1]} kolom (9 sensor × 3 fitur + kelembapan)")
''')
md(r"""
Kita juga membuat **tugas biner**: segar (`excellent`+`good`) vs tidak segar (`acceptable`+`spoiled`).
Ini sama dengan label *Segar / Tidak segar* di skripsi Erlangga.

## 4b. AqSolDB: scaffold dan deskriptor molekul
1. **Scaffold Bemis–Murcko** dihitung untuk setiap molekul, untuk membagi data secara jujur di Bagian 5.
2. **207 deskriptor RDKit-2D** dihitung, misalnya berat molekul, logP, dan luas permukaan polar.
3. Kolom yang >1% kosong atau nilainya tidak bervariasi dibuang, lalu nilai ekstrem dipotong.
4. Sisa nilai kosong **diisi median**, tetapi nanti di dalam pipeline model, supaya mediannya dihitung dari
   data latih saja.
""")
code(r'''
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

def scaffold_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return f"INVALID:{smi}"
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception: return f"INVALID:{smi}"
aq["scaffold"] = aq["SMILES"].map(scaffold_of)
empty = aq["scaffold"] == ""                                   # molekul tanpa cincin: grup sendiri-sendiri
aq.loc[empty, "scaffold"] = "ACYCLIC:" + aq.loc[empty, "InChIKey"]

t0 = time.time(); rows = []
for smi in aq["SMILES"]:
    mol = Chem.MolFromSmiles(smi)
    rows.append([fn(mol) if mol is not None else np.nan for _, fn in Descriptors.descList])
X_aq = pd.DataFrame(rows, columns=[n for n, _ in Descriptors.descList]).replace([np.inf, -np.inf], np.nan)
X_aq = X_aq.loc[:, X_aq.isna().mean() < 0.01].clip(-1e6, 1e6)
X_aq = X_aq.loc[:, X_aq.std() > 0].astype("float32")
y_aq = aq["logS"].to_numpy()
AQSOL_17 = ["MolWt", "MolLogP", "MolMR", "HeavyAtomCount", "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
            "NumRotatableBonds", "NumValenceElectrons", "NumAromaticRings", "NumSaturatedRings",
            "NumAliphaticRings", "RingCount", "TPSA", "LabuteASA", "BalabanJ", "BertzCT"]
X_aq17 = aq[AQSOL_17].astype("float32")
print(f"{aq.scaffold.nunique():,} scaffold unik · {X_aq.shape[1]} deskriptor RDKit-2D · {time.time() - t0:.0f} detik")
''')

# =====================================================================================
md(r"""
---
# Bagian 5 — Validasi yang jujur (Temuan 1: *data leakage*)

**Analogi:** kalau soal ujian diambil dari latihan yang sudah dihafal, nilainya tinggi tetapi tidak menunjukkan
kemampuan sebenarnya.

**Dua cara membagi data yang kita bandingkan:**

| | Split acak (bocor) | Split jujur |
|---|---|---|
| E-nose | baris diacak, jadi menit ke-100 di latih dan menit ke-101 di uji | **LOSO**: latih di 4 seri, uji di 1 seri baru |
| AqSolDB | molekul diacak | **Scaffold split**: kerangka molekul di data uji **tidak pernah** muncul di data latih |

**Aturan emas notebook ini:** apa pun yang "belajar dari data" (imputasi, scaling, SMOTE, augmentasi, tuning)
hanya boleh di-fit pada **data latih**.

Sel berikut mendefinisikan model dan fungsi evaluasi yang dipakai di Bagian 5–8.
""")
code(r'''
# ---------------- KUMPULAN MODEL ----------------
def single_classifiers(cw=None):
    return {"LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw)),
            "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight=cw, random_state=SEED)),
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
            "DecisionTree": DecisionTreeClassifier(max_depth=8, class_weight=cw, random_state=SEED)}

def ensemble_classifiers(cw=None):
    return {"RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                                   class_weight=cw, random_state=SEED),          # bagging
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, n_jobs=4, random_state=SEED, verbosity=0),  # boosting
            "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight=cw,
                                           random_state=SEED, verbose=-1, n_jobs=4),            # boosting
            "Stacking": StackingClassifier(                                                    # stacking
                estimators=[("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw))),
                            ("svm", make_pipeline(StandardScaler(), SVC(C=10, probability=True, class_weight=cw,
                                                                         random_state=SEED))),
                            ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                          class_weight=cw, random_state=SEED))],
                final_estimator=LogisticRegression(max_iter=2000), cv=5, n_jobs=-1)}

def ALL_CLS(cw=None): return {**single_classifiers(cw), **ensemble_classifiers(cw)}

imp = lambda: SimpleImputer(strategy="median")
SINGLE_REG = {"Ridge": make_pipeline(imp(), StandardScaler(), Ridge(alpha=1.0)),
              "KNN": make_pipeline(imp(), StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
              "SVR-RBF": make_pipeline(imp(), StandardScaler(), SVR(C=10, gamma="scale")),
              "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=SEED)}
ENSEMBLE_REG = {"RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
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

# ---------------- EVALUASI E-NOSE ----------------
def augment_series(d, copies, jitter, scale, rng):
    """Augmentasi (Bagian 8): jitter = derau sensor; scale = beda sensitivitas antar unit sensor."""
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

def evaluate_enose(model, d, split="loso", kind="engineered", with_temperature=False,
                   resampler=None, augment=None, seed=SEED):
    """Kembalikan skor per fold. Augmentasi & resampling hanya menyentuh fold LATIH."""
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
        rows.append({"fold": groups[te][0] if split == "loso" else len(rows),
                     "macro_f1": f1_score(y_all[te], p, average="macro"),
                     "accuracy": accuracy_score(y_all[te], p),
                     "recall_excellent": recall_score(y_all[te], p, labels=[0], average="macro", zero_division=0)})
    return pd.DataFrame(rows)

LOG_EN = []
def run_en(tag, name, model, d=enose, label="", **kw):
    r = evaluate_enose(model, d, **kw)
    LOG_EN.append({"bagian": tag, "model": name, "tugas": "biner" if d is binary else "4 kelas", "variasi": label,
                   "split": kw.get("split", "loso"), "fitur": kw.get("kind", "engineered"),
                   "f1": r.macro_f1.mean(), "sd": r.macro_f1.std(ddof=1), "recall_excellent": r.recall_excellent.mean()})
    print(f"  {name:13s} {LOG_EN[-1]['tugas']:7s} {LOG_EN[-1]['split']:6s}/{LOG_EN[-1]['fitur']:10s} {label:18s} "
          f"macro-F1 = {mean_sd(r.macro_f1)}")
    return r

# ---------------- EVALUASI AQSOLDB ----------------
def scaffold_split(d, test_frac=0.2, seed=SEED):
    """Scaffold besar -> data latih; scaffold langka mengisi data uji (= kimia baru)."""
    groups = defaultdict(list)
    for i, s in enumerate(d["scaffold"]): groups[s].append(i)
    keys = list(groups); np.random.default_rng(seed).shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)
    n_train, a, b = int(round((1 - test_frac) * len(d))), [], []
    for k in keys: (a if len(a) + len(groups[k]) <= n_train else b).extend(groups[k])
    return np.array(sorted(a)), np.array(sorted(b))

tr_aq, te_aq = scaffold_split(aq)
perm = np.random.default_rng(SEED).permutation(len(aq)); k = int(round(.2 * len(aq)))
rtr_aq, rte_aq = np.sort(perm[k:]), np.sort(perm[:k])
inner_aq = list(GroupKFold(5).split(tr_aq, groups=aq["scaffold"].iloc[tr_aq].to_numpy()))

def reg_scores(t, p):
    return {"rmse": float(np.sqrt(mean_squared_error(t, p))), "mae": float(mean_absolute_error(t, p)),
            "r2": float(r2_score(t, p))}

LOG_AQ = []
def run_aq(tag, name, model, X, split="scaffold", label="", sample_weight=None):
    a, b = (tr_aq, te_aq) if split == "scaffold" else (rtr_aq, rte_aq)
    m = clone(model)
    if hasattr(m, "cv") and split == "scaffold": m.set_params(cv=inner_aq)
    kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
    p = m.fit(X.iloc[a], y_aq[a], **kw).predict(X.iloc[b]); s = reg_scores(y_aq[b], p)
    LOG_AQ.append({"bagian": tag, "model": name, "split": split, "variasi": label, **s})
    print(f"  {name:16s} {split:8s} {label:16s} RMSE = {s['rmse']:.3f}   R² = {s['r2']:.3f}")
    return m, p

print(f"Scaffold split: {len(tr_aq):,} latih / {len(te_aq):,} uji · "
      f"scaffold uji yang juga ada di latih = {aq.scaffold.iloc[te_aq].isin(aq.scaffold.iloc[tr_aq]).sum()}")
''')
code(r'''
print("E-NOSE (4 kelas):")
lgbm_c, rf_c = ensemble_classifiers()["LightGBM"], ensemble_classifiers()["RandomForest"]
for split in ["random", "loso"]:
    for kind in ["raw", "engineered"]:
        run_en("5_leakage", "LightGBM", lgbm_c, split=split, kind=kind)
        run_en("5_leakage", "RandomForest", rf_c, split=split, kind=kind)
run_en("5_suhu", "LightGBM", lgbm_c, with_temperature=True, label="+Temperature")

print("\nAQSOLDB:")
for split in ["random", "scaffold"]:
    for fname, X in [("AqSolDB-17", X_aq17), ("RDKit-2D", X_aq)]:
        run_aq("5_leakage", "LightGBM", ENSEMBLE_REG["LightGBM"], X, split, fname)
        run_aq("5_leakage", "RandomForest", ENSEMBLE_REG["RandomForest"], X, split, fname)
''')
code(r'''
en, aqlog = pd.DataFrame(LOG_EN), pd.DataFrame(LOG_AQ)
f_rand = en.query("bagian=='5_leakage' and model=='LightGBM' and split=='random' and fitur=='engineered'").f1.iloc[0]
f_loso = en.query("bagian=='5_leakage' and model=='LightGBM' and split=='loso' and fitur=='engineered'").f1.iloc[0]
r_rand = aqlog.query("model=='LightGBM' and split=='random' and variasi=='RDKit-2D'").rmse.iloc[0]
r_scaf = aqlog.query("model=='LightGBM' and split=='scaffold' and variasi=='RDKit-2D'").rmse.iloc[0]
f_temp = en.query("bagian=='5_suhu'").f1.iloc[0]
HASIL.update(leak_enose_random=f_rand, leak_enose_loso=f_loso, leak_aq_random=r_rand, leak_aq_scaffold=r_scaf)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].bar(["split acak", "LOSO (jujur)"], [f_rand, f_loso], color=[RED, TEAL]); ax[0].set_ylim(0, 1.05)
ax[0].set_title("E-nose · macro-F1 (↑)"); [ax[0].text(i, v + .02, f"{v:.3f}", ha="center") for i, v in enumerate([f_rand, f_loso])]
ax[1].bar(["split acak", "scaffold (jujur)"], [r_rand, r_scaf], color=[RED, TEAL]); ax[1].set_ylim(0, 1.5)
ax[1].set_title("AqSolDB · RMSE (↓)"); [ax[1].text(i, v + .02, f"{v:.3f}", ha="center") for i, v in enumerate([r_rand, r_scaf])]
plt.tight_layout(); plt.show()
print(f"""📌 KESIMPULAN BAGIAN 5
- Model dan data yang sama: split acak memberi macro-F1 {f_rand:.3f}, tetapi pada seri baru hanya {f_loso:.3f}.
- AqSolDB: RMSE naik dari {r_rand:.3f} (acak) ke {r_scaf:.3f} (scaffold baru).
- Akurasi ≈99% pada dataset e-nose ini hanya bisa kami reproduksi dengan split acak, yang bocor.
- Kalau suhu dimasukkan sebagai input, macro-F1 LOSO naik ke {f_temp:.3f}. Suhu ruang ikut menentukan laju busuk,
  tetapi kami tetap memperlakukannya sebagai metadata (konsisten dengan skripsi). Ini bahan diskusi.
- Mulai bagian ini, SEMUA angka memakai split jujur (LOSO / scaffold).""")
''')

# =====================================================================================
md(r"""
---
# Bagian 6 — Ensemble learning

**Pertanyaan:** apakah menggabungkan banyak model mengalahkan satu model, **pada validasi yang jujur**?

| Jenis | Cara kerja (materi `Ensemble_Learning.pptx`) | Model di sini |
|---|---|---|
| Model tunggal | satu model saja | Logistic Regression, SVM, KNN, Decision Tree / Ridge, SVR |
| **Bagging** | banyak pohon dilatih paralel pada sampel bootstrap, lalu di-voting | Random Forest |
| **Boosting** | pohon dilatih berurutan; tiap pohon memperbaiki galat sebelumnya | XGBoost, LightGBM |
| **Stacking** | model-model berbeda, lalu meta-model belajar cara menggabungkannya | LR + SVM + RF → LR |

Detail penting: **meta-model stacking juga dilatih dengan LOSO / scaffold** di dalam data latih, supaya tidak bocor.

## 6a. E-nose
""")
code(r'''
SINGLE_NAMES = ["LogReg", "SVM-RBF", "KNN", "DecisionTree"]
for d, nama in [(enose, "4 KELAS"), (binary, "BINER segar vs tidak segar")]:
    print(nama)
    for name, m in ALL_CLS().items():
        run_en("6_ensemble", name, m, d=d)
en = pd.DataFrame(LOG_EN); e6 = en[en.bagian == "6_ensemble"]

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for a, t in zip(ax, ["4 kelas", "biner"]):
    s = e6[e6.tugas == t]
    a.bar(s.model, s.f1, yerr=s.sd, capsize=3, color=[GREY if m in SINGLE_NAMES else TEAL for m in s.model])
    a.set_title(f"E-nose {t} · macro-F1 LOSO (abu = tunggal, hijau = ensemble)"); a.set_ylim(0, 1.05)
    a.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()
''')
md(r"""
**Cara membaca:**
- Tinggi batang = rata-rata macro-F1 di 5 seri uji.
- Garis hitam = simpangan baku antar seri. Makin pendek, makin **stabil** model itu di setiap potongan daging.

## 6b. AqSolDB (scaffold split, 207 deskriptor)
""")
code(r'''
preds_aq = {}
for name, m in {**SINGLE_REG, **ENSEMBLE_REG}.items():
    _, preds_aq[name] = run_aq("6_ensemble", name, m, X_aq, label="RDKit-2D")
''')
md(r"""
## 6c. Optimasi hyperparameter dengan Optuna (Bayesian, TPE)
Optuna mencoba kombinasi hyperparameter LightGBM secara cerdas, belajar dari percobaan sebelumnya.
Skor yang dioptimalkan adalah **RMSE validasi silang yang dikelompokkan per scaffold**, dihitung **di dalam
data latih**. Data uji tidak disentuh.
""")
code(r'''
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
Xtr_aq, ytr_aq = X_aq.iloc[tr_aq], y_aq[tr_aq]
def objective(trial):
    params = dict(n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
                  learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                  num_leaves=trial.suggest_int("num_leaves", 15, 127),
                  min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
                  subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
                  colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                  reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                  random_state=SEED, verbose=-1, n_jobs=4)
    return float(np.mean([np.sqrt(np.mean((lgb.LGBMRegressor(**params).fit(Xtr_aq.iloc[a], ytr_aq[a])
                                           .predict(Xtr_aq.iloc[b]) - ytr_aq[b]) ** 2)) for a, b in inner_aq]))
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=CFG["optuna_trials"], show_progress_bar=True)
best_lgbm = lgb.LGBMRegressor(**study.best_params, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=4)
tuned_aq, p_tuned = run_aq("6_optuna", "LightGBM-Optuna", best_lgbm, X_aq, label=f"{CFG['optuna_trials']} trial")
print("CV RMSE terbaik di data latih:", round(study.best_value, 3))
''')
code(r'''
en, aqlog = pd.DataFrame(LOG_EN), pd.DataFrame(LOG_AQ)
b = en[(en.bagian == "6_ensemble") & (en.tugas == "biner")].set_index("model")
best_single = b.loc[SINGLE_NAMES].f1.idxmax()
a6 = aqlog[aqlog.bagian.isin(["6_ensemble", "6_optuna"])].set_index("model")
ax = a6.rmse.plot.barh(color=[GREY if m in SINGLE_REG else TEAL for m in a6.index], figsize=(7, 3.6),
                       title="AqSolDB · RMSE uji scaffold (↓)"); ax.invert_yaxis(); plt.show()
HASIL.update(ens_enose_xgb=b.loc["XGBoost", "f1"], ens_enose_xgb_sd=b.loc["XGBoost", "sd"],
             ens_enose_best_single=b.loc[best_single, "f1"], ens_aq_stacking=a6.loc["Stacking", "rmse"],
             ens_aq_optuna_test=a6.loc["LightGBM-Optuna", "rmse"], ens_aq_optuna_cv=study.best_value)
print(f"""📌 KESIMPULAN BAGIAN 6
- E-nose biner: XGBoost {b.loc['XGBoost','f1']:.3f} (sd {b.loc['XGBoost','sd']:.3f}) vs model tunggal terbaik
  {best_single} {b.loc[best_single,'f1']:.3f} (sd {b.loc[best_single,'sd']:.3f}). Boosting lebih tinggi DAN lebih stabil.
- E-nose 4 kelas: perbedaannya tipis. Disampaikan apa adanya.
- AqSolDB: stacking terbaik, RMSE {a6.loc['Stacking','rmse']:.3f}. Model heterogen saling menutupi galat.
- Optuna: CV RMSE {study.best_value:.3f}, tetapi di data uji {a6.loc['LightGBM-Optuna','rmse']:.3f}. Data uji berisi
  scaffold langka (distribution shift), jadi tuning tidak bisa menggantikan data yang mewakili domain target.""")
''')

# =====================================================================================
md(r"""
---
# Bagian 7 — Imbalanced data

## 7a. Klasifikasi e-nose: kelas `excellent` paling sedikit
Tiga cara yang dibandingkan. Semuanya hanya diterapkan pada **fold latih**:
- **class_weight = balanced**: kesalahan pada kelas langka dihukum lebih berat
- **SMOTE**: membuat contoh sintetis kelas langka dengan interpolasi tetangga terdekat
- **Random undersampling**: membuang sebagian contoh kelas mayoritas
""")
code(r'''
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
for name in ["LogReg", "LightGBM"]:
    for cw in [None, "balanced"]:
        run_en("7_imbalance", name, ALL_CLS(cw)[name], label=f"class_weight={cw}")
    run_en("7_imbalance", name, ALL_CLS()[name], resampler=SMOTE(random_state=SEED), label="SMOTE")
    run_en("7_imbalance", name, ALL_CLS()[name], resampler=RandomUnderSampler(random_state=SEED), label="RandomUnder")
en = pd.DataFrame(LOG_EN)
display(en[en.bagian == "7_imbalance"][["model", "variasi", "f1", "sd", "recall_excellent"]].round(3))
''')
md(r"""
## 7b. Regresi AqSolDB: nilai ekstrem jarang
Untuk regresi, "imbalance" berarti nilai LogS ekstrem jarang muncul. Kami memberi **bobot sampel = 1 ÷ frekuensi
kategori kelarutannya**, supaya senyawa langka lebih diperhatikan saat model belajar.
""")
code(r'''
bins_tr = aq["sol_bin"].iloc[tr_aq].astype(str)
w = 1.0 / bins_tr.map(bins_tr.value_counts(normalize=True)).astype(float).to_numpy(); w = w / w.mean()
_, p_w = run_aq("7_imbalance", "LightGBM-Optuna", best_lgbm, X_aq, label="bobot inv-freq", sample_weight=w)
t = aq.iloc[te_aq].assign(tanpa_bobot=np.abs(y_aq[te_aq] - p_tuned), dengan_bobot=np.abs(y_aq[te_aq] - p_w))
per_bin = t.groupby("sol_bin", observed=False)[["tanpa_bobot", "dengan_bobot"]].mean()
per_bin.plot.bar(color=[GREY, AMBER], figsize=(8, 3.4), title="MAE per kategori kelarutan (↓)", rot=15); plt.show()
aqlog = pd.DataFrame(LOG_AQ)
im = en[(en.bagian == "7_imbalance") & (en.model == "LightGBM")].set_index("variasi")
HASIL.update(imb_lgbm_none=im.loc["class_weight=None", "f1"], imb_lgbm_balanced=im.loc["class_weight=balanced", "f1"],
             imb_aq_rmse_weighted=aqlog.query("bagian=='7_imbalance'").rmse.iloc[0])
print(f"""📌 KESIMPULAN BAGIAN 7
- E-nose (LightGBM): tanpa penanganan {im.loc['class_weight=None','f1']:.3f}, class weight {im.loc['class_weight=balanced','f1']:.3f},
  SMOTE {im.loc['SMOTE','f1']:.3f}, undersampling {im.loc['RandomUnder','f1']:.3f}. Selisihnya kecil dan masih di dalam simpangan
  baku antar seri, jadi kami TIDAK mengklaim satu teknik lebih unggul.
- AqSolDB: galat kategori ekstrem turun (sangat tidak larut {per_bin.iloc[0,0]:.2f} → {per_bin.iloc[0,1]:.2f};
  sangat larut {per_bin.iloc[4,0]:.2f} → {per_bin.iloc[4,1]:.2f}), tetapi kategori tengah memburuk dan RMSE total naik
  {HASIL['ens_aq_optuna_test']:.3f} → {HASIL['imb_aq_rmse_weighted']:.3f}. Trade-off: dipilih sesuai tujuan aplikasi.""")
''')

# =====================================================================================
md(r"""
---
# Bagian 8 — Data augmentation (deret waktu sensor)

**Apa:** membuat salinan sintetis dari seri latih, lalu fiturnya dihitung ulang. **Data uji tetap asli.**

| Teknik | Meniru apa di dunia nyata |
|---|---|
| **Jitter**: derau Gaussian 5% dari simpangan baku sensor | derau pembacaan sensor |
| **Gain scaling**: tiap sensor dikali N(1; 0,10) | beda sensitivitas antar unit sensor MOS (sensor A dan B tidak pernah identik) |

**Harapannya:** model jadi lebih tahan terhadap perbedaan antar alat/seri.
""")
code(r'''
AUGS = {"jitter": dict(copies=2, jitter=0.05, scale=0.0), "scaling": dict(copies=2, jitter=0.0, scale=0.10),
        "jitter+scaling": dict(copies=2, jitter=0.05, scale=0.10),
        "jitter+scaling x4": dict(copies=4, jitter=0.05, scale=0.10)}
for name in ["LogReg", "LightGBM"]:
    run_en("8_augment", name, ALL_CLS()[name], label="tanpa")
    for aname, a in AUGS.items():
        run_en("8_augment", name, ALL_CLS()[name], augment=a, label=aname)
en = pd.DataFrame(LOG_EN); a8 = en[en.bagian == "8_augment"].pivot(index="variasi", columns="model", values="f1")
a8 = a8.loc[["tanpa", "jitter", "scaling", "jitter+scaling", "jitter+scaling x4"]]
a8.plot.bar(color=[TEAL, GREY], figsize=(8, 3.4), title="Augmentasi · macro-F1 LOSO 4 kelas", rot=0); plt.show()
HASIL.update(aug_logreg_gain=a8.loc["scaling", "LogReg"] - a8.loc["tanpa", "LogReg"],
             aug_lgbm_best_change=(a8["LightGBM"].drop("tanpa") - a8.loc["tanpa", "LightGBM"]).max())
print(f"""📌 KESIMPULAN BAGIAN 8
- Gain scaling membantu Logistic Regression ({a8.loc['tanpa','LogReg']:.3f} → {a8.loc['scaling','LogReg']:.3f}).
- Untuk LightGBM, semua augmentasi justru menurunkan skor (terbaik berubah {HASIL['aug_lgbm_best_change']:+.3f}).
  Model pohon sudah tahan terhadap perubahan skala, sehingga data sintetis hanya menambah derau.
- Pelajaran: augmentasi bukan obat universal. Manfaatnya tergantung model dan harus diuji dengan validasi yang benar.""")
''')

# =====================================================================================
md(r"""
---
# Bagian 9 — Reinforcement: *active learning* yang dikendalikan agen bandit

**Masalahnya:** label mahal. Kalau kita hanya sanggup melabeli sedikit data, **data mana yang sebaiknya
dilabeli dulu?**

**Active learning** menjawabnya dengan memilih data. Ada beberapa strategi:
- `random`: ambil acak
- `uncertainty`: ambil data yang paling membuat model ragu
- `diversity`: ambil data yang paling berbeda dari semua data yang sudah berlabel

Masalahnya, **kita tidak tahu strategi mana yang terbaik untuk dataset tertentu**. Di sinilah agen RL masuk.

```
          ┌──────────── reward = perbaikan skor VALIDASI ◄───────────┐
          ▼                                                          │
   [ AGEN BANDIT ] ── pilih arm ──► random / uncertainty / diversity │
                                          │                          │
                                          ▼                          │
                               ambil 1 batch data → ORACLE beri label│
                                          │                          │
                                          ▼                          │
                                  latih ulang model ─────────────────┘
```

Agen memperbarui nilai tiap arm, persis seperti materi `reinforcement.pptx`:
- **ε-greedy + rata-rata Monte Carlo:** Q(a) ← Q(a) + (r − Q(a)) / n(a). Dengan peluang ε = 0,2 agen mencoba
  arm acak (eksplorasi); sisanya memilih arm dengan Q tertinggi (eksploitasi).
- **UCB1:** pilih arm dengan Q(a) + c·√(2 ln N / n(a)). Arm yang jarang dicoba diberi bonus.

Ini adalah **multi-armed bandit**, yaitu RL tanpa state. Gagasannya mengikuti *Active Learning by Learning*
(Hsu & Lin, AAAI 2015). Dalam simulasi ini "oracle" adalah label asli yang disembunyikan. **Data uji tidak pernah
dipakai untuk reward.**
""")
code(r'''
ARMS = ["random", "uncertainty", "diversity"]
def q_random(ctx, k, rng): return rng.choice(ctx["pool"], size=k, replace=False)
def q_uncertainty(ctx, k, rng):
    u = ctx["uncertainty"](ctx["model"], ctx["X"][ctx["pool"]]); return ctx["pool"][np.argsort(-u)[:k]]
def q_diversity(ctx, k, rng):                      # k-center greedy
    Z, pool = ctx["Z"], ctx["pool"]; d = ctx["min_dist"][pool].copy(); chosen = []
    for _ in range(k):
        j = int(np.argmax(d)); chosen.append(pool[j])
        d = np.minimum(d, np.linalg.norm(Z[pool] - Z[pool[j]], axis=1))
    return np.array(chosen)
QUERY = {"random": q_random, "uncertainty": q_uncertainty, "diversity": q_diversity}
def forest_std(model, X): return np.stack([t.predict(X) for t in model.estimators_]).std(axis=0)   # regresi
def proba_entropy(model, X):                                                                       # klasifikasi
    p = np.clip(model.predict_proba(X), 1e-12, 1); return -(p * np.log(p)).sum(axis=1)

class Fixed:                                       # strategi tetap (pembanding)
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
        self.n[arm] += 1; self.q[arm] += (r - self.q[arm]) / self.n[arm]      # rata-rata Monte Carlo
class UCB1(EpsilonGreedy):
    def __init__(self, arms, c=1.0): super().__init__(arms); self.c = c
    def select(self, t, rng):
        untried = [a for a in self.arms if self.n[a] == 0]
        if untried: return untried[0]
        N = sum(self.n.values())
        return max(self.arms, key=lambda a: self.q[a] + self.c * np.sqrt(2 * np.log(N) / self.n[a]))
def make_policy(name):
    return Fixed(name) if name in ARMS else EpsilonGreedy(ARMS, 0.2) if name == "bandit-eps" else UCB1(ARMS, 1.0)
POLICIES = ["random", "uncertainty", "diversity", "bandit-eps", "bandit-ucb"]

def run_active_learning(model, Xp, yp, Xv, yv, Xt, yt, *, policy, score, uncertainty, n_init, batch,
                        rounds, seed, higher_is_better=True, stratify_init=False):
    rng = np.random.default_rng(seed); Xp = np.asarray(Xp); Z = StandardScaler().fit_transform(Xp); n = len(Xp)
    if stratify_init:                              # minimal 1 contoh per kelas di awal
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
        arm = pol.select(t, rng)                                                   # 1. agen memilih aksi
        ctx = {"pool": np.flatnonzero(mask), "X": Xp, "Z": Z, "model": m, "uncertainty": uncertainty, "min_dist": min_dist}
        new = QUERY[arm](ctx, min(batch, mask.sum()), rng); mask[new] = False     # 2. oracle melabeli batch
        for i in new: min_dist = np.minimum(min_dist, np.linalg.norm(Z - Z[i], axis=1))
        lab = np.concatenate([lab, new]); m, v_new, te_s = fit_eval(lab)           # 3. latih ulang
        r = sign * (v_new - v) / (abs(v) + 1e-9); pol.update(arm, r); v = v_new    # 4. reward → update Q
        log.append({"round": t, "n_labelled": len(lab), "arm": arm, "val": v, "test": te_s, "reward": r})
    return log

def plot_curves(al, ylabel, title, ref=None):
    c = al.groupby(["policy", "n_labelled"]).test.mean().unstack(0)
    ax = c.plot(figsize=(8.5, 4.2), marker="o", ms=3, title=title,
                color={"random": GREY, "uncertainty": "#2e6f95", "diversity": "#6c8f7d", "bandit-eps": AMBER, "bandit-ucb": RED})
    ax.set_xlabel("jumlah label"); ax.set_ylabel(ylabel)
    if ref is not None: ax.axhline(ref, ls="--", c="k", lw=.8, label="semua pool dilabeli")
    ax.legend(); plt.show(); return c
''')
md(r"""
## 9a. AqSolDB
- **Pool** = data latih scaffold, yaitu senyawa yang "belum dilabeli".
- **Validasi** (untuk reward) = 15% data latih, dipisah per scaffold.
- **Uji** = scaffold uji, hanya untuk laporan.
- Mulai dari 100 label, lalu +100 per ronde selama 20 ronde (total 2.100 label). Model: Random Forest.
""")
code(r'''
sub = aq.iloc[tr_aq].reset_index(drop=True); ptr, pval = scaffold_split(sub, test_frac=0.15)
pool_i, val_i = tr_aq[ptr], tr_aq[pval]
Xn = X_aq.to_numpy(); Xn = np.where(np.isnan(Xn), np.nanmedian(Xn[pool_i], axis=0), Xn)   # median dari pool saja
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
rf_al = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=SEED)
full_sol = rmse(y_aq[te_aq], clone(rf_al).fit(Xn[pool_i], y_aq[pool_i]).predict(Xn[te_aq]))
print(f"pool {len(pool_i):,} · validasi {len(val_i):,} · uji {len(te_aq):,} · RMSE kalau SEMUA pool dilabeli = {full_sol:.3f}")
logs = []
for seed in range(CFG["al_seeds_sol"]):
    for pol in POLICIES:
        log = run_active_learning(rf_al, Xn[pool_i], y_aq[pool_i], Xn[val_i], y_aq[val_i], Xn[te_aq], y_aq[te_aq],
                                  policy=pol, score=rmse, uncertainty=forest_std, n_init=100, batch=100, rounds=20,
                                  seed=seed, higher_is_better=False)
        logs += [dict(r, policy=pol, seed=seed) for r in log]
        print(f"  seed {seed} · {pol:12s} RMSE akhir = {log[-1]['test']:.3f}")
al_sol = pd.DataFrame(logs)
c_sol = plot_curves(al_sol, "RMSE uji (↓)", "Kurva belajar active learning · AqSolDB", full_sol)
''')
code(r'''
target = c_sol["random"].iloc[-1]
need = {p: next((int(n) for n, s in c_sol[p].items() if s <= target), None) for p in c_sol}
arms = al_sol[al_sol.policy.str.startswith("bandit") & (al_sol.arm != "init")]
display(arms.groupby("policy").arm.value_counts(normalize=True).unstack().round(2).rename_axis("proporsi pilihan arm"))
hemat = 1 - need["bandit-ucb"] / need["random"]
HASIL.update(al_sol_random=c_sol["random"].iloc[-1], al_sol_ucb=c_sol["bandit-ucb"].iloc[-1], al_sol_full=full_sol,
             al_sol_labels_ucb=need["bandit-ucb"], al_sol_labels_random=need["random"], al_sol_saving=hemat)
print(f"""📌 KESIMPULAN 9a
- Pada 2.100 label: random {c_sol['random'].iloc[-1]:.3f} vs bandit UCB1 {c_sol['bandit-ucb'].iloc[-1]:.3f} (semua pool: {full_sol:.3f}).
- Label yang dibutuhkan untuk menyamai RMSE akhir random ({target:.3f}): {need}
- Bandit UCB1 hemat ≈{hemat:.0%} label. Agen tidak tahu strategi terbaik di awal, tetapi hasilnya setara strategi terbaik.""")
''')
md(r"""
## 9b. E-nose (biner segar vs tidak segar)
Untuk setiap seri uji, satu seri lain menjadi validasi dan tiga sisanya menjadi pool.
Mulai dari 40 label, lalu +40 per ronde selama 15 ronde. Model: XGBoost.
""")
code(r'''
Xe, ye = X_en.to_numpy(), binary["y"].to_numpy(); ser = enose.series.to_numpy(); names = sorted(np.unique(ser))
xgb_al = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                       n_jobs=4, random_state=SEED, verbosity=0)
f1m = lambda a, b: float(f1_score(a, b, average="macro"))
logs, fulls = [], []
for i, test_s in enumerate(names):
    val_s = names[(i + 1) % 5]
    te_i, va_i = np.flatnonzero(ser == test_s), np.flatnonzero(ser == val_s)
    po_i = np.flatnonzero((ser != test_s) & (ser != val_s))
    fulls.append(f1m(ye[te_i], clone(xgb_al).fit(Xe[po_i], ye[po_i]).predict(Xe[te_i])))
    for seed in range(CFG["al_seeds_enose"]):
        for pol in POLICIES:
            log = run_active_learning(xgb_al, Xe[po_i], ye[po_i], Xe[va_i], ye[va_i], Xe[te_i], ye[te_i], policy=pol,
                                      score=f1m, uncertainty=proba_entropy, n_init=40, batch=40, rounds=15,
                                      seed=seed, stratify_init=True)
            logs += [dict(r, policy=pol, seed=seed, test_series=test_s) for r in log]
    print(f"  seri uji {test_s} selesai")
al_en = pd.DataFrame(logs)
c_en = plot_curves(al_en, "macro-F1 seri uji (↑)", "Kurva belajar active learning · e-nose", np.mean(fulls))
HASIL.update(al_en_random=c_en["random"].iloc[-1], al_en_ucb=c_en["bandit-ucb"].iloc[-1],
             al_en_eps=c_en["bandit-eps"].iloc[-1], al_en_start=c_en["random"].iloc[0])
print(f"""📌 KESIMPULAN 9b
- Dengan 40 label saja F1 sudah {c_en['random'].iloc[0]:.3f}, dan kurvanya hampir datar.
- Pada 640 label: random {c_en['random'].iloc[-1]:.3f}, bandit ε-greedy {c_en['bandit-eps'].iloc[-1]:.3f}, UCB1 {c_en['bandit-ucb'].iloc[-1]:.3f}.
- Di sini active learning TIDAK menang. Strategi "pintar" memilih titik aneh dari seri yang sama, padahal
  tantangannya menebak seri BARU. Bagian 10 membuktikan penyebabnya.""")
''')

# =====================================================================================
md(r"""
---
# Bagian 10 — Tambah *batch* atau tambah baris?

Ini menjawab rubrik *"bagaimana tambahan data meningkatkan performa"*, dan menjelaskan kenapa active learning gagal
di e-nose.

**Eksperimen:** XGBoost biner, setiap seri bergantian menjadi data uji.
- **Lebih banyak seri:** latih dengan 1, 2, 3, atau 4 seri (semua kombinasi).
- **Lebih sedikit baris:** latih dengan semua 4 seri, tetapi hanya 640, 160, atau 40 baris acak.
""")
code(r'''
rng, rows = np.random.default_rng(SEED), []
xgb_c = ensemble_classifiers()["XGBoost"]
for test_s in names:
    te = ser == test_s; others = [s for s in names if s != test_s]; all_tr = np.flatnonzero(~te)
    for k in range(1, 5):
        for combo in combinations(others, k):
            trn = np.flatnonzero(np.isin(ser, combo))
            rows.append({"desain": f"{k} seri", "f1": f1m(ye[te], clone(xgb_c).fit(X_en.iloc[trn], ye[trn]).predict(X_en[te]))})
    for k in range(1, 5):              # menjaga urutan acak sama dengan versi skrip
        rng.choice(all_tr, int(len(all_tr) * k / 4), replace=False)
    for n in [40, 160, 640]:
        trn = rng.choice(all_tr, n, replace=False)
        rows.append({"desain": f"4 seri · {n} baris", "f1": f1m(ye[te], clone(xgb_c).fit(X_en.iloc[trn], ye[trn]).predict(X_en[te]))})
curve = pd.DataFrame(rows).groupby("desain", sort=False).f1.mean()
curve = curve.loc[["1 seri", "2 seri", "3 seri", "4 seri", "4 seri · 640 baris", "4 seri · 160 baris", "4 seri · 40 baris"]]
ax = curve.plot.barh(color=[TEAL] * 4 + [AMBER] * 3, figsize=(8, 3.6), title="Macro-F1 seri uji: tambah seri vs tambah baris")
ax.invert_yaxis(); [ax.text(v + .01, i, f"{v:.3f}", va="center") for i, v in enumerate(curve)]; ax.set_xlim(0, 1.05); plt.show()
HASIL.update(batch_1=curve["1 seri"], batch_3=curve["3 seri"], batch_4=curve["4 seri"], batch_4_160=curve["4 seri · 160 baris"])
print(f"""📌 KESIMPULAN BAGIAN 10
- Menambah seri: 1 → 4 seri menaikkan F1 dari {curve['1 seri']:.3f} ke {curve['4 seri']:.3f}.
- 160 baris dari 4 seri ({curve['4 seri · 160 baris']:.3f}) MENGALAHKAN 6.480 baris dari 3 seri ({curve['3 seri']:.3f}).
- Yang kurang adalah KERAGAMAN BATCH, bukan jumlah label. Itu sebabnya active learning (yang memilih baris)
  tidak membantu di e-nose.
- Implikasi untuk skripsi Erlangga: rancangan 16 batch sudah tepat; lebih baik menambah batch daripada
  menambah bacaan per batch.""")
''')

# =====================================================================================
md(r"""
---
# Bagian 11 — Apa yang dipelajari model? (SHAP)

SHAP mengukur seberapa besar kontribusi tiap fitur terhadap prediksi. Untuk e-nose, kontribusi fitur dijumlahkan
**per sensor**, dan dihitung pada **seri uji LOSO**, supaya yang terukur adalah pola yang berlaku untuk potongan
daging baru.
""")
code(r'''
import shap
sample = X_aq.iloc[te_aq].sample(1000, random_state=SEED)
shap_aq = pd.Series(np.abs(shap.TreeExplainer(tuned_aq).shap_values(sample)).mean(0), index=sample.columns)
parts = []
for trn, tst in LeaveOneGroupOut().split(X_en, ye, ser):
    m = clone(ensemble_classifiers()["XGBoost"]).fit(X_en.iloc[trn], ye[trn])
    parts.append(pd.Series(np.abs(shap.TreeExplainer(m).shap_values(X_en.iloc[tst])).mean(0), index=X_en.columns))
feat = pd.concat(parts, axis=1).mean(1)
shap_sensor = feat.groupby(feat.index.str.split("_").str[0]).sum().sort_values(ascending=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
shap_aq.sort_values(ascending=False).head(8)[::-1].plot.barh(ax=ax[0], color=TEAL, title="AqSolDB · mean |SHAP| (top 8)")
shap_sensor[::-1].plot.barh(ax=ax[1], color=AMBER, title="E-nose biner · mean |SHAP| per sensor")
plt.tight_layout(); plt.show()
HASIL.update(shap_aq_top=shap_aq.idxmax(), shap_en_top=shap_sensor.index[0], shap_en_mq135=shap_sensor["MQ135"])
print(f"""📌 KESIMPULAN BAGIAN 11
- AqSolDB: fitur terpenting {shap_aq.idxmax()}. Sesuai kimia: makin lipofilik (logP tinggi), makin sulit larut di air.
- E-nose: sensor {', '.join(shap_sensor.index[:2])} paling menentukan; MQ135 hanya {shap_sensor['MQ135']:.2f}.
  Domainnya daging sapi mentah, jadi untuk alat Erlangga (makanan matang) ini HIPOTESIS:
  apakah satu MQ-135 cukup, atau perlu sensor metana (MQ4) / hidrogen (MQ8)?""")
''')

# =====================================================================================
md(r"""
---
# Bagian 12 — (Opsional) Slot topik Rijal: citra kuning telur puyuh

**Status: menunggu data.** Bagian ini tidak berjalan kecuali `JALANKAN_TRACK_C = True` di Bagian 1.

**Alurnya** (mengikuti proposal Rijal):
1. Citra kuning telur diubah menjadi vektor 1.280 angka oleh **EfficientNet-B0**, sebuah *deep learning* yang
   menurut dosen terhitung aspek ensemble.
2. Vektor itu diklasifikasi oleh **XGBoost** (skenario hybrid), dibandingkan dengan model tunggal, dengan dan
   tanpa SMOTE.
3. Fungsi `run_active_learning` dari Bagian 9 bisa langsung dipakai. Untuk telur, **1 label hemat = 1 telur
   tidak perlu dipecah.**

**Format data yang dibutuhkan** (simpan di Google Drive):
```
EGG_DIR/labels.csv     image,yolk_index,storage_day,temperature
EGG_DIR/images/<image>
```
Ambang kelas Yolk Index **wajib** diambil dari protokol penelitian Rijal, bukan ditebak.
""")
code(r'''
EGG_DIR = Path("/content/drive/MyDrive/egg_yolk")
YI_THRESHOLDS = None      # contoh (lo, hi) dari protokol Rijal - JANGAN ditebak
if JALANKAN_TRACK_C:
    assert YI_THRESHOLDS is not None, "isi ambang Yolk Index dari protokol penelitian"
    %pip install -q torch torchvision pillow
    import torch
    from PIL import Image
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    from imblearn.over_sampling import SMOTE
    wts = EfficientNet_B0_Weights.IMAGENET1K_V1; net = efficientnet_b0(weights=wts)
    net.classifier = torch.nn.Identity(); net.eval(); prep = wts.transforms()
    labels = pd.read_csv(EGG_DIR / "labels.csv"); lo, hi = YI_THRESHOLDS
    labels["y"] = np.select([labels.yolk_index < lo, labels.yolk_index < hi], [0, 1], 2)
    with torch.no_grad():
        F = np.vstack([net(prep(Image.open(EGG_DIR / "images" / n).convert("RGB")).unsqueeze(0)).squeeze(0).numpy()
                       for n in labels["image"]])
    Xg, yg = F, labels["y"].to_numpy(); rows = []
    models_g = {"LogReg (tunggal)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                "XGBoost (hybrid)": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                                  colsample_bytree=0.5, n_jobs=4, random_state=SEED, verbosity=0)}
    for trn, tst in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xg, yg):
        for name, m in models_g.items():
            for sm in [False, True]:
                Xa, ya = SMOTE(random_state=SEED).fit_resample(Xg[trn], yg[trn]) if sm else (Xg[trn], yg[trn])
                rows.append({"model": name, "smote": sm, "f1": f1m(yg[tst], clone(m).fit(Xa, ya).predict(Xg[tst]))})
    display(pd.DataFrame(rows).groupby(["model", "smote"]).f1.agg(["mean", "std"]).round(3))
else:
    print("Track C dilewati: data citra telur belum tersedia. (Tidak ada angka yang dilaporkan untuk track ini.)")
''')

# =====================================================================================
md(r"""
---
# Bagian 13 — Ringkasan semua angka & kesimpulan
""")
code(r'''
R = HASIL
tabel = pd.DataFrame([
    ["5 · Validasi", "E-nose macro-F1: split acak vs LOSO", f"{R['leak_enose_random']:.3f} → {R['leak_enose_loso']:.3f}"],
    ["5 · Validasi", "AqSolDB RMSE: split acak vs scaffold", f"{R['leak_aq_random']:.3f} → {R['leak_aq_scaffold']:.3f}"],
    ["6 · Ensemble", "E-nose biner: XGBoost vs model tunggal terbaik", f"{R['ens_enose_xgb']:.3f} vs {R['ens_enose_best_single']:.3f}"],
    ["6 · Ensemble", "AqSolDB: RMSE stacking (terbaik)", f"{R['ens_aq_stacking']:.3f}"],
    ["6 · Optuna", "AqSolDB: CV RMSE vs uji", f"{R['ens_aq_optuna_cv']:.3f} vs {R['ens_aq_optuna_test']:.3f}"],
    ["7 · Imbalance", "E-nose LightGBM: tanpa vs class weight", f"{R['imb_lgbm_none']:.3f} vs {R['imb_lgbm_balanced']:.3f}"],
    ["7 · Imbalance", "AqSolDB RMSE total: tanpa vs dengan bobot", f"{R['ens_aq_optuna_test']:.3f} vs {R['imb_aq_rmse_weighted']:.3f}"],
    ["8 · Augmentasi", "LogReg: perubahan F1 karena gain scaling", f"{R['aug_logreg_gain']:+.3f}"],
    ["9 · Reinforcement", "AqSolDB: label untuk menyamai random-2.100 (UCB1)", f"{R['al_sol_labels_ucb']:,} (hemat {R['al_sol_saving']:.0%})"],
    ["9 · Reinforcement", "E-nose 640 label: random vs bandit ε-greedy", f"{R['al_en_random']:.3f} vs {R['al_en_eps']:.3f}"],
    ["10 · Data", "E-nose: 3 seri vs 4 seri·160 baris", f"{R['batch_3']:.3f} vs {R['batch_4_160']:.3f}"],
    ["11 · SHAP", "Fitur/sensor terpenting", f"{R['shap_aq_top']} · {R['shap_en_top']} (MQ135 {R['shap_en_mq135']:.2f})"],
], columns=["Bagian", "Pertanyaan", "Hasil"])
display(tabel)
json.dump({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in R.items()},
          open(RESULTS / "hasil_ringkas.json", "w"), indent=2)
pd.DataFrame(LOG_EN).to_csv(RESULTS / "log_enose.csv", index=False)
pd.DataFrame(LOG_AQ).to_csv(RESULTS / "log_aqsoldb.csv", index=False)
print("Semua angka tersimpan di results/hasil_ringkas.json")
''')
md(r"""
## Kesimpulan (4 pesan untuk presentasi)

1. **Cara validasi lebih menentukan daripada pilihan model.** Split acak membuat skor terlihat hampir sempurna
   padahal bocor. Inilah alasan GroupKFold per `batch_id` (Erlangga) dan scaffold split (Michael) wajib dipakai.
2. **Ensemble boosting adalah baseline yang kuat dan stabil; stacking menang tipis saat modelnya heterogen.**
   Tuning Optuna tidak bisa menutup *distribution shift*.
3. **Agen bandit menghemat label ketika masalahnya memang kekurangan label** (AqSolDB), tanpa perlu menebak
   strategi terbaik di awal.
4. **Ketika masalahnya pergeseran domain antar batch, solusinya batch baru, bukan baris baru** (e-nose).
   Ini mendukung rancangan 16 batch Erlangga.

## Batasan yang kami akui
- E-nose hanya 5 seri, sehingga simpangan bakunya besar.
- Track A adalah proksi (label TVC, daging mentah), bukan replikasi skripsi Erlangga (panel sensori, makanan matang).
- Active learning disimulasikan dari label yang sudah ada.
- Track C belum punya data, jadi tidak ada klaim performa.

## Langkah berikutnya
- **Erlangga:** ambil 16 batch, lalu jalankan pipeline ini dengan GroupKFold per batch. Uji sensor MQ4/MQ8 tambahan.
- **Michael:** fitur fingerprint/graf untuk scaffold langka; tangani *distribution shift*.
- **Rijal:** citra pilot → embedding EfficientNet-B0 → XGBoost → bandit AL untuk menghemat telur yang dipecah.
- **Pengembangan RL:** active learning **tingkat batch**, yaitu agen memilih sesi/batch mana yang dilabeli.

## Referensi utama
Hsu & Lin (2015) *Active Learning by Learning*, AAAI · Sorkun dkk. (2019) *Scientific Data* 6:143 ·
Wijaya dkk. (2018) *Data in Brief* 21:2414–2420 · Bemis & Murcko (1996) *J. Med. Chem.* 39:2887 ·
Chawla dkk. (2002) SMOTE, *JAIR* 16 · Akiba dkk. (2019) Optuna, KDD · Chen & Guestrin (2016) XGBoost, KDD ·
Lundberg & Lee (2017) SHAP, NeurIPS · Auer dkk. (2002) UCB1, *Machine Learning* 47.
""")

# =====================================================================================
nb = {"cells": copy.deepcopy(CELLS), "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
(ROOT / "notebooks" / NAME).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", NAME, len(nb["cells"]), "cells")
