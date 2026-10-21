"""Generate notebooks/ETS_ML2_Kesegaran_Ayam.ipynb, the single end-to-end Colab notebook.

Edit the cells here, then run:  python3 scripts/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "MuhammadErlanggaKurniawan/ets-ml2-kelompok"
NAME = "ETS_ML2_Kesegaran_Ayam.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n")})


# =====================================================================================
md(rf"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{NAME})

# Deteksi Kesegaran Daging Ayam dengan Sensor Gas dan Machine Learning
### Proyek ETS Machine Learning II

**Kelompok:** Muhammad Erlangga Kurniawan (23083010045) · Michael Valencio Erasmo Bagas Ganendra (23083010042) ·
Rijal Apuila Khufi (23083010033)

---

## Ceritanya singkat

Daging ayam yang mulai busuk mengeluarkan gas, misalnya amonia dan senyawa organik yang mudah menguap.
Sensor gas murah seri **MQ** bisa "mencium" gas itu. Rangkaian beberapa sensor gas seperti ini disebut
**hidung elektronik (e-nose)**.

**Pertanyaan proyek:** dari bacaan sensor saja, bisakah komputer menebak apakah daging ayam masih
**Fresh**, **Semi-fresh**, atau sudah **Spoiled**?

**Data:** satu dataset publik, *e-nose* daging ayam dari Anwar & Anwar (2025). Isinya 7 sensor
(6 sensor gas MQ + kelembapan), 15 sesi rekaman, sekitar 218 ribu bacaan.

**Kenapa topik ini:** ini irisan skripsi kami bertiga.
- Erlangga: kesegaran makanan (termasuk ayam) dengan sensor **MQ-135**. Sensor yang sama ada di dataset ini.
- Rijal: mutu produk unggas (kuning telur puyuh).
- Michael: model *gradient boosting* (XGBoost), dipakai di bagian ensemble.

**Tiga teknik Machine Learning II yang dipakai** (minimal 3 aspek → nilai minimum kelompok 75):
1. **Augmentasi data**
2. **Ensemble learning**
3. **Reinforcement learning**

## Peta notebook

| Bagian | Isi | Rubrik dosen |
|---|---|---|
| 1 | Persiapan | — |
| 2 | **Akuisisi** data | pipeline: akuisisi |
| 3 | Kenali datanya | latar belakang |
| 4 | Membuat label Fresh / Semi-fresh / Spoiled | pengolahan |
| 5 | **Pengolahan data (wrangling)** | wrangling (15%) |
| 6 | Cara menguji yang jujur | analisis (25%) |
| 7 | **Ensemble learning** | ensemble (15%) |
| 8 | **Augmentasi data** | augmentasi (15%) |
| 9 | **Reinforcement learning** + efek tambah data | reinforcement (15%) |
| 10 | Sensor mana yang paling penting | analisis |
| 11 | **Ringkasan semua angka** + kesimpulan | presentasi (30%) |

## Cara menjalankan
Pilih menu **Runtime → Run all**. Tidak perlu GPU. Waktunya sekitar 5 menit di Colab.
Semua grafik dan angka disimpan di folder `hasil/`.
Setiap bagian diakhiri kotak **📌 Hasil**, yang angkanya ditulis otomatis dari hasil run.
""")

# =====================================================================================
md(r"""
---
# Bagian 1 — Persiapan
Memuat *library*. Semua sudah tersedia di Google Colab.
""")
code(r"""
import hashlib, json, subprocess, sys, urllib.request, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("hasil")
OUT.mkdir(exist_ok=True)
CLASSES = ["Fresh", "Semi-fresh", "Spoiled"]
CLASS_COLORS = ["#13866B", "#E3A33B", "#C4553B"]
TEAL, GREY, AMBER, RED, INK = "#0F5257", "#9AA5A6", "#E3A33B", "#C4553B", "#16262B"
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
RESULTS = {}  # every number used in the summary and the slides


def fmt(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


def pct(x):
    return f"{100 * x:.1f}%".replace(".", ",")


def num(n):
    return f"{n:,}".replace(",", ".")


def signed(x):
    return f"{x:+.3f}".replace(".", ",")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")


def result_box(text):
    display(Markdown("> **📌 Hasil**\n>\n" + "\n".join("> " + line for line in text.strip().split("\n"))))


print("Siap.")
""")

# =====================================================================================
md(r"""
---
# Bagian 2 — Akuisisi data

**Sumber:** Anwar, H., & Anwar, T. (2025). *Quality assessment of chicken using machine learning and electronic
nose.* Sensing and Bio-Sensing Research, 47, 100739. Data dan kode penulis ada di GitHub
`talhaanwarch/chicken_assessment_e-nose`.

**Apa yang dilakukan:** mengunduh file `complete_data.csv` (±12 MB), lalu mengecek "sidik jari" file (SHA-256).
Kalau isi file berubah satu angka saja, sidik jarinya ikut berubah. Jadi kita yakin data yang dipakai sama persis
dengan milik penulis.
""")
code(r"""
URL = ("https://raw.githubusercontent.com/talhaanwarch/chicken_assessment_e-nose/"
       "e7e3ca686bce0ef3e5c59f9833c85a3190c36e6e/complete_data.csv")
SHA256 = "7c9920fcfe523af1ee2855d53f13334ddba93db5135d7868b9f63c5cd488d953"
path = Path("data/raw/complete_data.csv")
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, path)
assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256, "Isi file berbeda dari sumber asli"

# column order follows the author's notebook: s1..s6 are MQ sensors, s8 is humidity
SENSOR_NAMES = {"s1": "MQ-135", "s2": "MQ-2", "s3": "MQ-3", "s4": "MQ-6", "s5": "MQ-7", "s6": "MQ-9",
                "s8": "Kelembapan"}
raw = pd.read_csv(path).rename(columns={**SENSOR_NAMES, "hour": "jam", "group": "sesi"})
MQ = ["MQ-135", "MQ-2", "MQ-3", "MQ-6", "MQ-7", "MQ-9"]
SENSORS = MQ + ["Kelembapan"]
print(f"Data terunduh dan terverifikasi: {num(len(raw))} bacaan, {raw.sesi.nunique()} sesi.")
display(raw.head())
""")

# =====================================================================================
md(r"""
---
# Bagian 3 — Kenali datanya

- Setiap **sesi** adalah satu rekaman: sampel ayam diletakkan di dalam kotak sensor, lalu dipantau terus selama
  12–27 jam.
- Sensor membaca sekitar 11–12 kali per menit. Kolom `jam` berisi sudah berapa jam ayam dipantau.
- Nilai sensor sudah dikurangi nilai awalnya, jadi setiap sesi mulai dari 0.

Grafik di bawah menunjukkan bacaan MQ-135 untuk setiap sesi.
""")
code(r"""
sessions = raw.groupby("sesi").agg(bacaan=("jam", "size"), lama_jam=("jam", "max"),
                                   puncak_MQ135=("MQ-135", "max")).reset_index()
display(sessions.T)

hourly = raw.groupby(["sesi", "jam"])["MQ-135"].mean().reset_index()
fig, ax = plt.subplots(figsize=(9, 4.2))
for s, d in hourly.groupby("sesi"):
    ax.plot(d.jam, d["MQ-135"], lw=1.6, color=TEAL if d.jam.max() > 15 else AMBER, alpha=0.85)
ax.plot([], [], color=TEAL, label="sesi panjang (> 15 jam)")
ax.plot([], [], color=AMBER, label="sesi pendek (≤ 15 jam)")
ax.set(xlabel="jam sejak mulai dipantau", ylabel="bacaan MQ-135",
       title="Gas naik seiring waktu, tapi levelnya beda-beda tiap sesi")
ax.legend(loc="upper left")
save(fig, "01_mq135_per_sesi")
plt.show()

RESULTS.update(n_readings=len(raw), n_sessions=int(raw.sesi.nunique()),
               peak_min=float(sessions.puncak_MQ135.min()), peak_max=float(sessions.puncak_MQ135.max()))
result_box(f'''
- {num(len(raw))} bacaan dari {raw.sesi.nunique()} sesi rekaman.
- Semua sesi naik seiring waktu, tapi **puncak MQ-135 berbeda jauh**: dari {sessions.puncak_MQ135.min():.0f} sampai
  {sessions.puncak_MQ135.max():.0f}. Model harus bisa menebak sesi yang levelnya berbeda dari yang pernah ia lihat.
''')
""")

# =====================================================================================
md(r"""
---
# Bagian 4 — Membuat label Fresh / Semi-fresh / Spoiled

CSV tidak punya kolom label. Kita memakai aturan dari notebook penulis, berdasarkan **lama penyimpanan**:

| Jenis sesi | Fresh | Semi-fresh | Spoiled |
|---|---|---|---|
| Sesi panjang (> 15 jam, musim dingin) | jam ≤ 8 | 8 < jam ≤ 16 | jam > 16 |
| Sesi pendek (≤ 15 jam, setelah musim dingin) | jam ≤ 5 | 5 < jam ≤ 10 | jam > 10 |

Batasnya berbeda karena ayam lebih cepat busuk saat udara lebih hangat.

**Catatan jujur:** label ini berasal dari waktu penyimpanan, bukan dari uji laboratorium per sampel. Jadi label
adalah *perkiraan* kondisi ayam. Karena label dibuat dari waktu, **waktu tidak boleh dipakai sebagai fitur**.
Kalau dipakai, model cuma "membaca jam", bukan mencium ayam.
""")
code(r"""
long_session = raw.groupby("sesi")["jam"].transform("max") > 15
limit1 = np.where(long_session, 8, 5)
limit2 = np.where(long_session, 16, 10)
raw["label"] = np.select([raw.jam <= limit1, raw.jam <= limit2], [0, 1], 2)
counts = raw.label.value_counts().sort_index()
display(pd.DataFrame({"kelas": CLASSES, "jumlah bacaan": counts.values}))
""")

# =====================================================================================
md(r"""
---
# Bagian 5 — Pengolahan data (*data wrangling*)

Tiga langkah, dan masing-masing ada alasannya:

**1. Rata-rata per 5 menit.** Sensor membaca ±12 kali per menit, dan bacaan yang berdekatan hampir sama.
Kita ringkas tiap jam menjadi 12 "jendela" berdurasi 5 menit, lalu ambil rata-ratanya. Data jadi lebih rapi dan
tidak berisik.

**2. Dua jam pertama = masa kalibrasi.** Sensor gas butuh waktu untuk stabil. Penulis aslinya juga membuang
baris awal tiap sesi. Dua jam pertama (selalu Fresh) **tidak ikut dinilai**. Bacaan jam ke-1 dipakai sebagai
**titik pembanding** untuk sesi itu.

**3. Membuat 19 fitur yang menggambarkan gas dengan 3 cara**, ditambah kelembapan:

| Kelompok fitur | Artinya | Kenapa perlu |
|---|---|---|
| `level_` (6) | seberapa **banyak** gas (skala log) | log membuat angka besar dan kecil sebanding |
| `porsi_` (6) | **proporsi** tiap sensor dari total gas | polanya tetap sama walau level gas sesi berbeda |
| `naik_` (6) | seberapa jauh gas **naik dibanding jam ke-1** | mirip Rs/R0 di skripsi Erlangga; menghilangkan beda alat/sampel antar sesi |
| `Kelembapan` (1) | kelembapan udara | kondisi lingkungan |

Semua fitur hanya memakai data **masa lalu** dari sesi itu sendiri, jadi bisa dihitung langsung di alat sungguhan.

**Bagaimana dengan data tidak seimbang (*imbalanced*)?** Kita cek jumlah tiap kelas. Kalau perbandingannya
tidak jauh, teknik seperti SMOTE tidak diperlukan.
""")
code(r"""
def make_windows(readings):
    # split every hour of every session into 12 five-minute windows and average them
    r = readings.copy()
    r["urutan"] = r.groupby(["sesi", "jam"]).cumcount()
    r["n"] = r.groupby(["sesi", "jam"])["jam"].transform("size")
    r["jendela"] = r.urutan * 12 // r.n
    return r.groupby(["sesi", "jam", "jendela"]).agg(**{c: (c, "mean") for c in SENSORS},
                                                    label=("label", "first")).reset_index()


def make_features(win):
    # returns (features, rows) for windows after the 2-hour calibration period
    level = np.log1p(win[MQ].clip(lower=0))
    reference = level.where(win.jam == 1).groupby(win.sesi).transform("mean")
    keep = (win.jam >= 2).to_numpy()
    rows = win[keep].reset_index(drop=True)
    level, reference = level[keep].reset_index(drop=True), reference[keep].reset_index(drop=True)
    gas = rows[MQ].clip(lower=0)
    features = pd.concat([level.add_prefix("level_"),
                          gas.div(gas.sum(axis=1) + 1, axis=0).add_prefix("porsi_"),
                          (level - reference).add_prefix("naik_"),
                          rows[["Kelembapan"]]], axis=1)
    return features, rows


windows = make_windows(raw)
features, data = make_features(windows)
X = features.to_numpy()
y = data.label.to_numpy()
groups = data.sesi.to_numpy()
X_raw = data[SENSORS].to_numpy()                                         # for comparison: averages only
X_no_rise = features.drop(columns=[c for c in features if c.startswith("naik_")]).to_numpy()

class_counts = np.bincount(y, minlength=3)
ratio = class_counts.max() / class_counts.min()
print(f"{num(len(raw))} bacaan → {num(len(windows))} jendela 5 menit → {num(len(data))} jendela setelah masa kalibrasi")
print(f"{X.shape[1]} fitur per jendela")
display(features.head().round(3))

fig, ax = plt.subplots(figsize=(5.5, 3.6))
bars = ax.bar(CLASSES, class_counts, color=CLASS_COLORS)
ax.bar_label(bars, padding=3)
ax.set(ylabel="jumlah jendela", title="Jumlah data per kelas", ylim=(0, class_counts.max() * 1.18))
save(fig, "02_jumlah_kelas")
plt.show()

RESULTS.update(n_windows_all=len(windows), n_windows=len(data), n_features=X.shape[1],
               class_counts=dict(zip(CLASSES, class_counts.tolist())), class_ratio=float(ratio))
result_box(f'''
- {num(len(raw))} bacaan diringkas menjadi **{num(len(data))} jendela 5 menit** dengan **{X.shape[1]} fitur**.
- Jumlah per kelas: Fresh {num(class_counts[0])}, Semi-fresh {num(class_counts[1])}, Spoiled {num(class_counts[2])}.
  Perbandingan terbesar : terkecil = **{fmt(ratio, 2)}**, jadi data **cukup seimbang**. Penanganan imbalance
  seperti SMOTE tidak diperlukan.
''')
""")

# =====================================================================================
md(r"""
---
# Bagian 6 — Cara menguji yang jujur

**Masalahnya:** jendela 5 menit yang berurutan dalam satu sesi isinya hampir sama. Kalau data dibagi **acak**,
menit ke-100 bisa masuk data latih dan menit ke-105 masuk data uji. Model cukup "menghafal" sesi itu, dan
skornya jadi terlalu bagus. Ini disebut **kebocoran data (*data leakage*)**.

**Cara yang jujur: Leave-One-Session-Out (LOSO).**
1. Sembunyikan **1 sesi** sebagai ujian.
2. Latih model dengan **14 sesi** lainnya.
3. Ulangi 15 kali sampai setiap sesi pernah jadi ujian.

Ini meniru kondisi nyata: alat dipakai untuk **ayam baru** yang belum pernah dilihat model.

**Ukuran skor:**
- **Akurasi:** persentase tebakan yang benar.
- **Macro-F1:** skor 0–1 yang menghitung ketiga kelas secara setara (makin tinggi makin baik). Ini ukuran utama.
""")
code(r"""
def loso(make_model, X_, y_=None, g_=None):
    # every window is predicted by a model that never saw its session
    y_ = y if y_ is None else y_
    g_ = groups if g_ is None else g_
    pred = np.empty_like(y_)
    proba = np.zeros((len(y_), 3))
    for train, test in LeaveOneGroupOut().split(X_, y_, g_):
        model = make_model()
        model.fit(X_[train], y_[train])
        pred[test] = model.predict(X_[test])
        proba[test] = model.predict_proba(X_[test])
    return pred, proba


def scores(pred, y_=None, g_=None):
    y_ = y if y_ is None else y_
    g_ = groups if g_ is None else g_
    per_session = [accuracy_score(y_[g_ == s], pred[g_ == s]) for s in np.unique(g_)]
    return {"akurasi": accuracy_score(y_, pred), "macro_f1": f1_score(y_, pred, average="macro"),
            "akurasi_sesi_terburuk": min(per_session)}


make_rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=6, random_state=SEED, n_jobs=-1)

pred_random = np.empty_like(y)
for train, test in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
    pred_random[test] = make_rf().fit(X[train], y[train]).predict(X[test])
pred_loso, _ = loso(make_rf, X)
leak = {"acak": scores(pred_random), "loso": scores(pred_loso)}

fig, ax = plt.subplots(figsize=(5.8, 3.8))
vals = [leak["acak"]["akurasi"], leak["loso"]["akurasi"]]
bars = ax.bar(["dibagi acak\n(bocor)", "per sesi / LOSO\n(jujur)"], vals, color=[RED, TEAL], width=0.55)
ax.bar_label(bars, labels=[pct(v) for v in vals], padding=3, fontweight="bold")
ax.set(ylim=(0, 1.12), ylabel="akurasi", title="Random Forest: cara membagi data mengubah skor")
save(fig, "03_kebocoran")
plt.show()

RESULTS["leakage"] = leak
result_box(f'''
- Dibagi acak: akurasi **{pct(leak["acak"]["akurasi"])}**. Diuji per sesi (LOSO): **{pct(leak["loso"]["akurasi"])}**.
- Paper aslinya menemukan hal yang sama: Random Forest 100% dengan data acak, tetapi 69% tanpa pembagian acak.
- Mulai sekarang **semua hasil memakai LOSO**.
''')
""")

# =====================================================================================
md(r"""
---
# Bagian 7 — Ensemble learning

**Idenya:** satu model bisa salah. Kalau banyak model digabung, kesalahan satu model bisa ditutup model lain.
Mirip minta pendapat banyak orang sebelum mengambil keputusan.

| Jenis | Model | Cara kerja sederhana |
|---|---|---|
| tunggal | Logistic Regression | menarik garis pemisah antar kelas |
| tunggal | KNN | melihat 15 data terdekat, ikut suara terbanyak |
| tunggal | Decision Tree | satu pohon keputusan "jika… maka…" (kedalaman 6) |
| **ensemble – bagging** | **Random Forest** | 300 pohon (kedalaman 6, sama dengan Decision Tree), masing-masing dilatih dengan sampel data berbeda, lalu voting |
| **ensemble – boosting** | **XGBoost** | 200 pohon kecil dibuat berurutan, tiap pohon memperbaiki kesalahan pohon sebelumnya |
| **ensemble – voting** | **Voting** | gabungan Logistic Regression + Random Forest + XGBoost, peluang ketiganya dirata-rata |

Pengaturan setiap model **ditetapkan dari awal**. Kita tidak mengutak-atik pengaturan berdasarkan data uji,
karena itu juga termasuk curang.
""")
code(r"""
def make_xgb():
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         random_state=SEED, n_jobs=4)


MODELS = {
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=6, random_state=SEED),
    "Random Forest": make_rf,
    "XGBoost": make_xgb,
    "Voting": lambda: VotingClassifier([
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
        ("rf", make_rf()), ("xgb", make_xgb())], voting="soft"),
}
KIND = {"Logistic Regression": "tunggal", "KNN": "tunggal", "Decision Tree": "tunggal",
        "Random Forest": "ensemble", "XGBoost": "ensemble", "Voting": "ensemble"}

model_rows, predictions, probabilities = [], {}, {}
for name, make_model in MODELS.items():
    predictions[name], probabilities[name] = loso(make_model, X)
    model_rows.append({"model": name, "jenis": KIND[name], **scores(predictions[name])})
model_table = pd.DataFrame(model_rows)
display(model_table.assign(akurasi=model_table.akurasi.map(pct),
                           macro_f1=model_table.macro_f1.round(3),
                           akurasi_sesi_terburuk=model_table.akurasi_sesi_terburuk.map(pct)))

fig, ax = plt.subplots(figsize=(8.6, 4.2))
bars = ax.bar(model_table.model, model_table.macro_f1,
              color=[GREY if k == "tunggal" else TEAL for k in model_table.jenis])
ax.bar_label(bars, labels=[fmt(v) for v in model_table.macro_f1], padding=3, fontweight="bold")
ax.set(ylim=(0, 1), ylabel="macro-F1 (LOSO)", title="Model tunggal vs ensemble")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Rectangle((0, 0), 1, 1, color=TEAL)],
          labels=["model tunggal", "ensemble"], loc="upper left", ncol=2)
plt.xticks(rotation=15)
save(fig, "04_ensemble")
plt.show()
""")
md(r"""
### Seberapa besar pengaruh pengolahan data?
Random Forest yang sama dilatih dengan 3 versi fitur: rata-rata mentah saja, tambah level dan porsi, lalu
tambah kenaikan sejak kalibrasi.
""")
code(r"""
wrangling = {
    "rata-rata mentah\n(7 fitur)": scores(loso(make_rf, X_raw)[0])["macro_f1"],
    "+ level & porsi\n(13 fitur)": scores(loso(make_rf, X_no_rise)[0])["macro_f1"],
    "+ kenaikan sejak\nkalibrasi (19 fitur)": scores(predictions["Random Forest"])["macro_f1"],
}
fig, ax = plt.subplots(figsize=(6.6, 3.8))
bars = ax.bar(list(wrangling), list(wrangling.values()), color=[GREY, GREY, TEAL], width=0.6)
ax.bar_label(bars, labels=[fmt(v) for v in wrangling.values()], padding=3, fontweight="bold")
ax.set(ylim=(0, 1), ylabel="macro-F1 (LOSO)", title="Random Forest dengan fitur yang makin baik")
save(fig, "05_efek_wrangling")
plt.show()

best_rf = predictions["Random Forest"]
cm = confusion_matrix(y, best_rf)
fig, ax = plt.subplots(figsize=(4.8, 4))
ax.imshow(cm, cmap="Greens")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK)
ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="tebakan model",
       ylabel="kelas sebenarnya", title="Random Forest: tebakan vs kenyataan")
save(fig, "06_confusion_rf")
plt.show()

single = model_table[model_table.jenis == "tunggal"]
ensemble = model_table[model_table.jenis == "ensemble"]
best_single = single.loc[single.macro_f1.idxmax()]
best_ens = ensemble.loc[ensemble.macro_f1.idxmax()]
recall = cm.diagonal() / cm.sum(axis=1)
RESULTS.update(models=model_rows, wrangling={k.replace("\n", " "): v for k, v in wrangling.items()},
               confusion_rf=cm.tolist(), recall_rf=dict(zip(CLASSES, recall.round(3).tolist())))
result_box(f'''
- Ensemble terbaik: **{best_ens.model} macro-F1 {fmt(best_ens.macro_f1)}** (akurasi {pct(best_ens.akurasi)}).
  Model tunggal terbaik: {best_single.model} {fmt(best_single.macro_f1)}.
- Rata-rata ensemble {fmt(ensemble.macro_f1.mean())} vs rata-rata model tunggal {fmt(single.macro_f1.mean())}.
- Random Forest (banyak pohon) vs Decision Tree (satu pohon, kedalaman sama):
  {fmt(model_table.set_index("model").macro_f1["Random Forest"])} vs
  {fmt(model_table.set_index("model").macro_f1["Decision Tree"])}.
- Pengolahan data menaikkan Random Forest dari {fmt(list(wrangling.values())[0])} ke {fmt(list(wrangling.values())[-1])}.
- Kelas yang paling sulit: **Semi-fresh** (recall {recall[1]:.0%}), karena letaknya di tengah, di antara segar dan busuk.
''')
""")

# =====================================================================================
md(r"""
---
# Bagian 8 — Augmentasi data

**Kenapa perlu:** kita hanya punya **15 sesi**. Alat sungguhan nanti bisa punya sensor yang sedikit lebih atau
kurang peka, dan bacaannya bisa lebih berisik. Augmentasi membuat **data latih tambahan buatan** untuk
meniru variasi itu.

**Bagaimana caranya** (2 salinan buatan per sesi latih):
- **Jitter:** menambahkan derau acak kecil (5% dari variasi sensor), seperti sensor yang bergetar.
- **Variasi kepekaan:** mengalikan bacaan tiap sensor dengan faktor acak (sekitar ±30%) per sesi, seolah-olah
  kita punya alat e-nose lain.

**Aturan penting:** augmentasi **hanya untuk data latih**. Data uji selalu data asli.
""")
code(r"""
def augment(train_windows, rng, method, copies=2):
    out = []
    for k in range(copies):
        a = train_windows.copy()
        if method == "jitter":
            a[MQ] = a[MQ] + rng.normal(0, 0.05, (len(a), len(MQ))) * train_windows[MQ].std().to_numpy()
        if method == "kepekaan":
            for s in a.sesi.unique():
                rows = a.sesi == s
                a.loc[rows, MQ] = a.loc[rows, MQ] * np.exp(rng.normal(0, 0.3, len(MQ)))
        a["sesi"] = a["sesi"] + 100 * (k + 1)  # a synthetic copy behaves like a new session
        out.append(a)
    return pd.concat(out, ignore_index=True)


AUG_MODELS = {"Decision Tree": MODELS["Decision Tree"], "Random Forest": make_rf}
aug_rows = []
for method in ["tanpa", "jitter", "kepekaan"]:
    for name, make_model in AUG_MODELS.items():
        rng = np.random.default_rng(SEED)
        pred = np.empty_like(y)
        for test_session in np.unique(groups):
            train = groups != test_session
            X_train, y_train = X[train], y[train]
            if method != "tanpa":
                synthetic = augment(windows[windows.sesi != test_session], rng, method)
                f_aug, d_aug = make_features(synthetic)
                X_train = np.vstack([X_train, f_aug.to_numpy()])
                y_train = np.concatenate([y_train, d_aug.label.to_numpy()])
            pred[~train] = make_model().fit(X_train, y_train).predict(X[~train])
        aug_rows.append({"augmentasi": method, "model": name, **scores(pred)})
aug_table = pd.DataFrame(aug_rows)
display(aug_table.pivot(index="augmentasi", columns="model", values="macro_f1").loc[["tanpa", "jitter", "kepekaan"]].round(3))

fig, ax = plt.subplots(figsize=(7, 3.9))
pos = np.arange(3)
for i, (name, color) in enumerate([("Decision Tree", GREY), ("Random Forest", TEAL)]):
    v = aug_table[aug_table.model == name].set_index("augmentasi").loc[["tanpa", "jitter", "kepekaan"], "macro_f1"]
    bars = ax.bar(pos + (i - 0.5) * 0.38, v, 0.36, color=color, label=name)
    ax.bar_label(bars, labels=[fmt(x) for x in v], padding=3, fontsize=10)
ax.set(xticks=pos, xticklabels=["tanpa augmentasi", "jitter (derau)", "variasi kepekaan"], ylim=(0, 1),
       ylabel="macro-F1 (LOSO)", title="Efek augmentasi (hanya pada data latih)")
ax.legend(loc="upper left", ncol=2)
save(fig, "07_augmentasi")
plt.show()

aug_pivot = aug_table.pivot(index="augmentasi", columns="model", values="macro_f1")
RESULTS["augmentation"] = aug_rows
delta = {m: {k: float(aug_pivot.loc[k, m] - aug_pivot.loc["tanpa", m]) for k in ["jitter", "kepekaan"]}
         for m in AUG_MODELS}
result_box(f'''
- Decision Tree: jitter {signed(delta["Decision Tree"]["jitter"])}, variasi kepekaan {signed(delta["Decision Tree"]["kepekaan"])}.
- Random Forest: jitter {signed(delta["Random Forest"]["jitter"])}, variasi kepekaan {signed(delta["Random Forest"]["kepekaan"])}.
- Artinya: data buatan **tidak otomatis menaikkan skor**. Variasi buatan tidak sama dengan perbedaan nyata antar
  sesi ayam. Bagian 9 menunjukkan bahwa **sesi asli tambahan** jauh lebih berguna.
''')
""")

# =====================================================================================
md(r"""
---
# Bagian 9 — Reinforcement learning: agen yang belajar mengambil keputusan

Rubrik dosen: *bagaimana tambahan **iterasi** dan **data** dapat meningkatkan performa sistem sesuai tujuan.*

**Tujuan sistem:** bukan cuma menebak kelas, tapi **mengambil keputusan** yang aman dan tidak boros:
- **Jual:** ayam dianggap masih layak.
- **Cek manual:** minta orang atau lab mengecek dulu. Ada biaya, tapi keputusannya pasti benar.
- **Buang:** ayam dianggap busuk.

**Agen reinforcement learning** belajar sendiri lewat *trial and error*:

| Unsur RL | Di proyek ini |
|---|---|
| **State** (yang dilihat agen) | tebakan Random Forest (Fresh/Semi-fresh/Spoiled) × model **yakin** atau **ragu** (peluang ≥ 0,7 atau < 0,7) → 6 state |
| **Aksi** | Jual, Cek manual, Buang |
| **Reward** (asumsi kami) | lihat tabel di bawah |
| **Cara belajar** | **ε-greedy**: 90% pilih aksi terbaik sejauh ini, 10% coba acak. Nilai tiap aksi diperbarui dengan rata-rata reward yang didapat (seperti materi *multi-armed bandit*). |

**Tabel reward.** "Layak" = Fresh atau Semi-fresh, "busuk" = Spoiled. Angkanya **asumsi kami**: menjual ayam
busuk dianggap paling berbahaya.

| Aksi | Ayam layak | Ayam busuk |
|---|---|---|
| Jual | +1 | **−5** (konsumen dirugikan) |
| Cek manual | +0,5 | +0,5 (aman, tapi ada biaya cek) |
| Buang | −1 (mubazir) | +1 |

**Agar jujur:** agen belajar dari 12 sesi dan **dinilai pada 3 sesi lain** (5 putaran, setiap sesi pernah jadi
ujian). Tebakan Random Forest yang dipakai berasal dari model yang tidak pernah melihat sesi tersebut.
""")
code(r"""
proba_rf = probabilities["Random Forest"]
predicted = proba_rf.argmax(axis=1)
unsure = proba_rf.max(axis=1) < 0.7
state = predicted * 2 + unsure
STATE_NAMES = [f"{c} ({'ragu' if u else 'yakin'})" for c in CLASSES for u in (False, True)]
ACTIONS = ["Jual", "Cek manual", "Buang"]
fit_to_sell = y <= 1
REWARD = np.column_stack([np.where(fit_to_sell, 1.0, -5.0), np.full(len(y), 0.5),
                          np.where(fit_to_sell, -1.0, 1.0)])


def train_agent(train_idx, test_idx, seed, steps=3000, eps=0.1, every=100):
    rng = np.random.default_rng(seed)
    Q, N = np.zeros((6, 3)), np.zeros((6, 3))
    curve = []
    for t in range(1, steps + 1):
        i = rng.choice(train_idx)                       # the agent meets one chicken sample
        s = state[i]
        a = rng.integers(3) if rng.random() < eps else int(np.argmax(Q[s]))
        N[s, a] += 1
        Q[s, a] += (REWARD[i, a] - Q[s, a]) / N[s, a]    # running average of the reward
        if t == 1 or t % every == 0:
            policy = Q.argmax(axis=1)
            curve.append(REWARD[test_idx, policy[state[test_idx]]].mean())
    return Q, np.array(curve)


folds = list(GroupKFold(5).split(X, y, groups))
curves, policies, trust_model, always_check, random_act = [], [], [], [], []
for train_idx, test_idx in folds:
    for seed in range(5):
        Q, c = train_agent(train_idx, test_idx, seed)
        curves.append(c)
        policies.append(Q.argmax(axis=1))
    trust_model.append(np.where(predicted[test_idx] <= 1, REWARD[test_idx, 0], REWARD[test_idx, 2]).mean())
    always_check.append(0.5)
    random_act.append(REWARD[test_idx].mean())
curves = np.array(curves)
steps_axis = np.r_[1, np.arange(100, 3001, 100)]
agent_final = curves[:, -1].mean()

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(steps_axis, curves.mean(axis=0), color=RED, lw=2.6, marker="o", ms=3, label="agen RL (ε-greedy)")
ax.axhline(np.mean(trust_model), color=TEAL, ls="--", lw=1.4, label="selalu ikut tebakan model")
ax.axhline(0.5, color=AMBER, ls="--", lw=1.4, label="selalu cek manual")
ax.axhline(np.mean(random_act), color=GREY, ls=":", lw=1.4, label="aksi acak")
ax.set(xlabel="iterasi belajar agen", ylabel="rata-rata reward di sesi uji",
       title="Makin banyak iterasi, keputusan agen makin baik")
ax.legend(loc="lower right")
save(fig, "08_rl_kurva")
plt.show()

policy_mode = [int(np.bincount([p[s] for p in policies], minlength=3).argmax()) for s in range(6)]
policy_table = pd.DataFrame({"state": STATE_NAMES, "jumlah data": np.bincount(state, minlength=6),
                             "aksi yang dipelajari": [ACTIONS[a] for a in policy_mode]})
display(policy_table)
RESULTS["rl"] = {"reward_start": float(curves[:, 0].mean()), "reward_after_100": float(curves[:, 1].mean()),
                 "reward_final": float(agent_final), "trust_model": float(np.mean(trust_model)),
                 "always_check": 0.5, "random_action": float(np.mean(random_act)),
                 "policy": dict(zip(STATE_NAMES, [ACTIONS[a] for a in policy_mode])),
                 "curve": curves.mean(axis=0).round(4).tolist(), "steps": steps_axis.tolist()}
""")
md(r"""
### Efek tambah data: tambah sesi ayam asli
Pertanyaan kedua di rubrik: apakah **tambahan data** meningkatkan performa? Untuk setiap sesi uji, Random Forest
dilatih dengan 1, 2, 4, … sampai 14 sesi lain yang dipilih acak.
""")
code(r"""
rng = np.random.default_rng(SEED)
curve_rows = []
for k in [1, 2, 4, 6, 8, 10, 12, 14]:
    for test_session in np.unique(groups):
        others = np.setdiff1d(np.unique(groups), [test_session])
        for repeat in range(2):
            chosen = rng.choice(others, k, replace=False)
            train, test = np.isin(groups, chosen), groups == test_session
            if len(np.unique(y[train])) < 3:
                continue
            model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1)
            pred = model.fit(X[train], y[train]).predict(X[test])
            curve_rows.append({"sesi_latih": k, "macro_f1": f1_score(y[test], pred, average="macro")})
session_curve = pd.DataFrame(curve_rows).groupby("sesi_latih").macro_f1.agg(["mean", "std"]).reset_index()
display(session_curve.round(3))

fig, ax = plt.subplots(figsize=(7, 3.9))
ax.errorbar(session_curve.sesi_latih, session_curve["mean"], yerr=session_curve["std"], color=TEAL, lw=2.4,
            marker="o", capsize=4)
for _, r in session_curve.iterrows():
    ax.annotate(fmt(r["mean"], 2), (r.sesi_latih, r["mean"]), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold")
ax.set(xlabel="jumlah sesi ayam untuk latihan", ylabel="macro-F1 di sesi uji", ylim=(0, 1),
       xticks=session_curve.sesi_latih, title="Makin banyak sesi asli, model makin akurat")
save(fig, "09_tambah_sesi")
plt.show()

RESULTS["session_curve"] = session_curve.round(4).to_dict("records")
rl = RESULTS["rl"]
result_box(f'''
- **Tambahan iterasi:** reward agen naik dari {fmt(rl["reward_start"], 2)} (iterasi pertama) ke
  {fmt(rl["reward_after_100"], 2)} setelah 100 iterasi, lalu stabil di **{fmt(rl["reward_final"], 2)}**.
- Agen mengalahkan "selalu ikut tebakan model" ({fmt(rl["trust_model"], 2)}) dan "selalu cek manual" (0,50).
- Agen belajar: tebakan Fresh → **Jual**; Semi-fresh → **Cek manual**; Spoiled yang yakin → **Buang**.
  Cek manual dipakai hanya saat risikonya tinggi.
- **Tambahan data:** macro-F1 naik dari {fmt(session_curve["mean"].iloc[0], 2)} (1 sesi) ke
  **{fmt(session_curve["mean"].iloc[-1], 2)}** (14 sesi). Kurvanya sedikit naik-turun karena sesi latih dipilih
  acak, tapi arahnya jelas naik.
''')
""")

# =====================================================================================
md(r"""
---
# Bagian 10 — Sensor mana yang paling penting?

Random Forest mencatat seberapa sering dan seberapa berguna tiap fitur dipakai untuk memisahkan kelas
(*feature importance*). Kita jumlahkan 3 fitur dari sensor yang sama (level + porsi + naik).
""")
code(r"""
final_rf = make_rf().fit(X, y)
importance = pd.Series(final_rf.feature_importances_, index=features.columns)
per_sensor = pd.Series({s: importance[[c for c in importance.index if c.endswith(s)]].sum() for s in SENSORS})
per_sensor = per_sensor.sort_values()

fig, ax = plt.subplots(figsize=(6.4, 3.8))
bars = ax.barh(per_sensor.index, per_sensor.values,
               color=[RED if s == "MQ-135" else TEAL for s in per_sensor.index])
ax.bar_label(bars, labels=[fmt(v, 2) for v in per_sensor.values], padding=3)
ax.set(xlabel="importance (jumlah = 1)", title="Sumbangan tiap sensor di Random Forest")
save(fig, "10_sensor_penting")
plt.show()

rank = per_sensor.sort_values(ascending=False)
RESULTS["sensor_importance"] = rank.round(4).to_dict()
result_box(f'''
- Sensor paling berpengaruh: **{rank.index[0]}** ({fmt(rank.iloc[0], 2)}), lalu {rank.index[1]} ({fmt(rank.iloc[1], 2)}).
- MQ-135, sensor yang dipakai skripsi Erlangga, ada di urutan ke-{list(rank.index).index("MQ-135") + 1} dari 7
  ({fmt(rank["MQ-135"], 2)}).
- Importance hanya menunjukkan apa yang dipakai model, **bukan** sebab-akibat kimia.
''')
""")

# =====================================================================================
md(r"""
---
# Bagian 11 — Ringkasan semua angka dan kesimpulan
""")
code(r"""
RESULTS["source"] = {"paper": "Anwar & Anwar (2025), Sensing and Bio-Sensing Research 47, 100739",
                     "csv_sha256": SHA256}
(OUT / "ringkasan.json").write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False, default=float))

m = model_table.set_index("model")
display(Markdown(f'''
| Pertanyaan | Jawaban |
|---|---|
| Berapa skor kalau data dibagi acak (bocor)? | akurasi {pct(leak["acak"]["akurasi"])} |
| Berapa skor yang jujur (per sesi)? | Random Forest akurasi {pct(leak["loso"]["akurasi"])}, macro-F1 {fmt(m.macro_f1["Random Forest"])} |
| Apakah ensemble lebih baik? | ensemble terbaik {best_ens.model} {fmt(best_ens.macro_f1)} vs tunggal terbaik {best_single.model} {fmt(best_single.macro_f1)} |
| Seberapa penting pengolahan data? | Random Forest {fmt(list(wrangling.values())[0])} → {fmt(list(wrangling.values())[-1])} |
| Apakah augmentasi membantu? | Random Forest: jitter {signed(delta["Random Forest"]["jitter"])}, kepekaan {signed(delta["Random Forest"]["kepekaan"])} |
| Apakah tambahan iterasi membantu (RL)? | reward {fmt(rl["reward_start"], 2)} → {fmt(rl["reward_final"], 2)}; ikut model saja {fmt(rl["trust_model"], 2)} |
| Apakah tambahan data membantu? | 1 sesi {fmt(session_curve["mean"].iloc[0], 2)} → 14 sesi {fmt(session_curve["mean"].iloc[-1], 2)} |
| Sensor paling berpengaruh? | {rank.index[0]} |
'''))
print("Semua angka tersimpan di hasil/ringkasan.json; grafik di folder hasil/.")
""")
md(r"""
## Kesimpulan
1. **Cara menguji menentukan segalanya.** Data yang dibagi acak memberi skor hampir sempurna yang palsu. Uji per sesi
   memberi angka yang jujur.
2. **Pengolahan data adalah langkah terpenting.** Kalibrasi per sesi (kenaikan dibanding jam pertama) membuat model
   bisa menilai ayam dari sesi yang levelnya berbeda.
3. **Ensemble lebih baik dari model tunggal**, terutama banyak pohon (Random Forest) dibanding satu pohon.
4. **Augmentasi buatan tidak menggantikan data asli.** Yang menaikkan skor adalah **sesi ayam asli tambahan**.
5. **Agen reinforcement learning** belajar kapan cukup percaya model dan kapan perlu cek manual, sehingga
   keputusannya lebih aman dibanding sekadar mengikuti tebakan model.

## Batasan
- Hanya **15 sesi**, sehingga hasil antar sesi masih naik-turun.
- Label berasal dari **lama penyimpanan**, bukan uji laboratorium per sampel.
- Angka reward pada agen RL adalah **asumsi kami**. Kalau biaya berubah, kebijakannya bisa berubah.
- Dataset bukan data skripsi kami sendiri. Pipeline ini siap dipakai untuk data skripsi Erlangga nanti.

## Pengembangan
- Jalankan pipeline yang sama pada data 16 batch skripsi Erlangga (sensor MQ-135 + DHT22).
- Tambah sesi rekaman di kondisi suhu yang berbeda.
- Label dari panel sensori atau uji mikrobiologi, bukan hanya dari waktu.

## Referensi
- Anwar, H., & Anwar, T. (2025). Quality assessment of chicken using machine learning and electronic nose.
  *Sensing and Bio-Sensing Research, 47*, 100739. https://doi.org/10.1016/j.sbsr.2025.100739
- Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.
- Materi kuliah Machine Learning II (github.com/jendralhxr/machinelearning).
""")

# =====================================================================================
nb = {"cells": CELLS, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
    lines = c["source"].split("\n")
    c["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
(ROOT / "notebooks" / NAME).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", NAME, len(nb["cells"]), "cells")
