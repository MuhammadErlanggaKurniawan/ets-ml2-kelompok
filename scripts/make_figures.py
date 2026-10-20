"""Render the deck charts as PNG from deck/data.json.

Google Slides drops the category labels of native PowerPoint charts on import, so the deck embeds
these images instead.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "deck" / "data.json").read_text())
OUT = ROOT / "deck" / "figures"
OUT.mkdir(exist_ok=True)

TEAL, GREY, AMBER, RED, BLUE, SAGE = "#0F5257", "#9AA5A6", "#E3A33B", "#C4553B", "#2E6F95", "#6C8F7D"
INK, MUTED, GRID = "#16262B", "#56676A", "#E1E8E6"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titlecolor": "#0F3D3E", "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "legend.frameon": False, "legend.fontsize": 11,
})
comma = lambda n: FuncFormatter(lambda x, _: f"{x:.{n}f}".replace(".", ","))  # noqa: E731
fmt = lambda x, n=3: f"{x:.{n}f}".replace(".", ",")  # noqa: E731


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def label_bars(ax, bars, n=3, inside=False, color=INK, horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        light_bar = matplotlib.colors.to_hex(b.get_facecolor()).upper() == AMBER
        if horizontal:
            ax.text(v - 0.01 if inside else v + 0.01, b.get_y() + b.get_height() / 2, fmt(v, n),
                    va="center", ha="right" if inside else "left", fontsize=11,
                    color=(INK if light_bar else "white") if inside else color)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, fmt(v, n), ha="center", va="bottom",
                    fontsize=10.5, color=color)


SINGLE = ["LogReg", "SVM-RBF", "KNN", "DecisionTree"]
DISP = {"LogReg": "Logistic Reg.", "DecisionTree": "Decision Tree", "RandomForest": "Random Forest",
        "LightGBM-Optuna": "LightGBM + Optuna"}

# --- slide 7: augmentation
aug = {(r["model"], r["variant"]): r["f1"] for r in D["enose_augment"]}
var = ["none", "jitter", "scaling", "jitter+scaling", "jitter+scaling x4"]
lab = ["tanpa", "jitter", "scaling", "jitter +\nscaling", "jitter +\nscaling ×4"]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
x = range(len(var))
b1 = ax.bar([i - 0.2 for i in x], [aug[("LogReg", v)] for v in var], 0.38, color=GREY, label="Logistic Regression")
b2 = ax.bar([i + 0.2 for i in x], [aug[("LightGBM", v)] for v in var], 0.38, color=TEAL, label="LightGBM")
label_bars(ax, b1); label_bars(ax, b2)
ax.set_xticks(list(x), lab); ax.set_ylim(0, 0.7); ax.yaxis.set_major_formatter(comma(1))
ax.set_title("Augmentasi (hanya data latih) · macro-F1 LOSO, 4 kelas"); ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
save(fig, "s07_augment")

# --- slide 8: e-nose binary ensembles
rows = D["enose_models_binary"]
fig, ax = plt.subplots(figsize=(7.6, 4.9))
cols = [GREY if r["model"] in SINGLE else TEAL for r in rows]
bars = ax.bar([DISP.get(r["model"], r["model"]) for r in rows], [r["f1"] for r in rows], color=cols,
              yerr=[r["sd"] for r in rows], capsize=3, error_kw={"ecolor": "#7A8A8C", "lw": 1})
for b, r in zip(bars, rows):
    ax.text(b.get_x() + b.get_width() / 2, 0.04, fmt(r["f1"]), ha="center", color="white", fontsize=11,
            fontweight="bold")
ax.set_ylim(0, 1.15); ax.yaxis.set_major_formatter(comma(1)); ax.tick_params(axis="x", rotation=30)
ax.set_title("Biner segar vs tidak segar · macro-F1 LOSO (garis = sd antar seri)")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Rectangle((0, 0), 1, 1, color=TEAL)],
          labels=["Model tunggal", "Ensemble"], loc="upper left", ncol=2)
save(fig, "s08_enose_ensemble")

# --- slide 9: solubility RMSE
rows = D["sol_models"]
fig, ax = plt.subplots(figsize=(7.6, 4.9))
names = [DISP.get(r["model"], r["model"]) for r in rows]
bars = ax.barh(names, [r["rmse"] for r in rows],
               color=[GREY if r["model"] in ["Ridge", "KNN", "SVR-RBF", "DecisionTree"] else TEAL for r in rows])
best = min(rows, key=lambda r: r["rmse"])["model"]
for b, r in zip(bars, rows):
    b.set_edgecolor(AMBER if r["model"] == best else "none"); b.set_linewidth(2.5)
label_bars(ax, bars, horizontal=True, inside=True)
ax.invert_yaxis(); ax.set_xlim(0, 1.8); ax.xaxis.set_major_formatter(comma(1))
ax.set_title("RMSE uji scaffold (lebih kecil lebih baik)"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
save(fig, "s09_sol_ensemble")

# --- slide 10: per-bin MAE
pb = D["sol_per_bin"]
lab = ["sangat tidak\nlarut (< −6)", "tidak larut\n(−6..−4)", "sedikit larut\n(−4..−2)", "larut\n(−2..0)",
       "sangat larut\n(≥ 0)"]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
x = range(len(pb))
b1 = ax.bar([i - 0.2 for i in x], [r["mae_unweighted"] for r in pb], 0.38, color=GREY, label="tanpa bobot")
b2 = ax.bar([i + 0.2 for i in x], [r["mae_weighted"] for r in pb], 0.38, color=AMBER, label="bobot frekuensi-invers")
label_bars(ax, b1, 2); label_bars(ax, b2, 2)
ax.set_xticks(list(x), lab, fontsize=10.5); ax.set_ylim(0, 1.7); ax.yaxis.set_major_formatter(comma(1))
ax.set_title("MAE per kategori LogS (data uji scaffold)"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.legend(loc="upper center", ncol=2)
save(fig, "s10_sol_imbalance")

# --- slides 12/13: AL curves
POL = [("random", "random", GREY), ("uncertainty", "uncertainty", BLUE), ("diversity", "diversity", SAGE),
       ("bandit-eps", "bandit ε-greedy", AMBER), ("bandit-ucb", "bandit UCB1", RED)]


def al_plot(al, name, title, ylabel, ylim, ref_label, n_dec, size):
    fig, ax = plt.subplots(figsize=size)
    for key, lab_, col in POL:
        ax.plot(al["n_labelled"], al["curves"][key], marker="o", ms=3.5, lw=2.4 if key.startswith("bandit") else 1.8,
                color=col, label=lab_)
    ax.axhline(al["full_pool_test"], ls="--", lw=1, color=INK)
    ax.text(al["n_labelled"][-1], al["full_pool_test"], f"  {ref_label}", va="bottom", ha="right", fontsize=10,
            color=INK)
    ax.set_ylim(*ylim); ax.yaxis.set_major_formatter(comma(n_dec))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
    ax.set_xlabel("jumlah label"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(color=GRID); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=10, handlelength=1.5)
    save(fig, name)


al_plot(D["al_solubility"], "s12_al_solubility", "AqSolDB · RMSE uji scaffold vs jumlah label (5 seed)",
        "RMSE (↓)", (1.3, 1.9), f"semua pool ({D['al_solubility']['pool_size']:,} label)".replace(",", "."), 1, (7.9, 4.9))
al_plot(D["al_enose"], "s13_al_enose", "E-nose · macro-F1 seri uji vs jumlah label",
        "macro-F1 (↑)", (0.7, 0.95), "semua pool", 2, (6.4, 4.6))

# --- slide 13 right: series curve
sc = D["series_curve"]
fig, ax = plt.subplots(figsize=(5.6, 4.6))
bars = ax.barh([r["label"] for r in sc], [r["f1"] for r in sc], color=[TEAL] * 4 + [AMBER] * 3)
label_bars(ax, bars, horizontal=True, inside=True)
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(comma(1))
ax.set_title("Tambah seri vs tambah baris (XGBoost biner)"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
save(fig, "s13_series_curve")

# --- slide 14: SHAP
st = list(D["sol_shap_top"].items())[:8]
fig, ax = plt.subplots(figsize=(5.9, 4.3))
bars = ax.barh([k for k, _ in st], [v for _, v in st], color=TEAL)
label_bars(ax, bars, 2, horizontal=True)
ax.invert_yaxis(); ax.set_xlim(0, 0.7); ax.xaxis.set_major_formatter(comma(1))
ax.set_title("Kelarutan · mean |SHAP| (LightGBM + Optuna)"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
save(fig, "s14_shap_sol")
se = list(D["enose_shap_sensors"].items())
fig, ax = plt.subplots(figsize=(5.9, 4.3))
bars = ax.barh([k for k, _ in se], [v for _, v in se], color=[RED if k == "MQ135" else AMBER for k, _ in se])
label_bars(ax, bars, 2, horizontal=True)
ax.invert_yaxis(); ax.set_xlim(0, 5); ax.xaxis.set_major_formatter(comma(0))
ax.set_title("E-nose biner · mean |SHAP| per sensor (seri uji)"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
save(fig, "s14_shap_enose")
