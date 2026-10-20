"""Track B: aqueous solubility (LogS) regression on AqSolDB with scaffold-based splitting.

Molecules sharing a Bemis-Murcko scaffold are near-duplicates for a model, so a random split
leaks chemistry from train to test. The scaffold split keeps each scaffold on one side.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import RAW, ROOT, SEED

RDLogger.DisableLog("rdApp.*")

AQSOL_DESC = ["MolWt", "MolLogP", "MolMR", "HeavyAtomCount", "NumHAcceptors", "NumHDonors",
              "NumHeteroatoms", "NumRotatableBonds", "NumValenceElectrons", "NumAromaticRings",
              "NumSaturatedRings", "NumAliphaticRings", "RingCount", "TPSA", "LabuteASA",
              "BalabanJ", "BertzCT"]
# Solubility categories used by Sorkun et al. (2019) for AqSolDB
BINS = [-np.inf, -6, -4, -2, 0, np.inf]
BIN_LABELS = ["very_insoluble(<-6)", "insoluble(-6..-4)", "slightly(-4..-2)",
              "soluble(-2..0)", "highly(>=0)"]
CACHE = ROOT / "data" / "processed"


def load() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(RAW / "aqsoldb_train.csv"), pd.read_csv(RAW / "aqsoldb_test.csv")],
                   ignore_index=True)
    df = df.drop_duplicates("InChIKey").rename(columns={"Y": "logS"}).reset_index(drop=True)
    df["sol_bin"] = pd.cut(df["logS"], BINS, labels=BIN_LABELS, right=False)
    return df


def scaffold_of(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"INVALID:{smiles}"
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return f"INVALID:{smiles}"


def add_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scaffold"] = df["SMILES"].map(scaffold_of)
    # acyclic molecules have an empty scaffold; keep each as its own group so they are not
    # forced together into one giant group
    empty = df["scaffold"] == ""
    df.loc[empty, "scaffold"] = "ACYCLIC:" + df.loc[empty, "InChIKey"]
    return df


def rdkit_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Full RDKit 2D descriptor set, cached to data/processed."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "aqsoldb_rdkit2d_v2.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        if len(cached) == len(df):
            return cached
    names = [n for n, _ in Descriptors.descList]
    rows = []
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        rows.append([fn(mol) if mol is not None else np.nan for _, fn in Descriptors.descList])
    X = pd.DataFrame(rows, columns=names).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.isna().mean() < 0.01].clip(-1e6, 1e6)
    X = X.loc[:, X.std() > 0].astype("float32")
    X.to_parquet(path)
    return X


def scaffold_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = SEED):
    """Bemis-Murcko split: big scaffold groups go to train first, small ones fill test.

    The ordering puts rare scaffolds in the test set, the harder and more realistic case
    (new chemistry). `seed` only breaks ties between groups of equal size.
    """
    groups = defaultdict(list)
    for i, s in enumerate(df["scaffold"]):
        groups[s].append(i)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)
    n_train = int(round((1 - test_frac) * len(df)))
    train, test = [], []
    for k in keys:
        (train if len(train) + len(groups[k]) <= n_train else test).extend(groups[k])
    return np.array(sorted(train)), np.array(sorted(test))


def random_split(n: int, test_frac: float = 0.2, seed: int = SEED):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(round(test_frac * n))
    return np.sort(idx[k:]), np.sort(idx[:k])


def reg_scores(y, p) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


def per_bin_errors(df_test: pd.DataFrame, pred) -> pd.DataFrame:
    t = df_test.assign(pred=pred, abs_err=np.abs(df_test["logS"].to_numpy() - pred))
    return (t.groupby("sol_bin", observed=False)
             .agg(n=("abs_err", "size"), mae=("abs_err", "mean"))
             .reset_index())


def inverse_frequency_weights(bins: pd.Series) -> np.ndarray:
    """Sample weight ~ 1/frequency of the solubility bin, normalised to mean 1."""
    freq = bins.map(bins.value_counts(normalize=True)).astype(float).to_numpy()
    w = 1.0 / freq
    return w / w.mean()
