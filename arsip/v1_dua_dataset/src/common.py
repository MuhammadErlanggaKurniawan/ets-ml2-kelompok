"""Shared paths, seeds and model zoo used by every track."""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
SEED = 42

# SVC(probability=True) is deprecated in sklearn 1.9 but still valid for the stacking base learner
warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def single_classifiers(class_weight=None):
    """Single (non-ensemble) baselines."""
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight=class_weight)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(
            C=10, gamma="scale", class_weight=class_weight, random_state=SEED)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8, class_weight=class_weight, random_state=SEED),
    }


def ensemble_classifiers(class_weight=None, stack_cv=5):
    """Bagging, boosting and stacking ensembles (deck slides 5-13 of the lecture)."""
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                class_weight=class_weight, random_state=SEED)
    return {
        "RandomForest": rf,
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                                 random_state=SEED, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                       class_weight=class_weight, random_state=SEED,
                                       verbose=-1, n_jobs=4),
        "Stacking": StackingClassifier(
            estimators=[
                ("lr", make_pipeline(StandardScaler(), LogisticRegression(
                    max_iter=2000, class_weight=class_weight))),
                ("svm", make_pipeline(StandardScaler(), SVC(
                    C=10, probability=True, class_weight=class_weight, random_state=SEED))),
                ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                              class_weight=class_weight, random_state=SEED)),
            ],
            final_estimator=LogisticRegression(max_iter=2000),
            cv=stack_cv, n_jobs=-1),
    }


def single_regressors():
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "KNN": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             KNeighborsRegressor(n_neighbors=10)),
        "SVR-RBF": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 SVR(C=10, gamma="scale")),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=SEED),
    }


def ensemble_regressors(stack_cv=5):
    return {
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                              n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                                random_state=SEED, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                      random_state=SEED, verbose=-1, n_jobs=4),
        "Stacking": StackingRegressor(
            estimators=[
                ("ridge", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))),
                ("svr", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                       SVR(C=10, gamma="scale"))),
                ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                             n_jobs=-1, random_state=SEED)),
                ("lgbm", lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                                           random_state=SEED, verbose=-1, n_jobs=4)),
            ],
            final_estimator=Ridge(alpha=1.0), cv=stack_cv, n_jobs=-1),
    }


def mean_sd(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}"
