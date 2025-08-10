import os
import re
import unicodedata
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

USE_KNN_NUM = False
KNN_K = 5
RARE_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def _strip_accents(x):
    if not isinstance(x, str):
        return x
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", x) if not unicodedata.combining(ch)
    )


def _norm_text(x):
    if not isinstance(x, str):
        return x
    x = _strip_accents(x).lower().strip()
    return re.sub(r"\s+", " ", x)


# ---------------------------------------------------------------------------
# Data preparation utilities
# ---------------------------------------------------------------------------

def infer_types(df: pd.DataFrame, target: str | None = None) -> Tuple[List[str], List[str]]:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    cat = [
        c
        for c in df.columns
        if (df[c].dtype == "O" or pd.api.types.is_string_dtype(df[c])) and c != target
    ]
    for c in df.columns:
        if c not in cat and c not in num and c != target:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 20:
                cat.append(c)
    num = [c for c in num if c not in cat and c != target]
    return cat, num


def standardize_cats(df: pd.DataFrame, cat_cols: List[str], rare_thr: float = RARE_THRESHOLD):
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            s = df[c].astype("object")
            s = s.map(lambda v: _norm_text(v) if pd.notna(v) else v)
            s = s.replace({"": np.nan, "na": np.nan, "n/a": np.nan, "none": np.nan, "null": np.nan})
            freq = s.value_counts(normalize=True, dropna=True)
            rare = set(freq[freq < rare_thr].index)
            s = s.map(lambda v: ("other" if v in rare else v) if pd.notna(v) else v)
            df[c] = s
    return df


def impute(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str], use_knn: bool = USE_KNN_NUM, k: int = KNN_K):
    df = df.copy()
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    if num_cols:
        if use_knn:
            df[num_cols] = KNNImputer(n_neighbors=k, weights="distance").fit_transform(df[num_cols])
        else:
            df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    return df


def ensure_binary_target(df: pd.DataFrame, target_final: str):
    y = df[target_final]
    if y.dropna().isin([0, 1, True, False]).all():
        df[target_final] = y.astype(int)
        return df
    m = {
        "yes": 1,
        "y": 1,
        "si": 1,
        "sí": 1,
        "true": 1,
        "fraud": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "legit": 0,
    }
    yy = y.astype(str).str.lower().map(m)
    yy = yy.fillna(pd.to_numeric(y, errors="coerce"))
    yy = yy.fillna(0).astype(float)
    df[target_final] = (yy > 0.5).astype(int)
    return df


def preprocess(csv_in: str, target_raw: str, target_final: str) -> pd.DataFrame:
    df = pd.read_csv(csv_in)
    if target_raw in df.columns and target_final not in df.columns:
        df = df.rename(columns={target_raw: target_final})
    if target_final not in df.columns:
        raise KeyError(f"Columna objetivo ausente: {target_final}")
    df = df.drop_duplicates(keep="first")
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    cats, nums = infer_types(df, target=target_final)
    df = standardize_cats(df, cats, rare_thr=RARE_THRESHOLD)
    df = impute(df, cats, nums, use_knn=USE_KNN_NUM, k=KNN_K)
    df = ensure_binary_target(df, target_final)
    return df


# ---------------------------------------------------------------------------
# Modeling utilities
# ---------------------------------------------------------------------------

def build_pipeline(cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    pipe = Pipeline(
        steps=[
            ("prep", ct),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, solver="liblinear", class_weight="balanced"
                ),
            ),
        ]
    )
    return pipe


def ks_stat(y_true, y_sc):
    d = pd.DataFrame({"y": y_true, "s": y_sc}).sort_values("s", ascending=False)
    pos = (d["y"] == 1).sum()
    neg = (d["y"] == 0).sum()
    if pos == 0 or neg == 0:
        return 0.0
    d["tpr"] = (d["y"] == 1).cumsum() / pos
    d["fpr"] = (d["y"] == 0).cumsum() / neg
    return float((d["tpr"] - d["fpr"]).max())


def fit_eval(df: pd.DataFrame, target: str, cat_cols: List[str], num_cols: List[str], thr: float | None = None):
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    pipe = build_pipeline(cat_cols, num_cols)
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    if thr is None:
        grid = np.linspace(0.01, 0.99, 99)
        f1_vals = []
        for t in grid:
            p = (prob >= t).astype(int)
            f1_vals.append(f1_score(y_test, p, pos_label=1))
        thr = float(grid[int(np.argmax(f1_vals))])
    pred = (prob >= thr).astype(int)
    auc = float(roc_auc_score(y_test, prob))
    cm = confusion_matrix(y_test, pred)
    metrics = {
        "threshold": thr,
        "precision_1": float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(y_test, pred, pos_label=1)),
        "f1_1": float(f1_score(y_test, pred, pos_label=1)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": auc,
        "ks": ks_stat(y_test, prob),
        "confusion_matrix": cm.tolist(),
    }
    report = ""  # placeholder for compatibility
    fpr, tpr, _ = roc_curve(y_test, prob)
    return pipe, metrics, report, (fpr, tpr)
