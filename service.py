import json
import os
from typing import Any, Dict, List, Optional

try:
    import joblib
except Exception:  # fallback when joblib is unavailable
    import pickle as joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml_utils import preprocess, infer_types, fit_eval

CSV_PATH = "fraud_train.csv"
TARGET_RAW = "FraudFound_P"
TARGET = "is_fraud"
MODEL_PATH = "model.joblib"
META_PATH = "metadata.json"

app = FastAPI(title="Fraud Scoring API", version="1.0")

# Global state
DATA: Optional[pd.DataFrame] = None
CAT_COLS_ALL: List[str] = []
NUM_COLS_ALL: List[str] = []
MODEL = None
SELECTED_COLS: List[str] = []
CAT_COLS: List[str] = []
NUM_COLS: List[str] = []
THRESHOLD: float | None = None
MODEL_VERSION: int = 0


class TrainRequest(BaseModel):
    selected_columns: Optional[List[str]] = None


class PredictItem(BaseModel):
    features: Dict[str, Any]


class PredictRequest(BaseModel):
    items: List[PredictItem]


class ThresholdRequest(BaseModel):
    threshold: float


@app.on_event("startup")
def startup_event():
    global DATA, CAT_COLS_ALL, NUM_COLS_ALL
    global MODEL, SELECTED_COLS, CAT_COLS, NUM_COLS, THRESHOLD, MODEL_VERSION

    DATA = preprocess(CSV_PATH, TARGET_RAW, TARGET)
    CAT_COLS_ALL, NUM_COLS_ALL = infer_types(DATA, target=TARGET)

    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        MODEL = joblib.load(MODEL_PATH)
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        SELECTED_COLS = meta.get("selected_columns", [])
        CAT_COLS = meta.get("cat_cols", [])
        NUM_COLS = meta.get("num_cols", [])
        THRESHOLD = meta.get("threshold")
        MODEL_VERSION = meta.get("model_version", 0)


def _persist_model():
    if MODEL is None:
        return
    meta = {
        "selected_columns": SELECTED_COLS,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "threshold": THRESHOLD,
        "model_version": MODEL_VERSION,
    }
    joblib.dump(MODEL, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f)


@app.get("/")
def root():
    return {"message": "Fraud detection API", "docs_url": "/docs"}


@app.get("/features")
def list_features():
    if DATA is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    return {"target": TARGET, "categorical": CAT_COLS_ALL, "numerical": NUM_COLS_ALL}


@app.post("/train")
def train(req: TrainRequest):
    global MODEL, SELECTED_COLS, CAT_COLS, NUM_COLS, THRESHOLD, MODEL_VERSION
    if DATA is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    cols = req.selected_columns or [c for c in DATA.columns if c != TARGET]
    missing = [c for c in cols if c not in DATA.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")
    cols = [c for c in cols if c != TARGET]

    cat_cols = [c for c in CAT_COLS_ALL if c in cols]
    num_cols = [c for c in NUM_COLS_ALL if c in cols]

    df_train = DATA[cols + [TARGET]]
    model, metrics, _, _ = fit_eval(df_train, TARGET, cat_cols, num_cols)

    MODEL = model
    SELECTED_COLS = cols
    CAT_COLS = cat_cols
    NUM_COLS = num_cols
    THRESHOLD = metrics["threshold"]
    MODEL_VERSION += 1

    _persist_model()

    return {
        "used_columns": SELECTED_COLS,
        "categorical": CAT_COLS,
        "numerical": NUM_COLS,
        "threshold": THRESHOLD,
        "auc": metrics["auc"],
        "f1_1": metrics["f1_1"],
        "precision_1": metrics["precision_1"],
        "recall_1": metrics["recall_1"],
        "accuracy": metrics["accuracy"],
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    global MODEL, THRESHOLD
    if DATA is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    if MODEL is None:
        train(TrainRequest())

    results = []
    for item in req.items:
        feat = item.features
        row = {c: feat.get(c, np.nan) for c in SELECTED_COLS}
        missing = [c for c in SELECTED_COLS if c not in feat or feat[c] is None]
        X = pd.DataFrame([row])
        prob = float(MODEL.predict_proba(X)[0, 1])
        label = int(prob >= THRESHOLD)
        results.append(
            {
                "fraud_probability": prob,
                "predicted_label": label,
                "threshold": THRESHOLD,
                "used_columns": SELECTED_COLS,
                "missing_or_defaulted": missing,
            }
        )
    return {"predictions": results}


@app.post("/config/threshold")
def set_threshold(req: ThresholdRequest):
    global THRESHOLD
    THRESHOLD = req.threshold
    _persist_model()
    return {"threshold": THRESHOLD}
