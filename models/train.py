import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PATH = Path(__file__).parent.parent / "data" / "sample_tasks.csv"
MODEL_PATH = Path(__file__).parent / "xgb_model.json"
ENCODER_PATH = Path(__file__).parent / "label_encoders.pkl"
METADATA_PATH = Path(__file__).parent / "feature_metadata.json"

CATEGORICAL_FEATURES = ["task_type", "complexity"]
NUMERIC_FEATURES = [
    "team_size",
    "dependency_count",
    "is_external",
    "optimistic",
    "most_likely",
    "pessimistic",
    "pert_expected",
    "pert_variance",
]
TARGET = "actual_duration"


def load_and_prepare(path: Path):
    df = pd.read_csv(path)

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    return X, y, encoders, feature_cols


def train():
    print("Loading data...")
    X, y, encoders, feature_cols = load_and_prepare(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} samples, evaluating on {len(X_test)}...")

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"  MAE:  {mae:.3f} days")
    print(f"  RMSE: {rmse:.3f} days")
    print(f"  R²:   {r2:.3f}")

    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print(f"\nTop Feature Importances:")
    for feat, imp in list(importances.items())[:5]:
        print(f"  {feat}: {imp:.4f}")

    model.save_model(MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)

    metadata = {
        "feature_cols": feature_cols,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "metrics": {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)},
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
        "training_samples": len(X_train),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved:")
    print(f"  Model    → {MODEL_PATH}")
    print(f"  Encoders → {ENCODER_PATH}")
    print(f"  Metadata → {METADATA_PATH}")


if __name__ == "__main__":
    train()