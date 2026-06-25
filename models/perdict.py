import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

MODEL_PATH = Path(__file__).parent / "xgb_model.json"
ENCODER_PATH = Path(__file__).parent / "label_encoders.pkl"
METADATA_PATH = Path(__file__).parent / "feature_metadata.json"


class DurationPredictor:
    def __init__(self):
        self.model = xgb.XGBRegressor()
        self.model.load_model(MODEL_PATH)
        self.encoders = joblib.load(ENCODER_PATH)
        with open(METADATA_PATH) as f:
            self.metadata = json.load(f)
        self.feature_cols = self.metadata["feature_cols"]
        self.categorical_features = self.metadata["categorical_features"]

    def predict(self, features: Dict) -> Dict:
        df = pd.DataFrame([features])

        for col in self.categorical_features:
            le = self.encoders[col]
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                df[col] = 0

        X = df[self.feature_cols]
        prediction = float(self.model.predict(X)[0])
        pert_expected = features.get("pert_expected", prediction)

        divergence = abs(prediction - pert_expected)
        pct_divergence = divergence / pert_expected if pert_expected > 0 else 0

        return {
            "ml_prediction": round(prediction, 2),
            "pert_expected": round(pert_expected, 2),
            "divergence_days": round(divergence, 2),
            "divergence_pct": round(pct_divergence * 100, 1),
            "risk_flag": pct_divergence > 0.20,
        }


_predictor: DurationPredictor = None


def get_predictor() -> DurationPredictor:
    global _predictor
    if _predictor is None:
        _predictor = DurationPredictor()
    return _predictor