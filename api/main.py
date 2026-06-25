from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, PredictionResponse, TaskFeatures
from models.predict import get_predictor

app = FastAPI(
    title="PERT-ML Project Optimizer",
    description=(
        "Combines classical PERT scheduling with ML to predict task durations "
        "and flag schedule risk. Built as a production MLOps portfolio project."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    try:
        predictor = get_predictor()
        model_loaded = predictor.model is not None
    except Exception:
        model_loaded = False

    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_duration(task: TaskFeatures):
    try:
        predictor = get_predictor()
        result = predictor.predict(task.to_feature_dict())
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python models/train.py` first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result["risk_flag"]:
        interpretation = (
            f"Risk flagged: ML predicts {result['ml_prediction']}d vs PERT estimate "
            f"of {result['pert_expected']}d ({result['divergence_pct']}% divergence). "
            f"Historical patterns suggest this task type runs longer than estimated."
        )
    else:
        interpretation = (
            f"Low risk: ML prediction ({result['ml_prediction']}d) aligns with "
            f"PERT estimate ({result['pert_expected']}d). "
            f"Divergence within acceptable range ({result['divergence_pct']}%)."
        )

    return PredictionResponse(**result, interpretation=interpretation)


@app.get("/model/info", tags=["Model"])
def model_info():
    try:
        predictor = get_predictor()
        return predictor.metadata
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python models/train.py` first.",
        )