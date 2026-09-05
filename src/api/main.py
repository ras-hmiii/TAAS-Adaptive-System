"""HTTP interface for model health and prediction operations."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.train_tabular import ENCODERS_PATH, MODEL_PATH
from src.experiments.storage import load_experiments
from src.explainability.shap_tabular import TabularExplainer


class PredictionRequest(BaseModel):
    """A single model-ready tabular observation."""

    features: dict[str, float | int | str] = Field(min_length=1)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_available: bool = True


app = FastAPI(
    title="TAAS API",
    description="Backend operations for the Traceable and Adaptive AI System.",
    version="0.1.0",
)


def _load_model() -> Any:
    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="No trained model is available. Run `python -m src.models.train_tabular` first.",
        )
    return joblib.load(model_path)


def _prepare_features(features: dict[str, float | int | str], model: Any) -> pd.DataFrame:
    frame = pd.DataFrame([features])
    expected = list(getattr(model, "feature_names_in_", frame.columns))
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})

    encoders_path = Path(ENCODERS_PATH)
    if encoders_path.is_file():
        encoders = joblib.load(encoders_path)
        for column, encoder in encoders.items():
            if column not in frame:
                continue
            values = frame[column].astype(str)
            unknown = sorted(set(values) - set(encoder.classes_))
            if unknown:
                raise HTTPException(
                    status_code=422,
                    detail={"unknown_values": {column: unknown}},
                )
            frame[column] = encoder.transform(values)
    return frame.loc[:, expected]


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": "available" if Path(MODEL_PATH).is_file() else "unavailable",
    }


@app.get("/models")
def models() -> dict[str, list[dict[str, str]]]:
    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        return {"models": []}
    return {
        "models": [
            {
                "name": "tabular_rf",
                "path": str(model_path),
                "status": "available",
            }
        ]
    }


@app.post("/predictions", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = _load_model()
    features = _prepare_features(request.features, model)
    try:
        probability = float(model.predict_proba(features)[0, 1])
        prediction = int(model.predict(features)[0])
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return PredictionResponse(prediction=prediction, probability=probability)


@app.post("/explanations")
def explain(request: PredictionRequest) -> dict[str, Any]:
    model = _load_model()
    features = _prepare_features(request.features, model)
    try:
        explanation = TabularExplainer(
            model, feature_names=list(features.columns)
        ).explain_instance(features.iloc[0])
    except (TypeError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return explanation


@app.get("/experiments")
def experiments() -> dict[str, list[dict[str, Any]]]:
    return {"experiments": load_experiments()}


@app.get("/results")
def results() -> dict[str, list[dict[str, Any]]]:
    return {"results": load_experiments()}


@app.get("/drift")
def drift() -> dict[str, list[dict[str, Any]]]:
    records = load_experiments()
    events = [event for record in records for event in record.get("drift_events", [])]
    return {"events": events}


@app.get("/adaptation")
def adaptation() -> dict[str, list[dict[str, Any]]]:
    records = load_experiments()
    events = [event for record in records for event in record.get("adaptation_events", [])]
    return {"events": events}
