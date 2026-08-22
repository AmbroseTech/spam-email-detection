"""FastAPI service exposing the trained classifier."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from spam_detector import __version__
from spam_detector.predict import DEFAULT_MODEL_PATH, SpamDetector


class PredictRequest(BaseModel):
    messages: list[str] = Field(..., min_length=1, description="messages to classify")


class PredictionOut(BaseModel):
    text: str
    label: str
    spam_probability: float


class PredictResponse(BaseModel):
    threshold: float
    predictions: list[PredictionOut]


def create_app(model_path: str | Path = DEFAULT_MODEL_PATH) -> FastAPI:
    app = FastAPI(title="Spam Email Detection API", version=__version__)
    state: dict[str, SpamDetector] = {}

    def detector() -> SpamDetector:
        if "detector" not in state:
            try:
                state["detector"] = SpamDetector.load(model_path)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return state["detector"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_available": Path(model_path).exists()}

    @app.get("/model")
    def model_info() -> dict:
        return detector().metadata

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model = detector()
        predictions = model.predict(request.messages)
        return PredictResponse(
            threshold=model.threshold,
            predictions=[
                PredictionOut(
                    text=p.text, label=p.label, spam_probability=round(p.spam_probability, 6)
                )
                for p in predictions
            ],
        )

    return app


app = create_app(os.environ.get("SPAM_MODEL_PATH", DEFAULT_MODEL_PATH))
