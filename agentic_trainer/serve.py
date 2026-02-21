from __future__ import annotations

from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from .schema import DatasetSchema, load_schema, validate_rows

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[Any]
    errors: Optional[List[str]] = None

def create_app(model_path: str, schema_path: str) -> FastAPI:
    app = FastAPI(title="Agentic Trainer Inference API")

    pipe = joblib.load(model_path)
    schema: DatasetSchema = load_schema(schema_path)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/schema")
    def schema_endpoint():
        return {"columns": [c.__dict__ for c in schema.columns]}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        errs = validate_rows(schema, req.rows)
        if errs:
            return PredictResponse(predictions=[], errors=errs)

        df = pd.DataFrame(req.rows)
        # Ensure expected columns exist (missing non-required columns are filled with NaN)
        for c in schema.columns:
            if c.name not in df.columns:
                df[c.name] = None
        df = df[[c.name for c in schema.columns]]

        preds = pipe.predict(df).tolist()
        return PredictResponse(predictions=preds, errors=None)

    return app
