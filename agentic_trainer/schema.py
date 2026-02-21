from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

@dataclass
class ColumnSchema:
    name: str
    kind: str               # "numeric" | "categorical" | "datetime" | "unknown"
    required: bool
    example_values: List[Any]
    min: Optional[float] = None
    max: Optional[float] = None
    top_categories: Optional[List[str]] = None

@dataclass
class DatasetSchema:
    columns: List[ColumnSchema]

def infer_schema(df: pd.DataFrame, feature_columns: List[str]) -> DatasetSchema:
    cols: List[ColumnSchema] = []
    for c in feature_columns:
        s = df[c]
        required = not s.isna().all()
        kind = "unknown"

        if pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_bool_dtype(s):
            kind = "categorical"

        examples = s.dropna().head(5).tolist()

        cs = ColumnSchema(
            name=c,
            kind=kind,
            required=required,
            example_values=examples,
        )

        if kind == "numeric":
            s2 = pd.to_numeric(s, errors="coerce")
            cs.min = float(np.nanmin(s2.values)) if np.isfinite(np.nanmin(s2.values)) else None
            cs.max = float(np.nanmax(s2.values)) if np.isfinite(np.nanmax(s2.values)) else None

        if kind == "categorical":
            top = s.dropna().astype(str).value_counts().head(20).index.tolist()
            cs.top_categories = top

        cols.append(cs)

    return DatasetSchema(columns=cols)

def save_schema(schema: DatasetSchema, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(schema), f, indent=2)

def load_schema(path: str) -> DatasetSchema:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    cols = [ColumnSchema(**c) for c in obj["columns"]]
    return DatasetSchema(columns=cols)

def validate_rows(schema: DatasetSchema, rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    expected = {c.name: c for c in schema.columns}

    for i, row in enumerate(rows):
        # required columns present?
        for cname, c in expected.items():
            if c.required and cname not in row:
                errors.append(f"row[{i}] missing required column '{cname}'")
        # unknown columns?
        for k in row.keys():
            if k not in expected:
                errors.append(f"row[{i}] has unexpected column '{k}'")

    return errors
