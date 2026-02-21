from __future__ import annotations

import dataclasses
import hashlib
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)

def now_utc() -> str:
    return datetime.utcnow().isoformat() + "Z"


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def is_probably_id_like(series: pd.Series) -> bool:
    s = series.dropna()
    if len(s) < 50:
        return False
    nunique = s.nunique(dropna=True)
    return (nunique / max(len(s), 1)) > 0.98


def detect_timestamp_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for c in df.columns:
        # Only consider time-like column names to avoid noisy parsing warnings
        name = str(c).lower()
        if not any(k in name for k in ["date", "time", "timestamp", "datetime"]):
            continue

        if pd.api.types.is_datetime64_any_dtype(df[c]):
            out.append(c)
            continue

        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            sample = df[c].dropna().astype(str).head(200)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", utc=True)
            ok_ratio = float(parsed.notna().mean())
            if ok_ratio >= 0.85:
                out.append(c)

    return out

# def detect_timestamp_columns(df: pd.DataFrame) -> List[str]:
#     out: List[str] = []
#     for c in df.columns:
#         name = c.lower()
#         if not any(k in name for k in ["date", "time", "timestamp", "datetime", "ts"]):
#             continue
#         if pd.api.types.is_datetime64_any_dtype(df[c]):
#             out.append(c)
#             continue
#         if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
#             sample = df[c].dropna().astype(str).head(200)
#             if sample.empty:
#                 continue
#             parsed = pd.to_datetime(sample, errors="coerce", utc=True)
#             if float(parsed.notna().mean()) >= 0.85:
#                 out.append(c)
#     return out


def coerce_datetime_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def infer_problem_type(y: pd.Series) -> str:
    y_nonnull = y.dropna()
    if y_nonnull.empty:
        raise ValueError("Target has no non-null values.")
    if not pd.api.types.is_numeric_dtype(y_nonnull):
        return "classification"
    n = len(y_nonnull)
    nunique = y_nonnull.nunique()
    if pd.api.types.is_integer_dtype(y_nonnull) and nunique <= max(20, int(0.02 * n)):
        return "classification"
    if nunique <= 20:
        return "classification"
    return "regression"


def evaluate_classification(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse, "mae": float(mean_absolute_error(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}


@dataclass
class TargetCandidate:
    column: str
    score: float
    notes: List[str]


@dataclass
class Plan:
    steps: List[str]
    assumptions: List[str]
    warnings: List[str]


@dataclass
class CandidateResult:
    name: str
    problem_type: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    elapsed_sec: float


def score_target_candidates(df: pd.DataFrame) -> List[TargetCandidate]:
    cols = list(df.columns)
    n_rows = len(df)
    cands: List[TargetCandidate] = []
    for c in cols:
        s = df[c]
        if s.isna().all():
            continue
        notes: List[str] = []
        score = 0.0
        name = c.lower()
        if any(k in name for k in ["target", "label", "class", "outcome", "y"]):
            score += 2.0
            notes.append("name suggests label")
        if is_probably_id_like(s):
            score -= 5.0
            notes.append("looks like ID (high uniqueness)")
        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            score -= 10.0
            notes.append("constant")
        else:
            uniq_ratio = nunique / max(n_rows, 1)
            if uniq_ratio < 0.05:
                score += 1.2
                notes.append("low cardinality")
            elif uniq_ratio < 0.20:
                score += 0.6
                notes.append("moderate cardinality")
            else:
                score -= 0.7
                notes.append("high cardinality")
        if c == cols[-1]:
            score += 0.4
            notes.append("last-column bias")
        cands.append(TargetCandidate(c, score, notes))
    return sorted(cands, key=lambda x: x.score, reverse=True)


def sigmoid_confidence(top_score: float, second_score: Optional[float]) -> float:
    margin = top_score - (second_score if second_score is not None else (top_score - 1.0))
    return float(1.0 / (1.0 + np.exp(-1.2 * margin)))


def pick_target(df: pd.DataFrame, explicit_target: Optional[str], interactive: bool) -> Tuple[str, float, List[TargetCandidate], List[str]]:
    if explicit_target:
        if explicit_target not in df.columns:
            raise ValueError(f"Target '{explicit_target}' not found in columns.")
        return explicit_target, 1.0, [], [f"Target explicitly provided: {explicit_target}"]

    cands = score_target_candidates(df)
    if not cands:
        raise ValueError("No viable target candidates found.")
    top = cands[0]
    conf = sigmoid_confidence(top.score, cands[1].score if len(cands) > 1 else None)
    notes = [f"Auto-picked target '{top.column}' (score={top.score:.2f}, confidence={conf:.2f})"]

    if interactive and conf < 0.65:
        show = cands[: min(7, len(cands))]
        print("\nTarget selection ambiguous. Pick target column:\n")
        for i, c in enumerate(show, 1):
            print(f"  [{i}] {c.column}  score={c.score:.2f}  notes={', '.join(c.notes)}")
        print("")
        while True:
            choice = input(f"Enter 1-{len(show)} (or type column name): ").strip()
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(show):
                    picked = show[idx - 1].column
                    return picked, 0.95, cands, notes + [f"User selected target: {picked}"]
            else:
                if choice in df.columns:
                    return choice, 0.95, cands, notes + [f"User typed target: {choice}"]
            print("Invalid selection. Try again.\n")

    if conf < 0.65:
        notes.append("WARNING: Low-confidence target guess; use --target or --interactive.")
    return top.column, conf, cands, notes


def plan_run(df: pd.DataFrame, target: str, problem_type: str, ts_cols: List[str]) -> Plan:
    steps = [
        "Validate dataset + detect leakage risks",
        "Infer feature types + build preprocessing",
        "Choose split strategy (time vs stratified/random)",
        "Train portfolio and evaluate",
        "Reflect and expand portfolio if weak",
        "Export model + schema + report + manifest",
    ]
    assumptions = [f"Target column: {target}", f"Problem type: {problem_type}"]
    warnings: List[str] = []

    if ts_cols:
        assumptions.append(f"Timestamp-like columns detected: {ts_cols}")
    else:
        assumptions.append("No timestamp-like columns detected")

    for c in df.columns:
        if c != target and target.lower() in c.lower():
            warnings.append(f"Possible leakage: feature '{c}' includes target name '{target}'.")

    if len(df) < 200:
        warnings.append("Small dataset; metrics may be unstable.")

    return Plan(steps=steps, assumptions=assumptions, warnings=warnings)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in numeric]

    num_t = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())])
    cat_t = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        [("num", num_t, numeric), ("cat", cat_t, cat)],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre, numeric, cat

def will_be_sparse(categorical_features: List[str]) -> bool:
    # If there are categorical features, OneHotEncoder will yield sparse by default
    return len(categorical_features) > 0

def detect_and_drop_id_columns(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Drops ID/index-like columns and returns (X_clean, dropped_reasons).
    Conservative rules:
      - Drop known index column names (e.g. 'Unnamed: 0', 'index')
      - Drop columns with very high uniqueness ratio (>0.98) when name hints ID-like
    """
    dropped: Dict[str, str] = {}
    X2 = X.copy()

    def norm(name: str) -> str:
        return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")

    for c in list(X2.columns):
        cn = norm(c)
        s = X2[c]

        # 1) Common "index column" patterns from CSV exports
        if cn in {"unnamed:0", "unnamed0", "index"}:
            dropped[c] = "index-like column name"
            X2 = X2.drop(columns=[c])
            continue

        # 2) ID-like by name + near-unique values
        name_hint = any(k in cn for k in ["id", "customerid", "userid", "accountid", "guid", "uuid"])
        if name_hint and is_probably_id_like(s):
            dropped[c] = "id-like name + near-unique values"
            X2 = X2.drop(columns=[c])
            continue

    return X2, dropped

def split_data(df: pd.DataFrame, target: str, problem_type: str, ts_cols: List[str], test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    X = df.drop(columns=[target])
    y = df[target]

    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # WOW improvement: drop ID/index-like columns before splitting
    X, dropped_reasons = detect_and_drop_id_columns(X)

    # We'll include these in split_info so they surface in logs/reports
    drop_info = {
        "dropped_columns": list(dropped_reasons.keys()),
        "drop_reasons": dropped_reasons
    }

    if ts_cols:
        ts = ts_cols[0]
        df2 = pd.concat([X, y.rename("__target__")], axis=1)
        df2 = coerce_datetime_column(df2, ts).sort_values(by=ts).reset_index(drop=True)
        cutoff = int((1.0 - test_size) * len(df2))
        tr = df2.iloc[:cutoff]
        va = df2.iloc[cutoff:]
        return tr.drop(columns="__target__"), va.drop(columns="__target__"), tr["__target__"], va["__target__"], {
            "strategy": "time",
            "timestamp_column": ts,
            "cutoff_index": cutoff,
            **drop_info,
        }

    stratify = None
    if problem_type == "classification" and y.nunique(dropna=True) <= max(50, int(0.1 * len(y))):
        stratify = y

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    return Xtr, Xva, ytr, yva, {
        "strategy": "stratified" if stratify is not None else "random",
        **drop_info,
    }

# def split_data(df: pd.DataFrame, target: str, problem_type: str, ts_cols: List[str], test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
#     X = df.drop(columns=[target])
#     y = df[target]
#
#     mask = y.notna()
#     X = X.loc[mask].reset_index(drop=True)
#     y = y.loc[mask].reset_index(drop=True)
#
#     if ts_cols:
#         ts = ts_cols[0]
#         df2 = pd.concat([X, y.rename("__target__")], axis=1)
#         df2 = coerce_datetime_column(df2, ts).sort_values(by=ts).reset_index(drop=True)
#         cutoff = int((1.0 - test_size) * len(df2))
#         tr = df2.iloc[:cutoff]
#         va = df2.iloc[cutoff:]
#         return tr.drop(columns="__target__"), va.drop(columns="__target__"), tr["__target__"], va["__target__"], {
#             "strategy": "time",
#             "timestamp_column": ts,
#             "cutoff_index": cutoff,
#         }
#
#     stratify = None
#     if problem_type == "classification" and y.nunique(dropna=True) <= max(50, int(0.1 * len(y))):
#         stratify = y
#
#     Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
#     return Xtr, Xva, ytr, yva, {"strategy": "stratified" if stratify is not None else "random"}


def make_portfolio(problem_type: str, seed: int, expanded: bool) -> List[Tuple[str, Any, Dict[str, Any]]]:
    if problem_type == "classification":
        base = [
            ("logreg", LogisticRegression(max_iter=2000), {"max_iter": 2000}),
            ("rf", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1), {"n_estimators": 400}),
            ("hgb", HistGradientBoostingClassifier(random_state=seed), {"model": "HGBClassifier"}),
        ]
        if expanded:
            base += [
                ("extratrees", ExtraTreesClassifier(n_estimators=700, random_state=seed, n_jobs=-1), {"n_estimators": 700}),
                ("sgd", SGDClassifier(loss="log_loss", max_iter=5000, random_state=seed), {"loss": "log_loss"}),
            ]
        return base
    else:
        base = [
            ("ridge", Ridge(random_state=seed), {"model": "Ridge"}),
            ("rf", RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1), {"n_estimators": 500}),
            ("hgb", HistGradientBoostingRegressor(random_state=seed), {"model": "HGBRegressor"}),
        ]
        if expanded:
            base += [
                ("extratrees", ExtraTreesRegressor(n_estimators=800, random_state=seed, n_jobs=-1), {"n_estimators": 800}),
                ("sgd", SGDRegressor(max_iter=5000, random_state=seed), {"model": "SGDRegressor"}),
            ]
        return base


def is_weak(problem_type: str, metrics: Dict[str, float]) -> bool:
    if problem_type == "classification":
        return (metrics.get("f1_macro", 0.0) < 0.65) and (metrics.get("accuracy", 0.0) < 0.70)
    return metrics.get("r2", -1.0) < 0.20


def pick_best(problem_type: str, results: List[CandidateResult]) -> CandidateResult:
    if problem_type == "classification":
        return sorted(results, key=lambda r: (r.metrics.get("f1_macro", -1.0), r.metrics.get("accuracy", -1.0)), reverse=True)[0]
    return sorted(results, key=lambda r: (r.metrics.get("rmse", float("inf")), -r.metrics.get("r2", -1.0)))[0]


def train_iterative(
    Xtr: pd.DataFrame, ytr: pd.Series,
    Xva: pd.DataFrame, yva: pd.Series,
    problem_type: str,
    seed: int,
    budget_seconds: int,
    max_models: int
) -> Tuple[Pipeline, List[CandidateResult], CandidateResult, Dict[str, Any]]:
    start = time.time()
    pre, num, cat = build_preprocessor(Xtr)
    sparse_expected = will_be_sparse(cat)
    fitted: Dict[str, Pipeline] = {}
    results: List[CandidateResult] = []
    notes: List[str] = []
    expanded = False
    rounds = 0

    while True:
        rounds += 1
        portfolio = make_portfolio(problem_type, seed, expanded)
        notes.append(f"Round {rounds}: trying {len(portfolio)} models (expanded={expanded}).")

        for name, model, params in portfolio:
            # HistGradientBoosting does not accept sparse matrices
            if sparse_expected and name == "hgb":
                notes.append("Skipping 'hgb' because OneHotEncoder produces sparse features and HGB requires dense.")
                continue
            if name in fitted:
                continue
            if len(results) >= max_models:
                notes.append("Model cap reached; stopping search.")
                break
            if (time.time() - start) > budget_seconds:
                notes.append("Time budget reached; stopping search.")
                break

            t0 = time.time()
            pipe = Pipeline([("preprocess", pre), ("model", model)])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xva)
            elapsed = time.time() - t0

            metrics = evaluate_classification(yva, pred) if problem_type == "classification" else evaluate_regression(yva, pred)
            cr = CandidateResult(name=name, problem_type=problem_type, metrics=metrics, params=params, elapsed_sec=float(elapsed))
            results.append(cr)
            fitted[name] = pipe

        best = pick_best(problem_type, results)
        notes.append(f"Best so far: {best.name} metrics={best.metrics}")

        if not expanded and is_weak(problem_type, best.metrics):
            expanded = True
            notes.append("Weak metrics detected → expanding portfolio and continuing.")
            continue

        best_pipe = fitted[best.name]
        return best_pipe, results, best, {"numeric_features": num, "categorical_features": cat, "reflection_notes": notes, "rounds": rounds}


def make_manifest(data_path: str, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": now_utc(),
        "data_path": os.path.abspath(data_path),
        "data_sha256": file_sha256(data_path),
        "args": args_dict,
        "platform": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "machine": platform.machine(),
        },
    }
