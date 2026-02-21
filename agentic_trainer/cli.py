from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np

import joblib
import pandas as pd
import uvicorn
from sklearn.inspection import permutation_importance

from .agent import (
    now_utc,
    normalize_columns,
    detect_timestamp_columns,
    infer_problem_type,
    pick_target,
    plan_run,
    split_data,
    train_iterative,
    make_manifest,
    file_sha256,
)
from .schema import infer_schema, save_schema
from .report import write_text_report, write_html_report
from .serve import create_app

def read_table(path: str, sheet: str | None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet if sheet else 0, engine="openpyxl")
    raise ValueError(f"Unsupported extension: {ext}")

def compute_feature_importance(best_pipe, X_val, y_val, problem_type: str, top_k: int = 15):
    """
    Returns dict:
      {
        "method": "...",
        "top_features": [{"name":..., "score":..., "sign":...}, ...]
      }
    """
    out = {"method": "none", "top_features": []}

    # Pipeline steps
    pre = best_pipe.named_steps.get("preprocess")
    model = best_pipe.named_steps.get("model")

    # 1) LogisticRegression coefficients mapped to OHE feature names
    if getattr(model, "__class__", None).__name__ == "LogisticRegression":
        try:
            # feature names from ColumnTransformer output
            feature_names = None
            if hasattr(pre, "get_feature_names_out"):
                feature_names = pre.get_feature_names_out()
            if feature_names is None:
                return out

            coef = model.coef_
            # Binary: shape (1, n_features). Multiclass: (n_classes, n_features)
            if coef.ndim == 2 and coef.shape[0] > 1:
                # Use mean abs across classes
                weights = np.mean(np.abs(coef), axis=0)
                signs = np.sign(np.mean(coef, axis=0))
            else:
                weights = np.abs(coef[0])
                signs = np.sign(coef[0])

            idx = np.argsort(weights)[::-1][:top_k]
            out["method"] = "logreg_coef"
            out["top_features"] = [
                {"name": str(feature_names[i]), "score": float(weights[i]), "sign": float(signs[i])}
                for i in idx
            ]
            return out
        except Exception:
            return out

    # 2) Permutation importance for tree models (or anything with predict)
    try:
        # Permutation importance expects raw X; pipeline will handle preprocessing
        scoring = "f1_macro" if problem_type == "classification" else "neg_root_mean_squared_error"
        r = permutation_importance(best_pipe, X_val, y_val, n_repeats=5, random_state=42, scoring=scoring)

        # We need names of the *original* columns here (not OHE-expanded)
        # So use X_val columns for display.
        importances = r.importances_mean
        cols = list(X_val.columns)
        idx = np.argsort(importances)[::-1][:top_k]

        out["method"] = "permutation_importance"
        out["top_features"] = [{"name": cols[i], "score": float(importances[i])} for i in idx]
        return out
    except Exception:
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to CSV/XLSX for training")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--interactive", action="store_true")

    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-seconds", type=int, default=180)
    ap.add_argument("--max-models", type=int, default=8)

    ap.add_argument("--serve", action="store_true", help="Run inference server instead of training")
    ap.add_argument("--model", default=None, help="Path to model.joblib for serve mode")
    ap.add_argument("--schema", default=None, help="Path to schema.json for serve mode")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)

    args = ap.parse_args()

    if args.serve:
        if not args.model or not args.schema:
            print("Serve mode requires --model and --schema", file=sys.stderr)
            sys.exit(2)
        app = create_app(args.model, args.schema)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if not args.data:
        print("Training mode requires --data", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)

    df = normalize_columns(read_table(args.data, args.sheet))
    if df.empty:
        print("ERROR: empty dataset", file=sys.stderr)
        sys.exit(2)

    decisions = []
    target, conf, _, notes = pick_target(df, args.target, args.interactive)
    decisions.extend(notes)

    ts_cols = detect_timestamp_columns(df.drop(columns=[target]))
    problem_type = infer_problem_type(df[target])

    plan = plan_run(df, target, problem_type, ts_cols)
    if plan.warnings:
        decisions.append(f"Plan warnings count: {len(plan.warnings)}")

    Xtr, Xva, ytr, yva, split_info = split_data(df, target, problem_type, ts_cols, args.test_size, args.seed)
    decisions.append(f"Split strategy: {split_info}")

    best_pipe, results, best, extra = train_iterative(
        Xtr, ytr, Xva, yva,
        problem_type=problem_type,
        seed=args.seed,
        budget_seconds=args.budget_seconds,
        max_models=args.max_models
    )
    feature_importance = compute_feature_importance(best_pipe, Xva, yva, problem_type=problem_type, top_k=15)

    decisions.extend(extra.get("reflection_notes", [])[-4:])

    model_path = os.path.join(args.out, "model.joblib")
    meta_path = os.path.join(args.out, "model_metadata.json")
    schema_path = os.path.join(args.out, "schema.json")
    report_txt = os.path.join(args.out, "report.txt")
    report_html = os.path.join(args.out, "report.html")
    manifest_path = os.path.join(args.out, "run_manifest.json")

    joblib.dump(best_pipe, model_path)

    feature_cols = list(df.drop(columns=[target]).columns)
    schema = infer_schema(df, feature_cols)
    save_schema(schema, schema_path)

    meta = {
        "created_at": now_utc(),
        "target": target,
        "target_confidence": float(conf),
        "problem_type": problem_type,
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "split_info": split_info,
        "feature_columns": feature_cols,
        "numeric_features": extra["numeric_features"],
        "categorical_features": extra["categorical_features"],
        "candidates": [json.loads(json.dumps(r, default=lambda o: o.__dict__)) for r in results],
        "best_model": json.loads(json.dumps(best, default=lambda o: o.__dict__)),
        "agent_decisions": decisions,
        "agent_plan": dataclasses_to_dict(plan),
        "feature_importance": feature_importance,
        "notes": "Serialized sklearn Pipeline via joblib. Use schema.json to validate inference input.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    write_text_report(report_txt, plan, decisions, best, results)
    write_html_report(report_html, meta)

    manifest = make_manifest(args.data, {
        "target": args.target,
        "interactive": bool(args.interactive),
        "test_size": args.test_size,
        "seed": args.seed,
        "budget_seconds": args.budget_seconds,
        "max_models": args.max_models,
    })
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Agentic Trainer v0.1.0 Complete ===")
    print(f"Model:    {model_path}")
    print(f"Schema:   {schema_path}")
    print(f"Metadata: {meta_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Report:   {report_html}")
    print("=====================================\n")

def dataclasses_to_dict(obj):
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return obj
