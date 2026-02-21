from __future__ import annotations

import html
from dataclasses import asdict
from typing import Any, Dict, List

from .agent import CandidateResult, Plan

def write_text_report(path: str, plan: Plan, decisions: List[str], best: CandidateResult, all_results: List[CandidateResult]) -> None:
    lines = []
    lines.append("AGENTIC TRAINING REPORT")
    lines.append("")
    lines.append("Plan:")
    for s in plan.steps:
        lines.append(f"  - {s}")
    lines.append("")
    lines.append("Assumptions:")
    for a in plan.assumptions:
        lines.append(f"  - {a}")
    lines.append("")
    if plan.warnings:
        lines.append("Warnings:")
        for w in plan.warnings:
            lines.append(f"  - {w}")
        lines.append("")
    lines.append("Decisions:")
    for d in decisions:
        lines.append(f"  - {d}")
    lines.append("")
    lines.append(f"Best model: {best.name}")
    lines.append(f"Metrics: {best.metrics}")
    lines.append("")
    lines.append("Leaderboard:")
    for r in all_results:
        lines.append(f"  - {r.name:12s} metrics={r.metrics} elapsed={r.elapsed_sec:.2f}s")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_html_report(path: str, meta: Dict[str, Any]) -> None:
    # Lightweight, dependency-free HTML.
    def esc(x: Any) -> str:
        return html.escape(str(x))

    best = meta["best_model"]
    candidates = meta["candidates"]
    plan = meta["agent_plan"]
    decisions = meta["agent_decisions"]
    warnings = plan.get("warnings", [])
    assumptions = plan.get("assumptions", [])
    fi = meta.get("feature_importance", {"method": "none", "top_features": []})
    fi_rows = ""
    if fi.get("top_features"):
        for item in fi["top_features"]:
            sign = item.get("sign", None)
            sign_txt = ""
            if sign is not None:
                sign_txt = "+" if float(sign) > 0 else "-"
            fi_rows += f"<tr><td>{esc(item.get('name'))}</td><td>{esc(sign_txt)}</td><td>{esc(item.get('score'))}</td></tr>\n"
    else:
        fi_rows = "<tr><td colspan='3'>No feature importance available</td></tr>"

    rows = ""
    for c in sorted(candidates, key=lambda x: x["metrics"].get("f1_macro", 0.0) if meta["problem_type"] == "classification"
                    else -x["metrics"].get("rmse", 1e18), reverse=True):
        rows += f"<tr><td>{esc(c['name'])}</td><td>{esc(c['metrics'])}</td><td>{esc(c['elapsed_sec'])}</td></tr>\n"

    decisions_li = "\n".join([f"<li>{esc(d)}</li>" for d in decisions])
    warnings_li = "\n".join([f"<li>{esc(w)}</li>" for w in warnings]) if warnings else "<li>None</li>"
    assumptions_li = "\n".join([f"<li>{esc(a)}</li>" for a in assumptions]) if assumptions else "<li>None</li>"

    html_doc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Agentic Trainer Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #eee; text-align: left; padding: 8px; }}
    th {{ background: #fafafa; }}
    code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Agentic Trainer Report</h1>
  <p><b>Created:</b> {esc(meta.get("created_at"))}</p>

  <div class="card">
    <h2>Summary</h2>
    <p><b>Target:</b> <code>{esc(meta.get("target"))}</code> (confidence {esc(meta.get("target_confidence"))})</p>
    <p><b>Problem:</b> <code>{esc(meta.get("problem_type"))}</code></p>
    <p><b>Split:</b> {esc(meta.get("split_info"))}</p>
    <p><b>Best model:</b> <code>{esc(best.get("name"))}</code></p>
    <p><b>Best metrics:</b> {esc(best.get("metrics"))}</p>
  </div>

  <div class="card">
    <h2>Leaderboard</h2>
    <table>
      <thead><tr><th>Model</th><th>Metrics</th><th>Elapsed (s)</th></tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Assumptions</h2>
    <ul>{assumptions_li}</ul>
  </div>

  <div class="card">
    <h2>Warnings</h2>
    <ul>{warnings_li}</ul>
  </div>

  <div class="card">
    <h2>Top Features</h2>
    <p><b>Method:</b> <code>{esc(fi.get("method"))}</code></p>
    <table>
      <thead><tr><th>Feature</th><th>Sign</th><th>Score</th></tr></thead>
      <tbody>
        {fi_rows}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Agent decisions</h2>
    <ul>{decisions_li}</ul>
  </div>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc.strip() + "\n")
