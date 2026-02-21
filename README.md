# agentic-trainer

Agentic AutoML-lite trainer for tabular **CSV/XLSX**:

- Builds an explicit **plan** and logs decisions
- Infers **problem type** (classification / regression)
- Chooses split strategy:
  - **time split** if time-like column detected
  - otherwise **stratified/random**
- Automatically drops **ID/index-like columns** (reduces OHE blow-up)
- Trains a small **model portfolio**, selects best, exports a full bundle
- Produces a clean **HTML report** including **Top Features**
- Optional: serves the trained model via **FastAPI**

## What you get (run bundle)

After training, the output folder contains:

- `model.joblib` — trained sklearn Pipeline
- `schema.json` — inference input contract (columns/types/examples)
- `model_metadata.json` — run metadata + scores + decisions + top features
- `run_manifest.json` — reproducibility manifest (data sha256 + env)
- `report.html` and `report.txt` — human-readable reports

## Prerequisites (Windows)

Use **Python 3.12 64-bit** (recommended).  
Older Python / 32-bit Python may trigger source builds for pandas.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

## Train (local)

```powershell
agentic-train --data .\data\mydata.csv --out .\out\run1 --target <TARGET_COLUMN>
```

If you want auto-target selection + prompts when ambiguous:

```powershell
agentic-train --data .\data\mydata.csv --out .\out\run1 --interactive
```

Open the report:

```powershell
start .\out\run1\report.html
```

## Demo datasets (Kaggle)

### 1) Telco churn (classification)

```powershell
mkdir .\data\telco -Force
kaggle datasets download -d blastchar/telco-customer-churn -p .\data\telco
Expand-Archive .\data\telco\telco-customer-churn.zip -DestinationPath .\data\telco -Force
agentic-train --data .\data\telco\WA_Fn-UseC_-Telco-Customer-Churn.csv --out .\out\telco --target Churn
start .\out\telco\report.html
```

### 2) Diamonds (regression)

```powershell
mkdir .\data\diamonds -Force
kaggle datasets download -d shivam2503/diamonds -p .\data\diamonds
Expand-Archive .\data\diamonds\diamonds.zip -DestinationPath .\data\diamonds -Force
agentic-train --data .\data\diamonds\diamonds.csv --out .\out\diamonds --target price
start .\out\diamonds\report.html
```

### 3) Iris (multiclass classification)

```powershell
mkdir .\data\iris -Force
kaggle datasets download -d uciml/iris -p .\data\iris
Expand-Archive .\data\iris\iris.zip -DestinationPath .\data\iris -Force
agentic-train --data .\data\iris\Iris.csv --out .\out\iris --target Species
start .\out\iris\report.html
```

## Serve (FastAPI)

```powershell
agentic-train --serve --model .\out\run1\model.joblib --schema .\out\run1\schema.json
```

Then open:
- http://127.0.0.1:8000/docs

Endpoints:
- `GET /health`
- `GET /schema`
- `POST /predict` with payload: `{"rows":[{...},{...}]}`

## Notes
- “Top Features” method:
  - `logreg_coef` for LogisticRegression
  - `permutation_importance` for tree models where applicable
