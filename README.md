# AI Project Health Predictor (Delivery Risk)

A local-first, free project health predictor that converts real delivery signals into:
- **Health Score (0–100)** (higher = healthier)
- **Delivery Confidence (%)** (heuristic)
- **Top risk drivers** (explainable, deterministic)
- Optional **ML prediction** of on-time likelihood if you provide labeled history

This is designed to be credible for program / project leadership: it avoids “one magic number” by showing **why** the score moved and what to act on.

## Purpose

Executive stakeholders often want a single “on track” indicator. That is risky without context.

This tool provides:
- A consistent health signal for quick scanning
- A ranked set of **drivers** so the score is actionable
- Optional ML that learns from historical outcomes (interpretable, not a black box)
- Sensitivity-ready inputs (you can tune thresholds and weights transparently)

## What it measures

The deterministic health score is built from these dimensions:

- **Schedule slip vs plan** (forecast end vs planned end)
- **Progress gap vs plan** (planned % complete vs actual)
- **Scope / change pressure** (backlog churn + requirements changes)
- **Quality pressure** (escape rate + critical defect fraction)
- **Blockers / waiting time** (blocked days)
- **Dependency load** (count of external dependencies)
- **Team stability** (churn rate)
- **Unplanned work load** (ratio)

All are computed from a simple CSV snapshot format.

---

## Repository structure
```
project-health-predictor/
  README.md
  pyproject.toml
  data/
    sample_projects.csv
    sample_history.csv
  src/project_health/
    __init__.py
    schemas.py
    features.py
    scoring.py
    model.py
    explain.py
    cli.py
  ui/
    app.py
  tests/
    test_scoring.py
    test_model.py
  .gitignore
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dev dependencies

```bash
pip install -e ".[dev]"
```

## Run 
```bash
health --snapshots data/sample_projects.csv --out health_scored.csv
```

Outputs:
- health_score_0to100
- delivery_confidence_0to1
- per-driver contributions (driver_*)
- explanation with top drivers

## Run with historical data - ML prediction
If you provide historical labeled data with on_time_0to1, the tool will:
- train a logistic regression model on historical features
- predict p_on_time for current snapshots

```bash
health --snapshots data/sample_projects.csv \
       --history data/sample_history.csv \
       --out health_scored_with_ml.csv
```

## Running the UI (Streamlit dashboard)

This repo includes a lightweight local UI to visualize the scored output CSV.

### 1) Generate a scored output file

From the repo root:

```bash
health --snapshots data/sample_projects.csv --out health_scored.csv
```

Or ML prediction based upon labelled historical data
```bash
health --snapshots data/sample_projects.csv \
       --history data/sample_history.csv \
       --out health_scored_with_ml.csv
```

### 2) Install UI Dependencies
```bash
pip install -e ".[ui]"
```
### 3) Launch the dashboard
```bash
streamlit run ui/app.py
```
In the sidebar, set “Path to scored CSV” to:
- health_scored.csv (deterministic only), or
- health_scored_with_ml.csv (includes p_on_time)



## Running the (Streamlit dashboard) UI

## Health score calculation
1. How the deterministic Health Score is calculated (high level)
1. Convert raw snapshot fields into features (e.g., schedule slip days, progress gap).
1. Convert each feature into a badness score in [0, 1] (0 good → 1 bad) using simple thresholds.
1. Compute a weighted sum of badness.
1. Convert to a health score:

```bash
    HealthScore =( 1 − WeightedBadness) × 100
```

## Test
```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
- test_scoring.py ensures health score and confidence stay within bounds.
- test_model.py ensures the ML model can train and produces probabilities in [0,1].
```
