from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED_COLS = [
    "project_id","project_name","week_ending","planned_end_date","forecast_end_date",
    "planned_percent_complete","actual_percent_complete",
    "backlog_items_added_last_4w","backlog_items_closed_last_4w","requirements_changed_last_4w",
    "defects_open","defects_open_critical","defect_escape_rate_last_4w",
    "blocked_days_last_2w","dependency_count",
    "team_size","team_churn_last_4w","unplanned_work_ratio_last_4w",
]

def validate_snapshot_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["planned_percent_complete","actual_percent_complete","defect_escape_rate_last_4w","unplanned_work_ratio_last_4w"]:
        if (~df[col].between(0, 1)).any():
            raise ValueError(f"{col} must be in [0, 1]")

    if (df["team_size"] <= 0).any():
        raise ValueError("team_size must be > 0")

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw snapshot inputs into model-ready features.
    All derived features are deterministic and explainable.
    """
    validate_snapshot_df(df)
    out = df.copy()

    # Parse dates
    we = pd.to_datetime(out["week_ending"])
    pe = pd.to_datetime(out["planned_end_date"])
    fe = pd.to_datetime(out["forecast_end_date"])

    out["days_to_planned_end"] = (pe - we).dt.days.astype(float)
    out["days_to_forecast_end"] = (fe - we).dt.days.astype(float)
    out["schedule_slip_days"] = out["days_to_forecast_end"] - out["days_to_planned_end"]

    # progress variance (positive means behind)
    out["progress_gap"] = (out["planned_percent_complete"] - out["actual_percent_complete"]).astype(float)

    # scope churn: added vs closed + requirements changes
    out["backlog_churn"] = (out["backlog_items_added_last_4w"] - out["backlog_items_closed_last_4w"]).astype(float)
    out["scope_change_pressure"] = out["backlog_churn"] + out["requirements_changed_last_4w"].astype(float)

    # quality pressure
    out["critical_defect_rate"] = (out["defects_open_critical"] / np.maximum(out["defects_open"], 1)).astype(float)

    # team churn rate
    out["team_churn_rate"] = (out["team_churn_last_4w"] / out["team_size"]).astype(float)

    return out
