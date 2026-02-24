from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Model predicts probability of on-time (or green) based on derived features.
FEATURES = [
    "schedule_slip_days",
    "progress_gap",
    "scope_change_pressure",
    "defect_escape_rate_last_4w",
    "critical_defect_rate",
    "blocked_days_last_2w",
    "dependency_count",
    "team_churn_rate",
    "unplanned_work_ratio_last_4w",
]

LABEL = "on_time_0to1"

def train_on_time_model(history_feat: pd.DataFrame) -> Pipeline:
    missing = [c for c in FEATURES + [LABEL] if c not in history_feat.columns]
    if missing:
        raise ValueError(f"History missing columns: {missing}")

    X = history_feat[FEATURES].copy()
    y = history_feat[LABEL].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X, y)
    return model

def predict_on_time(model: Pipeline, feat_df: pd.DataFrame) -> pd.Series:
    X = feat_df[FEATURES].copy()
    p = model.predict_proba(X)[:, 1]
    return pd.Series(p, index=feat_df.index, name="p_on_time")
