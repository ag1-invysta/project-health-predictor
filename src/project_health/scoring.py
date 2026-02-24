from __future__ import annotations
import pandas as pd
import numpy as np

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def norm01(value: float, good_lo: float, good_hi: float, bad_hi: float, higher_is_worse: bool = True) -> float:
    """
    Piecewise linear normalization to [0,1] representing "badness".
    0 = good, 1 = bad.
    - good range: [good_lo, good_hi]
    - beyond bad_hi: fully bad (1.0)
    """
    v = value
    if higher_is_worse:
        if v <= good_hi:
            return 0.0
        if v >= bad_hi:
            return 1.0
        return (v - good_hi) / (bad_hi - good_hi)
    else:
        # if higher is better, invert logic
        if v >= good_hi:
            return 0.0
        if v <= bad_hi:
            return 1.0
        return (good_hi - v) / (good_hi - bad_hi)

def compute_health(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic health scoring. Outputs:
    - health_score_0to100 (higher is better)
    - delivery_confidence_0to1 (heuristic)
    - driver_* columns with per-factor contributions
    """
    out = df_feat.copy()

    # Badness scores (0 good → 1 bad). Tunable thresholds.
    out["bad_schedule_slip"] = out["schedule_slip_days"].apply(lambda d: norm01(d, good_lo=0, good_hi=7, bad_hi=45, higher_is_worse=True))
    out["bad_progress_gap"] = out["progress_gap"].apply(lambda g: norm01(g, good_lo=0, good_hi=0.05, bad_hi=0.25, higher_is_worse=True))
    out["bad_scope_pressure"] = out["scope_change_pressure"].apply(lambda s: norm01(s, good_lo=0, good_hi=5, bad_hi=30, higher_is_worse=True))
    out["bad_quality"] = (out["defect_escape_rate_last_4w"].apply(lambda r: norm01(r, 0, 0.02, 0.15, True)) * 0.6
                          + out["critical_defect_rate"].apply(lambda r: norm01(r, 0, 0.05, 0.30, True)) * 0.4)
    out["bad_blockers"] = out["blocked_days_last_2w"].apply(lambda b: norm01(b, 0, 1, 8, True))
    out["bad_dependencies"] = out["dependency_count"].apply(lambda c: norm01(c, 0, 2, 12, True))
    out["bad_team_churn"] = out["team_churn_rate"].apply(lambda r: norm01(r, 0, 0.05, 0.25, True))
    out["bad_unplanned"] = out["unplanned_work_ratio_last_4w"].apply(lambda r: norm01(r, 0, 0.15, 0.60, True))

    # Weights (sum to 1.0). These are policy preferences.
    weights = {
        "bad_schedule_slip": 0.22,
        "bad_progress_gap": 0.18,
        "bad_scope_pressure": 0.16,
        "bad_quality": 0.14,
        "bad_blockers": 0.10,
        "bad_dependencies": 0.08,
        "bad_team_churn": 0.07,
        "bad_unplanned": 0.05,
    }

    # Weighted badness
    out["weighted_badness"] = 0.0
    for k, w in weights.items():
        out["weighted_badness"] += out[k] * w
        out[f"driver_{k}"] = out[k] * w  # contribution

    # Health score (0..100), invert badness
    out["health_score_0to100"] = (1.0 - out["weighted_badness"]).clip(0, 1) * 100.0

    # Delivery confidence heuristic (0..1)
    # penalize more when slip/progress are poor, and when close to planned end date
    time_pressure = (1.0 - (out["days_to_planned_end"].clip(lower=0) / 180.0)).clip(0, 1)  # 0 far away → 1 close
    out["delivery_confidence_0to1"] = (1.0
                                       - 0.45 * out["bad_schedule_slip"]
                                       - 0.35 * out["bad_progress_gap"]
                                       - 0.20 * time_pressure).clip(0, 1)

    return out
