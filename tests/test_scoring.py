import pandas as pd
from project_health.features import compute_features
from project_health.scoring import compute_health

def test_health_score_bounds():
    df = pd.DataFrame([{
        "project_id":"P1","project_name":"A","week_ending":"2026-02-07",
        "planned_end_date":"2026-06-01","forecast_end_date":"2026-06-15",
        "planned_percent_complete":0.40,"actual_percent_complete":0.32,
        "backlog_items_added_last_4w":20,"backlog_items_closed_last_4w":10,"requirements_changed_last_4w":6,
        "defects_open":40,"defects_open_critical":5,"defect_escape_rate_last_4w":0.06,
        "blocked_days_last_2w":3,"dependency_count":6,
        "team_size":8,"team_churn_last_4w":1,"unplanned_work_ratio_last_4w":0.25
    }])
    feat = compute_features(df)
    scored = compute_health(feat)
    s = float(scored.loc[0, "health_score_0to100"])
    c = float(scored.loc[0, "delivery_confidence_0to1"])
    assert 0.0 <= s <= 100.0
    assert 0.0 <= c <= 1.0
