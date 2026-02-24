import pandas as pd
from project_health.features import compute_features
from project_health.model import train_on_time_model, predict_on_time

def test_model_trains_and_predicts():
    hist = pd.DataFrame([
        {"project_id":"H1","project_name":"H1","week_ending":"2026-01-01","planned_end_date":"2026-03-01","forecast_end_date":"2026-03-01",
         "planned_percent_complete":0.5,"actual_percent_complete":0.5,
         "backlog_items_added_last_4w":5,"backlog_items_closed_last_4w":8,"requirements_changed_last_4w":1,
         "defects_open":10,"defects_open_critical":0,"defect_escape_rate_last_4w":0.01,
         "blocked_days_last_2w":0,"dependency_count":1,
         "team_size":6,"team_churn_last_4w":0,"unplanned_work_ratio_last_4w":0.10,
         "on_time_0to1":1},
        {"project_id":"H2","project_name":"H2","week_ending":"2026-01-01","planned_end_date":"2026-03-01","forecast_end_date":"2026-04-15",
         "planned_percent_complete":0.5,"actual_percent_complete":0.35,
         "backlog_items_added_last_4w":30,"backlog_items_closed_last_4w":10,"requirements_changed_last_4w":10,
         "defects_open":60,"defects_open_critical":8,"defect_escape_rate_last_4w":0.10,
         "blocked_days_last_2w":6,"dependency_count":10,
         "team_size":6,"team_churn_last_4w":2,"unplanned_work_ratio_last_4w":0.55,
         "on_time_0to1":0},
    ])
    hist_feat = compute_features(hist)
    model = train_on_time_model(hist_feat)

    snap = hist.drop(columns=["on_time_0to1"]).copy()
    snap_feat = compute_features(snap)
    p = predict_on_time(model, snap_feat)
    assert (p.between(0, 1)).all()
