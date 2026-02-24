from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ProjectSnapshot:
    project_id: str
    project_name: str
    week_ending: str  # YYYY-MM-DD
    planned_end_date: str  # YYYY-MM-DD
    forecast_end_date: str  # YYYY-MM-DD

    # schedule / progress
    planned_percent_complete: float  # 0..1
    actual_percent_complete: float   # 0..1

    # scope / change
    backlog_items_added_last_4w: int
    backlog_items_closed_last_4w: int
    requirements_changed_last_4w: int

    # quality
    defects_open: int
    defects_open_critical: int
    defect_escape_rate_last_4w: float  # 0..1

    # delivery flow
    blocked_days_last_2w: int
    dependency_count: int

    # team stability / resourcing
    team_size: int
    team_churn_last_4w: int  # people leaving or rotating out
    unplanned_work_ratio_last_4w: float  # 0..1
