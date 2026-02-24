from __future__ import annotations
import pandas as pd

DRIVER_LABELS = {
    "driver_bad_schedule_slip": "Schedule slip vs plan",
    "driver_bad_progress_gap": "Progress gap vs plan",
    "driver_bad_scope_pressure": "Scope / change pressure",
    "driver_bad_quality": "Quality / defect pressure",
    "driver_bad_blockers": "Blockers / waiting time",
    "driver_bad_dependencies": "External dependencies",
    "driver_bad_team_churn": "Team churn / stability",
    "driver_bad_unplanned": "Unplanned work load",
}

def top_drivers(row: pd.Series, n: int = 3) -> list[tuple[str, float]]:
    pairs = []
    for k, label in DRIVER_LABELS.items():
        if k in row:
            pairs.append((label, float(row[k])))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]

def add_explanations(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    out = df.copy()
    expl = []
    for _, r in out.iterrows():
        drivers = top_drivers(r, n=n)
        if drivers and drivers[0][1] > 0:
            parts = [f"{name} ({val:.2f})" for name, val in drivers]
            expl.append("Top drivers: " + "; ".join(parts))
        else:
            expl.append("Top drivers: none (healthy across tracked dimensions).")
    out["explanation"] = expl
    return out
