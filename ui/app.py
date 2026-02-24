from __future__ import annotations

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Project Health Predictor", layout="wide")

DEFAULT_FILE = "health_scored.csv"

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)

    required = {"project_id", "project_name", "health_score_0to100", "delivery_confidence_0to1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in output CSV: {sorted(missing)}")

    # Parse dates if present (needed for trends)
    if "week_ending" in df.columns:
        df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")

    df["delivery_confidence_pct"] = (df["delivery_confidence_0to1"] * 100.0).round(1)
    if "p_on_time" in df.columns:
        df["p_on_time_pct"] = (df["p_on_time"] * 100.0).round(1)

    return df


def driver_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("driver_")]


def plot_hist(series: pd.Series, title: str, xlabel: str):
    fig = plt.figure()
    plt.hist(series.dropna(), bins=12)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    st.pyplot(fig)


def plot_scatter(df: pd.DataFrame):
    fig = plt.figure()
    x = df["health_score_0to100"]
    y = df["delivery_confidence_pct"]
    plt.scatter(x, y)
    plt.xlabel("Health Score (0–100)")
    plt.ylabel("Delivery Confidence (%)")
    plt.title("Health vs Delivery Confidence")
    st.pyplot(fig)


def plot_driver_stack(df: pd.DataFrame, drivers: list[str], top_n: int = 10):
    view = df.sort_values("health_score_0to100", ascending=True).head(top_n).copy()
    view = view[["project_name", "health_score_0to100"] + drivers]

    fig = plt.figure(figsize=(10, 5))
    bottom = None
    labels = view["project_name"].tolist()

    for d in drivers:
        vals = view[d].fillna(0).tolist()
        if bottom is None:
            plt.bar(labels, vals)
            bottom = vals
        else:
            plt.bar(labels, vals, bottom=bottom)
            bottom = [b + v for b, v in zip(bottom, vals)]

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Weighted Contribution to Badness")
    plt.title(f"Top {top_n} At-Risk Projects: Driver Contributions (Stacked)")
    st.pyplot(fig)


def latest_snapshot_table(df: pd.DataFrame) -> pd.DataFrame:
    if "week_ending_dt" not in df.columns or df["week_ending_dt"].isna().all():
        # No usable time axis; treat current file as latest-only already
        return df.copy()

    idx = df.sort_values("week_ending_dt").groupby("project_id")["week_ending_dt"].idxmax()
    return df.loc[idx].copy()


def compute_latest_delta(project_df: pd.DataFrame) -> dict:
    """
    Assumes project_df contains multiple rows for one project.
    Returns deltas latest - previous for selected metrics if possible.
    """
    if "week_ending_dt" not in project_df.columns or project_df["week_ending_dt"].isna().all():
        return {}

    ordered = project_df.sort_values("week_ending_dt").dropna(subset=["week_ending_dt"])
    if len(ordered) < 2:
        return {}

    latest = ordered.iloc[-1]
    prev = ordered.iloc[-2]

    out = {
        "delta_health": float(latest["health_score_0to100"]) - float(prev["health_score_0to100"]),
        "delta_conf_pct": float(latest["delivery_confidence_pct"]) - float(prev["delivery_confidence_pct"]),
    }
    if "p_on_time_pct" in ordered.columns:
        out["delta_p_on_time_pct"] = float(latest["p_on_time_pct"]) - float(prev["p_on_time_pct"])
    return out


def plot_trend(project_df: pd.DataFrame):
    if "week_ending_dt" not in project_df.columns or project_df["week_ending_dt"].isna().all():
        st.info("Trend view requires a valid `week_ending` column in the scored CSV.")
        return

    ordered = project_df.sort_values("week_ending_dt").dropna(subset=["week_ending_dt"])
    if len(ordered) < 2:
        st.info("Not enough snapshots for a trend. Provide multiple weeks per project.")
        return

    # Health trend
    fig1 = plt.figure()
    plt.plot(ordered["week_ending_dt"], ordered["health_score_0to100"])
    plt.title("Health Score Trend")
    plt.xlabel("Week Ending")
    plt.ylabel("Health Score (0–100)")
    st.pyplot(fig1)

    # Confidence trend
    fig2 = plt.figure()
    plt.plot(ordered["week_ending_dt"], ordered["delivery_confidence_pct"])
    plt.title("Delivery Confidence Trend")
    plt.xlabel("Week Ending")
    plt.ylabel("Delivery Confidence (%)")
    st.pyplot(fig2)

    # Optional ML trend
    if "p_on_time_pct" in ordered.columns:
        fig3 = plt.figure()
        plt.plot(ordered["week_ending_dt"], ordered["p_on_time_pct"])
        plt.title("Predicted On-Time Probability Trend (ML)")
        plt.xlabel("Week Ending")
        plt.ylabel("p(on-time) (%)")
        st.pyplot(fig3)


st.title("AI Project Health Predictor — Dashboard")

with st.sidebar:
    st.header("Data")
    path = st.text_input("Path to scored CSV", value=DEFAULT_FILE)
    st.caption("Tip: run the CLI first to generate health_scored.csv")
    show_latest_only = st.checkbox("Show latest snapshot per project (recommended)", value=True)

try:
    df = load_data(path)
except Exception as e:
    st.error(str(e))
    st.stop()

drivers = driver_columns(df)

# Choose whether to work with latest-only view or full time series
df_latest = latest_snapshot_table(df)
view_df = df_latest if show_latest_only else df.copy()

with st.sidebar:
    st.header("Filters")
    min_health, max_health = st.slider("Health Score Range", 0, 100, (0, 100))
    order_at_risk = st.checkbox("Order by at-risk first (lowest health)", value=True)

filtered = view_df[(view_df["health_score_0to100"] >= min_health) & (view_df["health_score_0to100"] <= max_health)].copy()
filtered = filtered.sort_values("health_score_0to100", ascending=True if order_at_risk else False)

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Projects", len(filtered))
c2.metric("Avg Health", f"{filtered['health_score_0to100'].mean():.1f}")
c3.metric("Avg Confidence", f"{filtered['delivery_confidence_pct'].mean():.1f}%")
if "p_on_time_pct" in filtered.columns:
    c4.metric("Avg p(on-time)", f"{filtered['p_on_time_pct'].mean():.1f}%")
else:
    c4.metric("ML p(on-time)", "not available")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Distribution")
    plot_hist(filtered["health_score_0to100"], "Health Score Distribution", "Health Score (0–100)")
    plot_hist(filtered["delivery_confidence_pct"], "Delivery Confidence Distribution", "Delivery Confidence (%)")

with right:
    st.subheader("Relationship")
    plot_scatter(filtered)

st.divider()

st.subheader("At-a-glance table")

display_cols = ["project_id", "project_name", "health_score_0to100", "delivery_confidence_pct"]
if "p_on_time_pct" in filtered.columns:
    display_cols.append("p_on_time_pct")
if "week_ending" in filtered.columns:
    display_cols.append("week_ending")
if "explanation" in filtered.columns:
    display_cols.append("explanation")

st.dataframe(filtered[display_cols], use_container_width=True)

if drivers:
    st.divider()
    st.subheader("Top drivers (portfolio view)")
    top_n = st.slider("Show top N at-risk projects", min_value=3, max_value=20, value=10)
    plot_driver_stack(filtered, drivers, top_n=top_n)

st.divider()
st.subheader("Project detail")

project_names = filtered["project_name"].tolist()
if not project_names:
    st.info("No projects match the current filters.")
    st.stop()

selected = st.selectbox("Select a project", project_names)
project_all_rows = df[df["project_name"] == selected].copy()
project_latest_row = latest_snapshot_table(project_all_rows).iloc[0] if len(project_all_rows) > 0 else None

# Summary metrics for selected project (latest)
if project_latest_row is not None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Health (latest)", f"{float(project_latest_row['health_score_0to100']):.1f}")
    m2.metric("Confidence (latest)", f"{float(project_latest_row['delivery_confidence_pct']):.1f}%")
    if "p_on_time_pct" in project_latest_row.index:
        m3.metric("p(on-time) (latest)", f"{float(project_latest_row['p_on_time_pct']):.1f}%")
    else:
        m3.metric("p(on-time) (latest)", "n/a")

    deltas = compute_latest_delta(project_all_rows)
    if deltas:
        d1, d2, d3 = st.columns(3)
        d1.metric("Δ Health vs prior", f"{deltas['delta_health']:+.1f}")
        d2.metric("Δ Confidence vs prior", f"{deltas['delta_conf_pct']:+.1f}%")
        if "delta_p_on_time_pct" in deltas:
            d3.metric("Δ p(on-time) vs prior", f"{deltas['delta_p_on_time_pct']:+.1f}%")
        else:
            d3.metric("Δ p(on-time) vs prior", "n/a")

# Explanation + drivers (latest row)
if project_latest_row is not None:
    st.write("**Explanation (latest)**")
    st.write(project_latest_row.get("explanation", "No explanation column found."))

    if drivers:
        st.write("**Drivers (latest; weighted contributions)**")
        driver_df = pd.DataFrame({
            "driver": drivers,
            "contribution": [float(project_latest_row[d]) if pd.notna(project_latest_row[d]) else 0.0 for d in drivers],
        }).sort_values("contribution", ascending=False)
        st.dataframe(driver_df, use_container_width=True)

# Trend charts (if multiple snapshots exist)
st.divider()
st.subheader("Trend over time")
plot_trend(project_all_rows)
