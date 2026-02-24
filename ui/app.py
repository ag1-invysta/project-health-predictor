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

    # Basic sanity checks
    required = {"project_id", "project_name", "health_score_0to100", "delivery_confidence_0to1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in output CSV: {sorted(missing)}")

    # Convert confidence to %
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
    # Show top N lowest health projects by default (highest risk)
    view = df.sort_values("health_score_0to100", ascending=True).head(top_n).copy()
    view = view[["project_name", "health_score_0to100"] + drivers]

    # Stacked bar: driver contributions (already weighted) per project
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


st.title("AI Project Health Predictor — Dashboard")

with st.sidebar:
    st.header("Data")
    path = st.text_input("Path to scored CSV", value=DEFAULT_FILE)
    st.caption("Tip: run the CLI first to generate health_scored.csv")

try:
    df = load_data(path)
except Exception as e:
    st.error(str(e))
    st.stop()

drivers = driver_columns(df)

# Filters
with st.sidebar:
    st.header("Filters")
    min_health, max_health = st.slider("Health Score Range", 0, 100, (0, 100))
    show_mandatory = st.checkbox("Show only lowest health (at-risk) first", value=True)

filtered = df[(df["health_score_0to100"] >= min_health) & (df["health_score_0to100"] <= max_health)].copy()
if show_mandatory:
    filtered = filtered.sort_values("health_score_0to100", ascending=True)
else:
    filtered = filtered.sort_values("health_score_0to100", ascending=False)

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

display_cols = [
    "project_id", "project_name", "health_score_0to100", "delivery_confidence_pct"
]
if "p_on_time_pct" in filtered.columns:
    display_cols.append("p_on_time_pct")
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
    selected = st.selectbox("Select a project", project_names)
    row = filtered[filtered["project_name"] == selected].iloc[0]

    st.write("**Explanation**")
    st.write(row.get("explanation", "No explanation column found."))

    st.write("**Drivers (weighted contributions)**")
    driver_df = pd.DataFrame({
        "driver": drivers,
        "contribution": [float(row[d]) if pd.notna(row[d]) else 0.0 for d in drivers],
    }).sort_values("contribution", ascending=False)

    st.dataframe(driver_df, use_container_width=True)
else:
    st.info("No driver_ columns found in the scored CSV.")
