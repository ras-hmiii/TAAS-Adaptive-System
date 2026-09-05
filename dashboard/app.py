"""Streamlit research dashboard backed by saved experiment artifacts."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


RESULTS_DIR = Path("results")


def load_experiment_results() -> list[dict[str, Any]]:
    """Load JSON experiment records without inventing placeholder metrics."""
    if not RESULTS_DIR.is_dir():
        return []
    records = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        if isinstance(record, dict):
            records.append(record)
    return records


def render_overview(records: list[dict[str, Any]]) -> None:
    st.header("TAAS")
    st.caption("Traceable & Adaptive AI System")
    st.write("Explainable and adaptive machine learning under distribution shift.")

    if not records:
        st.info("Run an experiment to populate model health, drift, and adaptation results.")
        return

    latest = records[-1]
    metrics = latest.get("metrics", {})
    first, second, third = st.columns(3)
    first.metric("Experiment", latest.get("experiment_id", "Unavailable"))
    second.metric("Model", latest.get("model", "Unavailable"))
    third.metric("Status", latest.get("status", "Unavailable"))

    if metrics:
        st.subheader("Observed metrics")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)


def render_experiments(records: list[dict[str, Any]]) -> None:
    st.header("Experiments")
    if not records:
        st.info("No experiment artifacts are available yet.")
        return
    rows = []
    for record in records:
        rows.append(
            {
                "Experiment": record.get("experiment_id", "Unavailable"),
                "Dataset": record.get("dataset", "Unavailable"),
                "Model": record.get("model", "Unavailable"),
                "Detector": record.get("detector", "Unavailable"),
                "Adaptation": record.get("adaptation_strategy", "Unavailable"),
                "Status": record.get("status", "Unavailable"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_audit(records: list[dict[str, Any]]) -> None:
    st.header("Audit trail")
    events = [event for record in records for event in record.get("events", [])]
    if not events:
        st.info("No audit events are available yet.")
        return
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)


def render_predictions(records: list[dict[str, Any]]) -> None:
    st.header("Predictions")
    predictions = [event for record in records for event in record.get("predictions", [])]
    if not predictions:
        st.info("No prediction artifacts are available yet.")
        return
    st.dataframe(pd.DataFrame(predictions), use_container_width=True, hide_index=True)


def render_drift(records: list[dict[str, Any]]) -> None:
    st.header("Drift monitor")
    events = [event for record in records for event in record.get("drift_events", [])]
    if not events:
        st.info("Run a streaming drift experiment to populate this view.")
        return
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)


def render_adaptation(records: list[dict[str, Any]]) -> None:
    st.header("Adaptation")
    events = [event for record in records for event in record.get("adaptation_events", [])]
    if not events:
        st.info("No adaptation decisions are available yet.")
        return
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)


def render_comparison(records: list[dict[str, Any]]) -> None:
    st.header("Model comparison")
    rows = []
    for record in records:
        row = {"Experiment": record.get("experiment_id", "Unavailable")}
        row.update(record.get("metrics", {}))
        rows.append(row)
    if not rows:
        st.info("Run comparable experiments to populate this view.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_research_results(records: list[dict[str, Any]]) -> None:
    st.header("Research results")
    if not records:
        st.info("Run the required experiment to populate this result.")
        return
    st.write("Results below are calculated from saved experiment artifacts.")
    render_comparison(records)


def main() -> None:
    st.set_page_config(page_title="TAAS Research Console", page_icon=None, layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8fa; color: #111827; }
        [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e5e7eb; padding: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    records = load_experiment_results()
    page = st.sidebar.radio(
        "Research console",
        [
            "Overview",
            "Experiments",
            "Predictions",
            "Explainability",
            "Drift Monitor",
            "Adaptation",
            "Model Comparison",
            "Audit Trail",
            "Research Results",
            "Settings",
        ],
    )
    if page == "Overview":
        render_overview(records)
    elif page == "Experiments":
        render_experiments(records)
    elif page == "Predictions":
        render_predictions(records)
    elif page == "Drift Monitor":
        render_drift(records)
    elif page == "Adaptation":
        render_adaptation(records)
    elif page == "Model Comparison":
        render_comparison(records)
    elif page == "Research Results":
        render_research_results(records)
    elif page == "Explainability":
        st.header("Explainability")
        st.info("Select a saved prediction and model artifact to generate SHAP or Grad-CAM output.")
    elif page == "Settings":
        st.header("Settings")
        st.write({"results_directory": str(RESULTS_DIR.resolve())})
    else:
        render_audit(records)


if __name__ == "__main__":
    main()
