"""Streamlit dashboard for MLB GBSV predictions & analytics.

Run:
    streamlit run src/mlbv1/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path when run directly
_src = str(Path(__file__).resolve().parents[2])
if _src not in sys.path:
    sys.path.insert(0, _src)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from mlbv1.tracking.database import TrackingDB  # noqa: E402
from mlbv1.tracking.roi import BankrollConfig, BankrollManager  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MLB GBSV Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚾ MLB GBSV v1.0")
st.sidebar.markdown("---")

db_path = st.sidebar.text_input("Tracking DB Path", value="artifacts/tracking.db")
db = TrackingDB(db_path)

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Predictions", "Model Comparison", "Bankroll", "Runs"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("*MLB GBSV v1.0 — Spread Prediction*")

# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------

if page == "Dashboard":
    st.title("MLB GBSV — Overview")

    # ROI Summary
    roi = db.get_roi_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bets", roi.get("total_bets", 0))
    col2.metric("Win Rate", f"{roi.get('win_rate', 0):.1%}")
    col3.metric("ROI", f"{roi.get('roi', 0):.1%}")
    col4.metric("Net Profit", f"${roi.get('net_profit', 0):,.2f}")

    st.markdown("---")

    # Recent predictions
    st.subheader("Recent Predictions")
    recent = db.get_predictions(limit=20)
    if recent:
        df = pd.DataFrame(recent)
        display_cols = [
            "game_date",
            "home_team",
            "away_team",
            "spread",
            "prediction",
            "probability",
            "model_name",
            "settled",
            "actual_result",
        ]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, hide_index=True)
    else:
        st.info(
            "No predictions logged yet. Run `scripts/daily_run.py` to generate predictions."
        )

    # Model comparison chart
    st.subheader("Model Comparison")
    comparison = db.get_model_comparison()
    if comparison:
        comp_df = pd.DataFrame(comparison)
        st.bar_chart(comp_df.set_index("model_name")["accuracy"])
    else:
        st.info("No model comparison data available yet.")

# ---------------------------------------------------------------------------
# Predictions page
# ---------------------------------------------------------------------------

elif page == "Predictions":
    st.title("Prediction History")

    col1, col2 = st.columns(2)
    with col1:
        filter_settled = st.selectbox("Status", ["All", "Settled", "Unsettled"])
    with col2:
        filter_limit = st.slider("Max results", 10, 500, 100)

    settled_filter = None
    if filter_settled == "Settled":
        settled_filter = True
    elif filter_settled == "Unsettled":
        settled_filter = False

    preds = db.get_predictions(settled=settled_filter, limit=filter_limit)
    if preds:
        df = pd.DataFrame(preds)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Win/loss chart for settled
        settled_df = df[df["settled"] == 1].copy()
        if not settled_df.empty and "actual_result" in settled_df.columns:
            settled_df["correct"] = (
                settled_df["actual_result"] == settled_df["prediction"]
            )
            st.subheader("Accuracy Over Time")
            settled_df["game_date"] = pd.to_datetime(settled_df["game_date"])
            daily = (
                settled_df.groupby(settled_df["game_date"].dt.date)["correct"]
                .mean()
                .reset_index()
            )
            daily.columns = ["date", "accuracy"]
            st.line_chart(daily.set_index("date"))
    else:
        st.info("No predictions found.")

# ---------------------------------------------------------------------------
# Model Comparison page
# ---------------------------------------------------------------------------

elif page == "Model Comparison":
    st.title("Model Comparison")

    comparison = db.get_model_comparison()
    if comparison:
        df = pd.DataFrame(comparison)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Accuracy by Model")
        st.bar_chart(df.set_index("model_name")[["accuracy"]])

        st.subheader("Total Predictions by Model")
        st.bar_chart(df.set_index("model_name")[["total_predictions"]])
    else:
        st.info("No model data available yet.")

    # Runs table
    runs = db.get_runs()
    if runs:
        st.subheader("Training Runs")
        runs_df = pd.DataFrame(runs)
        st.dataframe(runs_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Bankroll page
# ---------------------------------------------------------------------------

elif page == "Bankroll":
    st.title("Bankroll Tracker")

    col1, col2, col3 = st.columns(3)
    initial = col1.number_input("Initial Balance ($)", value=10000, step=1000)
    unit = col2.number_input("Base Unit ($)", value=100, step=25)
    kelly_frac = col3.number_input("Kelly Fraction", value=0.25, step=0.05)

    config = BankrollConfig(
        initial_balance=float(initial),
        base_unit=float(unit),
        kelly_fraction=kelly_frac,
    )
    manager = BankrollManager(config)

    # Simulate bankroll from settled predictions
    settled = db.get_predictions(settled=True, limit=5000)
    if settled:
        settled_df = pd.DataFrame(settled)
        settled_df = settled_df.sort_values("game_date")

        balances = [config.initial_balance]
        dates = []
        for _, row in settled_df.iterrows():
            won = row.get("actual_result") == row.get("prediction")
            payout = config.base_unit * (100.0 / 110.0) if won else -config.base_unit
            manager.balance += payout
            balances.append(manager.balance)
            dates.append(row["game_date"])

        if dates:
            bankroll_df = pd.DataFrame({"date": dates, "balance": balances[1:]})
            bankroll_df["date"] = pd.to_datetime(bankroll_df["date"])

            st.subheader("Bankroll Progression")
            st.line_chart(bankroll_df.set_index("date")["balance"])

            stats = manager.get_stats()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Balance", f"${manager.balance:,.2f}")
            col2.metric(
                "Net Profit", f"${manager.balance - config.initial_balance:,.2f}"
            )
            col3.metric("Max Drawdown", f"{stats.get('max_drawdown', 0):.1%}")
            col4.metric("Sharpe Ratio", f"{stats.get('sharpe', 0):.3f}")
    else:
        st.info(
            "No settled predictions — bankroll simulation requires historical results."
        )

# ---------------------------------------------------------------------------
# Runs page
# ---------------------------------------------------------------------------

elif page == "Runs":
    st.title("Training & Prediction Runs")

    runs = db.get_runs()
    if runs:
        df = pd.DataFrame(runs)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Select a run to drill into
        run_ids = df["run_id"].tolist()
        selected = st.selectbox("Select Run", run_ids)
        if selected:
            roi = db.get_roi_summary(run_id=selected)
            preds = db.get_predictions(run_id=selected, limit=200)

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{roi.get('win_rate', 0):.1%}")
            col2.metric("ROI", f"{roi.get('roi', 0):.1%}")
            col3.metric("Bets", roi.get("total_bets", 0))

            if preds:
                st.subheader("Predictions for This Run")
                st.dataframe(
                    pd.DataFrame(preds), use_container_width=True, hide_index=True
                )
    else:
        st.info("No runs recorded yet.")
