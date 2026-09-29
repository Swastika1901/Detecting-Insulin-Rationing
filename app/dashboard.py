"""
app/dashboard.py
================
Streamlit Web Dashboard for Detecting Insulin Rationing in CMS Claims Data.

Features:
1. Interactive overview of processed beneficiaries & risk score distribution.
2. Sortable table of flagged high-risk patients.
3. Patient deep-dive explorer with metric color indicators (Green <40, Amber 40-70, Red >70).
4. Per-fill longitudinal line chart of `gap_vs_supply_ratio` over fill sequence
   loaded from data/processed/cohort_events.csv.
5. Sidebar disclaimer explaining synthetic DE-SynPUF demonstration purpose.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page config
st.set_page_config(
    page_title="Insulin Rationing Detection Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look and feel
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1.2rem;
        border-radius: 12px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .badge-red {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-amber {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-green {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_risk_scores() -> pd.DataFrame:
    """Load scored patient risk predictions CSV."""
    path = PROJECT_ROOT / "data" / "processed" / "patient_risk_scores.csv"
    if not path.exists():
        st.error(
            f"Processed risk scores file not found at `{path}`. Please run `python scripts/run_pipeline.py` first."
        )
        st.stop()
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_cohort_events() -> pd.DataFrame:
    """Load raw cohort events CSV saved by run_pipeline.py."""
    path = PROJECT_ROOT / "data" / "processed" / "cohort_events.csv"
    if not path.exists():
        st.error(
            f"Processed cohort events file not found at `{path}`. Please run `python scripts/run_pipeline.py` first."
        )
        st.stop()
    df = pd.read_csv(path)
    return df


def get_patient_fill_history(df_events: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    """Extract and compute per-fill actual gap days and gap_vs_supply_ratio for a patient."""
    p_df = df_events[df_events["DESYNPUF_ID"] == patient_id].copy()
    if p_df.empty:
        return pd.DataFrame()

    p_df["SRVC_DT"] = pd.to_datetime(p_df["SRVC_DT"])
    p_df = p_df.sort_values("SRVC_DT").reset_index(drop=True)
    p_df["fill_seq"] = np.arange(len(p_df)) + 1

    next_dt = p_df["SRVC_DT"].shift(-1)
    p_df["actual_gap_days"] = (next_dt - p_df["SRVC_DT"]).dt.days
    p_df["DAYS_SUPLY_NUM"] = pd.to_numeric(p_df["DAYS_SUPLY_NUM"], errors="coerce")
    p_df["gap_vs_supply_ratio"] = p_df["actual_gap_days"] / p_df["DAYS_SUPLY_NUM"].replace(0, np.nan)
    return p_df


def main():
    # Sidebar
    st.sidebar.title("🩸 Insulin Rationing AI")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "ℹ️ **Demonstration Purpose Only**\n\n"
        "This application processes synthetic and sample CMS DE-SynPUF claims data. "
        "It uses unsupervised IsolationForest anomaly detection to flag potential insulin rationing behavior."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Detecting Insulin Rationing Project © 2026")

    # Main Header
    st.markdown('<div class="main-header">Insulin Rationing Surveillance Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Unsupervised Anomaly Detection & Risk Scoring over Refill-Gap and Cost-Burden Features</div>',
        unsafe_allow_html=True,
    )

    # Load Data
    risk_df = load_risk_scores()
    events_df = load_cohort_events()

    # Overview KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_patients = len(risk_df)
    flagged_patients = int(risk_df["flagged"].sum())
    flagged_pct = (flagged_patients / total_patients * 100.0) if total_patients > 0 else 0.0
    avg_risk = float(risk_df["risk_score"].mean()) if total_patients > 0 else 0.0

    col1.metric("Total Beneficiaries", f"{total_patients}")
    col2.metric("Flagged High Risk", f"{flagged_patients}", delta=f"{flagged_pct:.1f}%", delta_color="inverse")
    col3.metric("Average Risk Score", f"{avg_risk:.1f} / 100")
    col4.metric("Contamination Rate", "5.0%")

    st.markdown("---")

    # Section 1: Flagged High Risk Patients Table
    st.subheader("⚠️ High-Risk Flagged Patients")

    show_all = st.checkbox("Show all patients (including non-flagged)", value=False)
    if show_all:
        table_df = risk_df.copy()
    else:
        table_df = risk_df[risk_df["flagged"]].copy()

    if table_df.empty:
        st.info("No patients currently flagged in this dataset slice.")
    else:
        st.dataframe(
            table_df.sort_values("risk_score", ascending=False),
            use_container_width=True,
            column_config={
                "DESYNPUF_ID": "Beneficiary ID",
                "risk_score": st.column_config.NumberColumn("Risk Score (0-100)", format="%.2f"),
                "mean_gap_ratio": st.column_config.NumberColumn("Mean Gap Ratio", format="%.3f"),
                "gap_trend_slope": st.column_config.NumberColumn("Gap Trend Slope", format="%.4f"),
                "pay_amt_trend": st.column_config.NumberColumn("Pay Trend ($/day)", format="%.4f"),
                "flagged": st.column_config.CheckboxColumn("Flagged Anomaly"),
            },
            hide_index=True,
        )

    st.markdown("---")

    # Section 2: Patient Deep Dive Explorer
    st.subheader("🔍 Beneficiary Longitudinal Deep Dive")

    all_pids = risk_df["DESYNPUF_ID"].tolist()
    flagged_pids = risk_df[risk_df["flagged"]]["DESYNPUF_ID"].tolist()

    # Default to first flagged patient if available
    default_idx = all_pids.index(flagged_pids[0]) if flagged_pids and flagged_pids[0] in all_pids else 0

    selected_pid = st.selectbox(
        "Select Beneficiary ID to inspect refill history:",
        options=all_pids,
        index=default_idx,
    )

    if selected_pid:
        patient_row = risk_df[risk_df["DESYNPUF_ID"] == selected_pid].iloc[0]
        risk_val = float(patient_row["risk_score"])
        is_flagged = bool(patient_row["flagged"])

        # Determine Risk Level & Color Badge
        if risk_val > 70:
            badge_html = f'<div class="badge-red">HIGH RISK ({risk_val:.1f} / 100) — FLAGGED</div>'
        elif risk_val >= 40:
            badge_html = f'<div class="badge-amber">MODERATE RISK ({risk_val:.1f} / 100)</div>'
        else:
            badge_html = f'<div class="badge-green">LOW RISK ({risk_val:.1f} / 100)</div>'

        detail_col1, detail_col2 = st.columns([1, 2])

        with detail_col1:
            st.markdown("#### Patient Risk Summary")
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.write(f"**Beneficiary ID:** `{selected_pid}`")
            st.write(f"**Total Fills:** `{int(patient_row['num_fills'])}`")
            st.write(f"**Mean Gap Ratio:** `{patient_row['mean_gap_ratio']:.3f}`")
            st.write(f"**Std Gap Ratio:** `{patient_row['std_gap_ratio']:.3f}`")
            st.write(f"**Gap Trend Slope:** `{patient_row['gap_trend_slope']:.4f}`")
            st.write(f"**Pay Amount Trend:** `{patient_row['pay_amt_trend']:.4f}`")

        with detail_col2:
            st.markdown("#### Longitudinal Refill Gap vs Supply Ratio")
            p_history = get_patient_fill_history(events_df, selected_pid)

            if p_history.empty:
                st.info("No fill history available")
            else:
                valid_history = p_history.dropna(subset=["gap_vs_supply_ratio"]).copy()
                if valid_history.empty:
                    st.info("No fill history available")
                else:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(
                        valid_history["fill_seq"],
                        valid_history["gap_vs_supply_ratio"],
                        marker="o",
                        linewidth=2.5,
                        color="#E63946" if is_flagged else "#1D3557",
                        label="Gap vs Supply Ratio",
                    )
                    ax.axhline(1.0, color="#64748B", linestyle="--", label="Expected Supply Ratio (1.0)")
                    ax.set_xlabel("Fill Sequence Index", fontsize=10, fontweight="bold")
                    ax.set_ylabel("Actual Gap / Days Supply Ratio", fontsize=10, fontweight="bold")
                    ax.set_title(f"Refill Delay Trend for {selected_pid}", fontsize=11, fontweight="bold")
                    ax.grid(True, linestyle=":", alpha=0.6)
                    ax.legend(loc="upper left")
                    st.pyplot(fig)


if __name__ == "__main__":
    main()
