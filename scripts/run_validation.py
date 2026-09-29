"""
scripts/run_validation.py
==========================
Runs the synthetic rationing validation pipeline to empirically verify that
our feature engineering and IsolationForest model detect rationing behavior.

Pipeline Steps:
1. Load cohort data (or generate synthetic cohort if data/raw/ is empty/sparse).
2. Extract baseline patient features on untouched cohort data.
3. Inject synthetic rationing into ~15% of patients using `inject_synthetic_rationing`.
4. Extract patient features on perturbed cohort data.
5. Train `IsolationForest` model & `StandardScaler` on patient features.
6. Score all patients and compare flagged rate (%) between synthetic rationers vs rest.
7. Save comparison metrics and generate `report/validation_chart.png`.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cohort  # noqa: E402
from src.feature_engineering import build_patient_features  # noqa: E402
from src.model import score_patients, train_anomaly_model  # noqa: E402
from src.validation import inject_synthetic_rationing  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_cohort(n_patients: int = 150) -> pd.DataFrame:
    """Generate a realistic synthetic cohort for offline validation execution."""
    rng = np.random.RandomState(42)
    rows = []

    start_date = pd.Timestamp("2020-01-01")
    for i in range(n_patients):
        pid = f"SYN_BENE_{i:04d}"
        n_fills = rng.randint(4, 10)
        curr_date = start_date + pd.Timedelta(days=rng.randint(0, 30))
        base_copay = float(rng.uniform(10.0, 50.0))

        for f_idx in range(n_fills):
            # Normal fill gap: 28 to 32 days for 30-day supply
            gap_days = rng.randint(28, 33)
            if f_idx > 0:
                curr_date += pd.Timedelta(days=gap_days)

            # Slight random variation in copay
            copay = max(5.0, round(base_copay + rng.normal(0, 2.0), 2))

            rows.append(
                {
                    "DESYNPUF_ID": pid,
                    "SRVC_DT": curr_date.date(),
                    "DAYS_SUPLY_NUM": 30,
                    "PTNT_PAY_AMT": copay,
                    "SP_DIABETES": 1,
                }
            )

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("  INSULIN RATIONING DETECTION – SYNTHETIC VALIDATION EXPERIMENT")
    print("=" * 70)

    # 1. Load or generate cohort data
    df_cohort = None
    try:
        logger.info("Attempting to load cohort from data/raw ...")
        df_cohort = load_cohort()
        logger.info("Successfully loaded %d cohort rows from raw data.", len(df_cohort))

        # Check if enough patients have >= 4 fills in raw slice
        feats_test = build_patient_features(df_cohort, min_fills=4)
        if len(feats_test) < 10:
            logger.warning(
                "Raw data slice has only %d beneficiaries with >= 4 fills. Using min_fills=2 for sparse sample.",
                len(feats_test),
            )
            feats_test = build_patient_features(df_cohort, min_fills=2)

        if len(feats_test) < 10:
            logger.warning("Raw sample too small for robust validation. Generating 150-patient synthetic cohort...")
            df_cohort = generate_synthetic_cohort(n_patients=150)
    except (FileNotFoundError, KeyError) as exc:
        logger.warning(
            "Raw CMS data unavailable (%s). Generating synthetic baseline cohort ...",
            exc,
        )
        df_cohort = generate_synthetic_cohort(n_patients=150)

    # 2. Inject synthetic rationing into 15% of beneficiaries
    injection_fraction = 0.15
    df_injected, injected_pids = inject_synthetic_rationing(
        df_cohort, fraction=injection_fraction, random_state=42, multiplier_step=0.40
    )
    injected_set = set(injected_pids)

    # 3. Extract patient features
    logger.info("Extracting patient-level features...")
    min_fills_thresh = 4 if df_cohort["DESYNPUF_ID"].nunique() >= 100 else 2
    features_df = build_patient_features(df_injected, min_fills=min_fills_thresh)

    if features_df.empty:
        logger.warning("features_df is empty with min_fills=%d. Using synthetic fallback.", min_fills_thresh)
        df_cohort = generate_synthetic_cohort(n_patients=150)
        df_injected, injected_pids = inject_synthetic_rationing(
            df_cohort, fraction=injection_fraction, random_state=42, multiplier_step=0.40
        )
        injected_set = set(injected_pids)
        features_df = build_patient_features(df_injected, min_fills=4)

    # Label rows with synthetic rationing status
    features_df["is_synthetic_rationer"] = features_df["DESYNPUF_ID"].isin(injected_set)

    # 4. Train anomaly detection model
    logger.info("Training IsolationForest anomaly detection model...")
    model, scaler = train_anomaly_model(
        features_df, contamination=0.10, save_dir=PROJECT_ROOT / "models"
    )

    # 5. Score patients
    scored_df = score_patients(model, scaler, features_df)

    # 6. Evaluate flagged rates
    rationers_df = scored_df[scored_df["is_synthetic_rationer"]]
    rest_df = scored_df[~scored_df["is_synthetic_rationer"]]

    total_rationers = len(rationers_df)
    flagged_rationers = int(rationers_df["flagged"].sum())
    rate_rationers = (flagged_rationers / total_rationers * 100.0) if total_rationers > 0 else 0.0

    total_rest = len(rest_df)
    flagged_rest = int(rest_df["flagged"].sum())
    rate_rest = (flagged_rest / total_rest * 100.0) if total_rest > 0 else 0.0

    mean_risk_rationers = float(rationers_df["risk_score"].mean()) if total_rationers > 0 else 0.0
    mean_risk_rest = float(rest_df["risk_score"].mean()) if total_rest > 0 else 0.0

    print("\n" + "-" * 70)
    print("  VALIDATION EXPERIMENT RESULTS")
    print("-" * 70)
    print(f"  Total Beneficiaries Analysed:  {len(scored_df)}")
    print(f"  Synthetic Rationers Injected:  {total_rationers} ({injection_fraction*100:.1f}%)")
    print(f"  Control / Rest Beneficiaries:  {total_rest}")
    print("-" * 70)
    print(f"  Synthetic Rationers Flagged:  {flagged_rationers} / {total_rationers} ({rate_rationers:.1f}%)")
    print(f"  Control (Rest) Flagged:       {flagged_rest} / {total_rest} ({rate_rest:.1f}%)")
    print(f"  Detection Lift:               {rate_rationers / max(0.1, rate_rest):.2f}x")
    print("-" * 70)
    print(f"  Mean Risk Score (Rationers):  {mean_risk_rationers:.2f} / 100")
    print(f"  Mean Risk Score (Control):    {mean_risk_rest:.2f} / 100")
    print("-" * 70 + "\n")

    # 7. Generate bar chart visualization
    report_dir = PROJECT_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    chart_path = report_dir / "validation_chart.png"

    # Styling matplotlib chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Synthetic Validation: Insulin Rationing Detection Performance",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Chart 1: Flagged Rate (%)
    categories = ["Injected Rationers", "Control Group"]
    rates = [rate_rationers, rate_rest]
    colors = ["#E63946", "#457B9D"]

    bars1 = ax1.bar(categories, rates, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Flagged Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Model Flagged Rate (%) by Group", fontsize=12)
    ax1.set_ylim(0, max(100, max(rates) * 1.15))
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Value annotations on bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Chart 2: Average Risk Score
    mean_scores = [mean_risk_rationers, mean_risk_rest]
    bars2 = ax2.bar(categories, mean_scores, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel("Mean Risk Score (0-100)", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Composite Risk Score (0-100)", fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(chart_path, dpi=300)
    plt.close()

    logger.info("Saved validation chart to %s", chart_path)
    print(f"Validation chart saved successfully to: {chart_path}\n")


if __name__ == "__main__":
    main()
