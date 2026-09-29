"""
scripts/run_pipeline.py
=======================
End-to-end execution pipeline for insulin rationing detection:

Pipeline steps:
1. `load_cohort()`: Filter CMS DE-SynPUF beneficiary & PDE data to diabetic insulin users.
   Falls back gracefully to `generate_synthetic_cohort(n_patients=150)` if data/raw is empty.
2. `build_patient_features()`: Aggregate per-patient refill gaps, slopes, and cost trends.
3. `train_anomaly_model()`: Fit StandardScaler and IsolationForest; save artifacts to `models/`.
4. `score_patients()`: Compute raw anomaly scores, 0-100 risk scores, and anomaly flags.
5. Save raw cohort events to `data/processed/cohort_events.csv`.
6. Save scored predictions to `data/processed/patient_risk_scores.csv` sorted by `risk_score` descending.
7. Display summary statistics and print the top 10 flagged patients.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cohort  # noqa: E402
from src.feature_engineering import build_patient_features  # noqa: E402
from src.model import score_patients, train_anomaly_model  # noqa: E402
from src.synthetic import generate_synthetic_cohort  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("      INSULIN RATIONING DETECTION – END-TO-END PROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Load Cohort (with try/except FileNotFoundError fallback)
    logger.info("STEP 1: Loading diabetic insulin cohort from CMS DE-SynPUF data...")
    is_synthetic_run = False
    try:
        df_cohort = load_cohort(
            data_dir=PROJECT_ROOT / "data" / "raw",
            ndc_path=PROJECT_ROOT / "data" / "reference" / "insulin_ndc_list.csv",
        )
        logger.info(
            "Cohort loaded: %d total PDE fill records across %d unique beneficiaries.",
            len(df_cohort),
            df_cohort["DESYNPUF_ID"].nunique(),
        )

        # Verify if raw slice contains enough patients with >= 4 fills
        features_test = build_patient_features(df_cohort, min_fills=4)
        if features_test.empty:
            logger.warning(
                "Raw data slice contains no beneficiaries with >= 4 fills. Falling back to synthetic cohort."
            )
            print("\n" + "*" * 70)
            print("               RUNNING ON SYNTHETIC DATA")
            print("*" * 70 + "\n")
            is_synthetic_run = True
            df_cohort = generate_synthetic_cohort(n_patients=150)
    except FileNotFoundError as exc:
        logger.warning(
            "Raw CMS data files not found in data/raw/ (%s). Falling back to synthetic cohort.",
            exc,
        )
        print("\n" + "*" * 70)
        print("               RUNNING ON SYNTHETIC DATA")
        print("*" * 70 + "\n")
        is_synthetic_run = True
        df_cohort = generate_synthetic_cohort(n_patients=150)

    # Step 2: Save Cohort Events to data/processed/cohort_events.csv
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    events_output_path = output_dir / "cohort_events.csv"
    df_cohort.to_csv(events_output_path, index=False)
    logger.info("Saved %d raw cohort fill event records to %s", len(df_cohort), events_output_path)

    # Step 3: Extract Patient Features
    logger.info("STEP 3: Building patient-level refill gap and cost burden features...")
    features_df = build_patient_features(df_cohort, min_fills=4)

    logger.info(
        "Features extracted for %d beneficiaries across %d metrics.",
        len(features_df),
        len(features_df.columns) - 1,
    )

    # Step 4: Train Anomaly Model
    logger.info("STEP 4: Training IsolationForest anomaly detection model...")
    model_dir = PROJECT_ROOT / "models"
    model, scaler = train_anomaly_model(
        features_df,
        contamination=0.05,
        save_dir=model_dir,
        random_state=42,
    )

    # Step 5: Score Patients
    logger.info("STEP 5: Scoring patient risk profiles...")
    scored_df = score_patients(model, scaler, features_df)

    # Step 6: Sort by risk_score descending & save output CSV
    logger.info("STEP 6: Sorting and saving output to data/processed/patient_risk_scores.csv...")
    scored_df = scored_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    risk_output_path = output_dir / "patient_risk_scores.csv"
    scored_df.to_csv(risk_output_path, index=False)
    logger.info("Saved %d scored patient records to %s", len(scored_df), risk_output_path)

    # Step 7: Print Summary & Top 10 Flagged Patients
    total_patients = len(scored_df)
    flagged_df = scored_df[scored_df["flagged"]]
    total_flagged = len(flagged_df)
    flagged_pct = (total_flagged / total_patients * 100.0) if total_patients > 0 else 0.0

    print("\n" + "=" * 80)
    print("                            PIPELINE EXECUTION SUMMARY")
    if is_synthetic_run:
        print("                            (RUNNING ON SYNTHETIC DATA)")
    print("=" * 80)
    print(f"  Total Beneficiaries Processed:  {total_patients}")
    print(f"  Beneficiaries Flagged:          {total_flagged} ({flagged_pct:.1f}%)")
    print(f"  Events Saved To:                {events_output_path}")
    print(f"  Scores Saved To:                {risk_output_path}")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("                       TOP 10 HIGHEST RISK BENEFICIARIES")
    print("-" * 80)

    display_cols = [
        "DESYNPUF_ID",
        "num_fills",
        "mean_gap_ratio",
        "gap_trend_slope",
        "pay_amt_trend",
        "risk_score",
        "flagged",
    ]
    top_10 = scored_df.head(10)[display_cols]

    # Format header & rows
    print(
        f"{'ID':<15} | {'Fills':<6} | {'Mean Gap Ratio':<14} | {'Gap Trend':<10} | {'Pay Trend':<10} | {'Risk Score':<10} | {'Flagged':<7}"
    )
    print("-" * 80)
    for _, row in top_10.iterrows():
        print(
            f"{str(row['DESYNPUF_ID']):<15} | "
            f"{int(row['num_fills']):<6} | "
            f"{float(row['mean_gap_ratio']):<14.3f} | "
            f"{float(row['gap_trend_slope']):<10.4f} | "
            f"{float(row['pay_amt_trend']):<10.4f} | "
            f"{float(row['risk_score']):<10.2f} | "
            f"{str(row['flagged']):<7}"
        )
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
