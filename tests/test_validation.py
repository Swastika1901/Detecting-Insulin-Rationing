"""
tests/test_validation.py
========================
Unit tests for :func:`src.validation.inject_synthetic_rationing`.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

# Ensure ``src`` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation import inject_synthetic_rationing  # noqa: E402


def test_inject_synthetic_rationing_expands_gaps():
    """Synthetic injection expands gap between consecutive fills."""
    df = pd.DataFrame(
        {
            "DESYNPUF_ID": ["BENE_01"] * 4,
            "SRVC_DT": ["2020-01-01", "2020-01-31", "2020-03-02", "2020-04-01"],
            "DAYS_SUPLY_NUM": [30] * 4,
            "PTNT_PAY_AMT": [10.0] * 4,
        }
    )

    injected_df, injected_ids = inject_synthetic_rationing(
        df, fraction=1.0, random_state=42, multiplier_step=0.5
    )

    assert injected_ids == ["BENE_01"]
    assert len(injected_df) == 4

    # Check that gaps expand progressively
    dates = pd.to_datetime(injected_df["SRVC_DT"]).tolist()
    gap0 = (dates[1] - dates[0]).days
    gap1 = (dates[2] - dates[1]).days
    gap2 = (dates[3] - dates[2]).days

    assert gap0 == 30  # multiplier = 1.0 (30 * 1.0)
    assert gap1 > gap0  # multiplier = 1.5 (31 * 1.5 = ~46)
    assert gap2 > gap1  # multiplier = 2.0 (30 * 2.0 = 60)


def test_inject_synthetic_rationing_fraction():
    """Correct fraction of patients is injected."""
    df = pd.DataFrame(
        {
            "DESYNPUF_ID": [f"BENE_{i:02d}" for i in range(10) for _ in range(4)],
            "SRVC_DT": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"] * 10,
            "DAYS_SUPLY_NUM": [30] * 40,
            "PTNT_PAY_AMT": [10.0] * 40,
        }
    )

    injected_df, injected_ids = inject_synthetic_rationing(
        df, fraction=0.3, random_state=42
    )

    assert len(injected_ids) == 3  # 30% of 10 patients
    assert len(injected_df) == 40


def test_missing_columns_raises_keyerror():
    """Missing required columns raises KeyError."""
    df = pd.DataFrame({"DESYNPUF_ID": ["BENE_01"]})

    with pytest.raises(KeyError, match="must contain 'DESYNPUF_ID' and 'SRVC_DT'"):
        inject_synthetic_rationing(df)


def test_robustness_csv_produced_and_auc_in_range():
    """Verify report/robustness.csv is produced and all AUC values lie in [0, 1]."""
    csv_path = Path(__file__).resolve().parent.parent / "report" / "robustness.csv"
    assert csv_path.exists(), f"Expected robustness CSV at {csv_path}"

    df = pd.read_csv(csv_path)
    assert not df.empty, "report/robustness.csv is empty"

    required_cols = [
        "multiplier_step",
        "injection_fraction",
        "iforest_auc_mean",
        "iforest_auc_std",
        "logreg_auc_mean",
        "logreg_auc_std",
        "confounder_flagged_pct",
        "leakage_auc_drop",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in robustness.csv"

    # Check AUC values are in [0, 1]
    for _, row in df.iterrows():
        assert 0.0 <= row["iforest_auc_mean"] <= 1.0, f"Invalid iforest_auc_mean: {row['iforest_auc_mean']}"
        assert 0.0 <= row["logreg_auc_mean"] <= 1.0, f"Invalid logreg_auc_mean: {row['logreg_auc_mean']}"
        assert row["iforest_auc_std"] >= 0.0
        assert row["logreg_auc_std"] >= 0.0
        assert 0.0 <= row["confounder_flagged_pct"] <= 100.0


def test_run_robustness_experiment_fast():
    """Fast execution test of run_robustness_experiment producing valid AUCs."""
    from scripts.run_validation import run_robustness_experiment

    res_df = run_robustness_experiment(
        multiplier_steps=[0.20],
        injection_fractions=[0.10],
        n_seeds=2,
        n_patients=60,
    )

    assert not res_df.empty
    assert len(res_df) == 1
    row = res_df.iloc[0]
    assert 0.0 <= row["iforest_auc_mean"] <= 1.0
    assert 0.0 <= row["logreg_auc_mean"] <= 1.0

