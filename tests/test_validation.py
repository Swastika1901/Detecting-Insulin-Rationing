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
