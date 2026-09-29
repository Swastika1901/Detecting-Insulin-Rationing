"""
tests/test_feature_engineering.py
==================================
Unit tests for :func:`src.feature_engineering.build_patient_features`.

Tests use synthetic DataFrames to verify:
1. Patients with fewer than 4 fills are dropped.
2. Rising gap_vs_supply_ratio over fill sequence produces a positive gap_trend_slope.
3. Falling gap_vs_supply_ratio produces a negative gap_trend_slope.
4. Pay amount trends are correctly computed.
5. Edge cases (empty DataFrame, missing columns, constant values) are handled cleanly.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd
import pytest

# Ensure `src` is importable regardless of test runner working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import build_patient_features  # noqa: E402


def test_rising_gap_ratio_positive_slope():
    """Rising ratio over time (e.g. 30d, 40d, 50d gaps for 30d supply) -> positive slope."""
    df = pd.DataFrame(
        {
            "DESYNPUF_ID": ["BENE_RISING"] * 4,
            "SRVC_DT": [
                "2020-01-01",
                "2020-01-31",  # 30 day gap -> ratio 30/30 = 1.0
                "2020-03-11",  # 40 day gap -> ratio 40/30 = 1.333
                "2020-04-30",  # 50 day gap -> ratio 50/30 = 1.667
            ],
            "DAYS_SUPLY_NUM": [30, 30, 30, 30],
            "PTNT_PAY_AMT": [10.0, 10.0, 10.0, 10.0],
        }
    )

    res = build_patient_features(df)

    assert len(res) == 1
    assert res.iloc[0]["DESYNPUF_ID"] == "BENE_RISING"
    assert res.iloc[0]["num_fills"] == 4

    # Rising gap ratio -> positive slope
    slope = res.iloc[0]["gap_trend_slope"]
    assert slope > 0, f"Expected positive slope, got {slope}"
    # Mean gap ratio should be (1.0 + 40/30 + 50/30) / 3 = 4/3 ~ 1.3333
    assert math.isclose(res.iloc[0]["mean_gap_ratio"], 4 / 3, rel_tol=1e-3)


def test_falling_gap_ratio_negative_slope():
    """Falling ratio over time (e.g. 50d, 40d, 30d gaps) -> negative slope."""
    df = pd.DataFrame(
        {
            "DESYNPUF_ID": ["BENE_FALLING"] * 4,
            "SRVC_DT": [
                "2020-01-01",
                "2020-02-20",  # 50 day gap -> ratio 50/30 = 1.667
                "2020-03-31",  # 40 day gap -> ratio 40/30 = 1.333
                "2020-04-30",  # 30 day gap -> ratio 30/30 = 1.0
            ],
            "DAYS_SUPLY_NUM": [30, 30, 30, 30],
            "PTNT_PAY_AMT": [10.0, 10.0, 10.0, 10.0],
        }
    )

    res = build_patient_features(df)

    assert len(res) == 1
    slope = res.iloc[0]["gap_trend_slope"]
    assert slope < 0, f"Expected negative slope, got {slope}"


def test_pay_amt_trend_sign():
    """Increasing copay over time yields positive pay_amt_trend."""
    df = pd.DataFrame(
        {
            "DESYNPUF_ID": ["BENE_PAY"] * 4,
            "SRVC_DT": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
            "DAYS_SUPLY_NUM": [30, 30, 30, 30],
            "PTNT_PAY_AMT": [10.0, 25.0, 40.0, 60.0],
        }
    )

    res = build_patient_features(df)

    assert len(res) == 1
    pay_slope = res.iloc[0]["pay_amt_trend"]
    assert pay_slope > 0, f"Expected positive pay_amt_trend, got {pay_slope}"


def test_drops_patients_fewer_than_4_fills():
    """Patients with < 4 fills are excluded, patients with >= 4 are retained."""
    df = pd.DataFrame(
        {
            "DESYNPUF_ID": ["BENE_3"] * 3 + ["BENE_4"] * 4,
            "SRVC_DT": [
                "2020-01-01", "2020-02-01", "2020-03-01",
                "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01",
            ],
            "DAYS_SUPLY_NUM": [30] * 7,
            "PTNT_PAY_AMT": [10.0] * 7,
        }
    )

    res = build_patient_features(df)

    assert len(res) == 1
    assert res.iloc[0]["DESYNPUF_ID"] == "BENE_4"
    assert res.iloc[0]["num_fills"] == 4


def test_missing_columns_raises_keyerror():
    """Input missing required columns raises KeyError."""
    df = pd.DataFrame({"DESYNPUF_ID": ["BENE_1"], "SRVC_DT": ["2020-01-01"]})
    with pytest.raises(KeyError, match="missing required column"):
        build_patient_features(df)


def test_empty_dataframe_returns_empty_features():
    """Empty input DataFrame returns empty feature DataFrame with correct columns."""
    df = pd.DataFrame(columns=["DESYNPUF_ID", "SRVC_DT", "DAYS_SUPLY_NUM", "PTNT_PAY_AMT"])
    res = build_patient_features(df)

    assert res.empty
    expected_cols = [
        "DESYNPUF_ID",
        "num_fills",
        "mean_gap_ratio",
        "std_gap_ratio",
        "gap_trend_slope",
        "pay_amt_trend",
    ]
    assert list(res.columns) == expected_cols
