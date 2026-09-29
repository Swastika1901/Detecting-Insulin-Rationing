"""
tests/test_model.py
===================
Unit tests for :func:`src.model.train_anomaly_model` and :func:`src.model.score_patients`.
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Make ``src`` importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import FEATURE_COLS, score_patients, train_anomaly_model  # noqa: E402


@pytest.fixture()
def synthetic_features_df() -> pd.DataFrame:
    """Create synthetic patient features DataFrame with 20 patients."""
    np.random.seed(42)
    n = 20
    df = pd.DataFrame(
        {
            "DESYNPUF_ID": [f"BENE_{i:02d}" for i in range(n)],
            "num_fills": np.random.randint(4, 12, size=n),
            "mean_gap_ratio": np.random.normal(1.0, 0.1, size=n),
            "std_gap_ratio": np.random.uniform(0.05, 0.2, size=n),
            "gap_trend_slope": np.random.normal(0.0, 0.05, size=n),
            "pay_amt_trend": np.random.normal(0.0, 0.1, size=n),
        }
    )
    # Inject one obvious anomaly (extreme positive gap trend slope and high pay trend)
    df.iloc[-1, df.columns.get_loc("gap_trend_slope")] = 2.5
    df.iloc[-1, df.columns.get_loc("pay_amt_trend")] = 5.0
    df.iloc[-1, df.columns.get_loc("mean_gap_ratio")] = 3.0
    return df


def test_train_anomaly_model_returns_fitted_objects(synthetic_features_df, tmp_path):
    """train_anomaly_model returns fitted IsolationForest and StandardScaler."""
    model, scaler = train_anomaly_model(
        synthetic_features_df, contamination=0.05, save_dir=tmp_path
    )

    assert isinstance(model, IsolationForest)
    assert isinstance(scaler, StandardScaler)

    # Verify artifacts were written to disk
    assert (tmp_path / "isolation_forest.joblib").exists()
    assert (tmp_path / "scaler.joblib").exists()

    # Verify loaded artifacts match
    loaded_model = joblib.load(tmp_path / "isolation_forest.joblib")
    assert isinstance(loaded_model, IsolationForest)


def test_score_patients_adds_columns_and_ranges(synthetic_features_df, tmp_path):
    """score_patients appends anomaly_score, risk_score, and flagged columns."""
    model, scaler = train_anomaly_model(
        synthetic_features_df, contamination=0.05, save_dir=tmp_path
    )

    scored_df = score_patients(model, scaler, synthetic_features_df)

    assert "anomaly_score" in scored_df.columns
    assert "risk_score" in scored_df.columns
    assert "flagged" in scored_df.columns

    # Length matches input
    assert len(scored_df) == len(synthetic_features_df)

    # Risk scores are within [0, 100]
    assert scored_df["risk_score"].min() >= 0.0
    assert scored_df["risk_score"].max() <= 100.0

    # Flagged is boolean
    assert scored_df["flagged"].dtype == bool


def test_extreme_outlier_gets_highest_risk_score(synthetic_features_df, tmp_path):
    """The injected extreme outlier patient receives risk_score of 100.0 and flagged=True."""
    model, scaler = train_anomaly_model(
        synthetic_features_df, contamination=0.05, save_dir=tmp_path
    )
    scored_df = score_patients(model, scaler, synthetic_features_df)

    outlier_row = scored_df.iloc[-1]
    assert outlier_row["flagged"] is True or outlier_row["flagged"] == np.bool_(True)
    assert outlier_row["risk_score"] == 100.0


def test_missing_columns_raises_keyerror(tmp_path):
    """Missing required feature columns raises KeyError."""
    incomplete_df = pd.DataFrame({"DESYNPUF_ID": ["BENE_01"], "gap_trend_slope": [0.1]})

    with pytest.raises(KeyError, match="missing required feature column"):
        train_anomaly_model(incomplete_df, save_dir=tmp_path)


def test_empty_dataframe_raises_valueerror(tmp_path):
    """Empty DataFrame raises ValueError during training."""
    empty_df = pd.DataFrame(columns=FEATURE_COLS)

    with pytest.raises(ValueError, match="empty DataFrame"):
        train_anomaly_model(empty_df, save_dir=tmp_path)
