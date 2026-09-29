"""
tests/test_synthetic.py
========================
Unit tests for :func:`src.synthetic.generate_synthetic_cohort` and pipeline synthetic fallback.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import pytest

# Ensure ``src`` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import main as run_pipeline_main  # noqa: E402
from src.synthetic import generate_synthetic_cohort  # noqa: E402


def test_generate_synthetic_cohort_shape_and_columns():
    """generate_synthetic_cohort creates requested patients and required columns."""
    df = generate_synthetic_cohort(n_patients=50, random_state=42)

    assert isinstance(df, pd.DataFrame)
    expected_cols = {"DESYNPUF_ID", "SRVC_DT", "DAYS_SUPLY_NUM", "PTNT_PAY_AMT", "SP_DIABETES"}
    assert expected_cols.issubset(df.columns)

    # Unique patients count matches requested n_patients
    assert df["DESYNPUF_ID"].nunique() == 50

    # Every patient has at least 4 fills
    counts = df.groupby("DESYNPUF_ID").size()
    assert (counts >= 4).all()


def test_run_pipeline_synthetic_fallback(tmp_path, capsys):
    """run_pipeline falls back to synthetic cohort and prints RUNNING ON SYNTHETIC DATA when raw files missing."""
    with patch("scripts.run_pipeline.load_cohort", side_effect=FileNotFoundError("No raw files")):
        with patch("scripts.run_pipeline.PROJECT_ROOT", tmp_path):
            run_pipeline_main()

    captured = capsys.readouterr().out
    assert "RUNNING ON SYNTHETIC DATA" in captured
    assert (tmp_path / "data" / "processed" / "patient_risk_scores.csv").exists()
