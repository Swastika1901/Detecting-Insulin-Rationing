"""
tests/test_pipeline_consistency.py
===================================
Integration test verifying that every beneficiary in the scored risk output
also appears in the saved cohort events file, ensuring the dashboard can
always render fill history for any patient it displays.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import main as run_pipeline_main  # noqa: E402


def test_all_scored_ids_exist_in_cohort_events(tmp_path):
    """Every DESYNPUF_ID in patient_risk_scores.csv must exist in cohort_events.csv."""
    with patch("scripts.run_pipeline.load_cohort", side_effect=FileNotFoundError("No raw")):
        with patch("scripts.run_pipeline.PROJECT_ROOT", tmp_path):
            run_pipeline_main()

    scores_path = tmp_path / "data" / "processed" / "patient_risk_scores.csv"
    events_path = tmp_path / "data" / "processed" / "cohort_events.csv"

    assert scores_path.exists(), "patient_risk_scores.csv was not created"
    assert events_path.exists(), "cohort_events.csv was not created"

    scores_df = pd.read_csv(scores_path)
    events_df = pd.read_csv(events_path)

    scored_ids = set(scores_df["DESYNPUF_ID"])
    event_ids = set(events_df["DESYNPUF_ID"])

    missing = scored_ids - event_ids
    assert not missing, (
        f"{len(missing)} scored patient(s) have no cohort events: {sorted(missing)[:5]}"
    )
