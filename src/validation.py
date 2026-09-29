"""
validation.py
=============
Synthetic data perturbation for validating insulin rationing detection pipelines.

Why Synthetic Validation?
-------------------------
CMS DE-SynPUF medical claims data contains no ground-truth labels for insulin
rationing. To empirically prove that our feature engineering pipeline and
unsupervised IsolationForest model can detect rationing behavior, we inject
synthetic rationing into a subset of patients.

Rationing is simulated by progressively expanding the gap (in days) between
consecutive refills across a patient's fill history. If the pipeline is working
as designed, the IsolationForest model will flag injected patients at a
significantly higher rate than unperturbed patients.
"""

from __future__ import annotations

import logging
from typing import List, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def inject_synthetic_rationing(
    df: pd.DataFrame,
    fraction: float = 0.1,
    random_state: int = 42,
    multiplier_step: float = 0.35,
) -> Tuple[pd.DataFrame, List[str]]:
    """Inject synthetic rationing behavior into a random subset of patients.

    For each selected patient, refill dates are adjusted so that the gap
    between refill :math:`i` and refill :math:`i+1` expands progressively by a
    factor of :math:`(1.0 + i \\times \\text{multiplier\\_step})`.

    Parameters
    ----------
    df : pd.DataFrame
        Cohort DataFrame containing at least ``DESYNPUF_ID`` and ``SRVC_DT``.
    fraction : float, default=0.1
        Fraction of unique patients to select for synthetic rationing injection.
    random_state : int, default=42
        Random seed for reproducible patient selection.
    multiplier_step : float, default=0.35
        The incremental multiplier added to each successive refill gap.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        1. A copy of *df* with modified ``SRVC_DT`` for injected patients.
        2. List of ``DESYNPUF_ID`` strings representing the injected patients.

    Raises
    ------
    KeyError
        If ``DESYNPUF_ID`` or ``SRVC_DT`` are missing from *df*.
    """
    if "DESYNPUF_ID" not in df.columns or "SRVC_DT" not in df.columns:
        raise KeyError("Input DataFrame must contain 'DESYNPUF_ID' and 'SRVC_DT'.")

    if df.empty:
        return df.copy(), []

    work_df = df.copy()
    work_df["SRVC_DT"] = pd.to_datetime(work_df["SRVC_DT"])

    # Unique patients
    unique_patients = work_df["DESYNPUF_ID"].unique()
    n_total = len(unique_patients)
    n_injected = int(np.ceil(n_total * fraction))

    rng = np.random.RandomState(random_state)
    injected_patients: List[str] = list(
        rng.choice(unique_patients, size=n_injected, replace=False)
    )
    injected_set: Set[str] = set(injected_patients)

    logger.info(
        "Injecting synthetic rationing into %d / %d patients (%.1f%%)",
        n_injected,
        n_total,
        fraction * 100.0,
    )

    # Sort fills by patient and date before applying perturbation
    work_df = work_df.sort_values(["DESYNPUF_ID", "SRVC_DT"]).reset_index(drop=True)

    # Modify fill dates for injected patients
    modified_rows = []
    for pid, group in work_df.groupby("DESYNPUF_ID", sort=False):
        group_copy = group.copy()
        if pid in injected_set and len(group_copy) > 1:
            dates = group_copy["SRVC_DT"].tolist()
            new_dates = [dates[0]]

            for i in range(len(dates) - 1):
                orig_gap_days = (dates[i + 1] - dates[i]).days
                # Prevent negative or 0 gaps
                orig_gap_days = max(1, orig_gap_days)

                # Progressive expansion factor: 1.0, 1.35, 1.70, 2.05...
                multiplier = 1.0 + (i * multiplier_step)
                new_gap_days = max(1, int(round(orig_gap_days * multiplier)))

                new_date = new_dates[-1] + pd.Timedelta(days=new_gap_days)
                new_dates.append(new_date)

            group_copy["SRVC_DT"] = new_dates

        modified_rows.append(group_copy)

    result_df = pd.concat(modified_rows, ignore_index=True)
    # Convert SRVC_DT back to date object format matching load_cohort output
    result_df["SRVC_DT"] = result_df["SRVC_DT"].dt.date

    return result_df, injected_patients
