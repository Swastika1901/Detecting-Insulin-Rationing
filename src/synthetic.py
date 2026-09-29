"""
synthetic.py
============
Generates synthetic CMS DE-SynPUF beneficiary and prescription fill datasets
for testing, demonstration, and fallback execution when raw data files are absent.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_cohort(
    n_patients: int = 150,
    random_state: int = 42,
    min_fills_per_patient: int = 4,
    max_fills_per_patient: int = 10,
) -> pd.DataFrame:
    """Generate a synthetic diabetic cohort with insulin prescription fill events.

    Parameters
    ----------
    n_patients : int, default=150
        Number of unique synthetic beneficiaries to generate.
    random_state : int, default=42
        Random seed for reproducibility.
    min_fills_per_patient : int, default=4
        Minimum number of prescription fills per patient.
    max_fills_per_patient : int, default=10
        Maximum number of prescription fills per patient.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``DESYNPUF_ID``, ``SRVC_DT``,
        ``DAYS_SUPLY_NUM``, ``PTNT_PAY_AMT``, ``SP_DIABETES``.
    """
    rng = np.random.RandomState(random_state)
    rows = []

    start_date = pd.Timestamp("2020-01-01")
    for i in range(n_patients):
        pid = f"SYN_BENE_{i:04d}"
        n_fills = rng.randint(min_fills_per_patient, max_fills_per_patient + 1)
        curr_date = start_date + pd.Timedelta(days=int(rng.randint(0, 30)))
        base_copay = float(rng.uniform(10.0, 50.0))

        for f_idx in range(n_fills):
            # Normal fill gap: 28 to 32 days for a 30-day supply
            gap_days = int(rng.randint(28, 33))
            if f_idx > 0:
                curr_date += pd.Timedelta(days=gap_days)

            # Random variations in copay
            copay = max(5.0, round(base_copay + float(rng.normal(0, 2.0)), 2))

            rows.append(
                {
                    "DESYNPUF_ID": pid,
                    "SRVC_DT": curr_date.date(),
                    "DAYS_SUPLY_NUM": 30,
                    "PTNT_PAY_AMT": copay,
                    "SP_DIABETES": 1,
                }
            )

    df = pd.DataFrame(rows)
    logger.info(
        "Generated synthetic cohort with %d events across %d beneficiaries.",
        len(df),
        n_patients,
    )
    return df
