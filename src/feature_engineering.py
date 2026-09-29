"""
feature_engineering.py
======================
Extracts patient-level features from insulin prescription fill data.

Typical usage::

    from src.data_loader import load_cohort
    from src.feature_engineering import build_patient_features

    df_cohort = load_cohort()
    df_features = build_patient_features(df_cohort)

Feature Definitions
-------------------
For each patient (``DESYNPUF_ID``) with at least ``min_fills`` prescription fills:

1. ``num_fills``:
   Total count of insulin prescription fills.
2. ``actual_gap_days``:
   Days between fill :math:`i` and fill :math:`i+1` (``SRVC_DT_{i+1} - SRVC_DT_i``).
3. ``gap_vs_supply_ratio``:
   Ratio of ``actual_gap_days`` to ``DAYS_SUPLY_NUM`` for fill :math:`i`.
4. ``mean_gap_ratio``:
   Mean of ``gap_vs_supply_ratio`` across all valid gap observations.
5. ``std_gap_ratio``:
   Sample standard deviation (ddof=1) of ``gap_vs_supply_ratio``.
6. ``gap_trend_slope``:
   Linear regression slope of ``gap_vs_supply_ratio`` over the fill sequence
   index (:math:`0, 1, 2, \\dots`). A positive slope indicates increasing delay
   relative to supply over time (potential rationing).
7. ``pay_amt_trend``:
   Linear regression slope of ``PTNT_PAY_AMT`` over time (days elapsed since
   the patient's first fill).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"DESYNPUF_ID", "SRVC_DT", "DAYS_SUPLY_NUM", "PTNT_PAY_AMT"}
FEATURE_COLUMNS = [
    "DESYNPUF_ID",
    "num_fills",
    "mean_gap_ratio",
    "std_gap_ratio",
    "gap_trend_slope",
    "pay_amt_trend",
]


def build_patient_features(df: pd.DataFrame, min_fills: int = 4) -> pd.DataFrame:
    """Compute patient-level summary features from insulin fill events.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing cohort fill events. Must contain at least the
        columns ``DESYNPUF_ID``, ``SRVC_DT``, ``DAYS_SUPLY_NUM``, and
        ``PTNT_PAY_AMT``.
    min_fills : int, default=4
        Minimum number of fills required for a patient to be included.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per patient (with :math:`\\ge \\text{min\\_fills}`) and
        columns: ``DESYNPUF_ID``, ``num_fills``, ``mean_gap_ratio``,
        ``std_gap_ratio``, ``gap_trend_slope``, ``pay_amt_trend``.

    Raises
    ------
    KeyError
        If required columns are missing from *df*.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            f"Input DataFrame is missing required column(s): {sorted(missing)}"
        )

    if df.empty:
        logger.warning("Input DataFrame is empty. Returning empty feature DataFrame.")
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    # Make a copy and standardise column dtypes
    work_df = df.copy()
    work_df["SRVC_DT"] = pd.to_datetime(work_df["SRVC_DT"])
    work_df["DAYS_SUPLY_NUM"] = pd.to_numeric(
        work_df["DAYS_SUPLY_NUM"], errors="coerce"
    )
    work_df["PTNT_PAY_AMT"] = pd.to_numeric(
        work_df["PTNT_PAY_AMT"], errors="coerce"
    )

    # Step 1: Sort fills by patient and fill date
    work_df = work_df.sort_values(["DESYNPUF_ID", "SRVC_DT"]).reset_index(
        drop=True
    )

    # Step 4: Drop patients with fewer than min_fills
    counts = work_df.groupby("DESYNPUF_ID", sort=False).size()
    eligible_ids = set(counts[counts >= min_fills].index)
    logger.info(
        "Feature engineering: %d total patients, %d have >= %d fills",
        len(counts),
        len(eligible_ids),
        min_fills,
    )

    if not eligible_ids:
        logger.warning("No patients with >= %d fills found.", min_fills)
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    work_df = work_df[work_df["DESYNPUF_ID"].isin(eligible_ids)].copy()

    # Step 2: Compute actual_gap_days (days until next fill)
    next_srvc_dt = work_df.groupby("DESYNPUF_ID", sort=False)["SRVC_DT"].shift(
        -1
    )
    work_df["actual_gap_days"] = (next_srvc_dt - work_df["SRVC_DT"]).dt.days

    # Step 3: Compute gap_vs_supply_ratio = actual_gap_days / DAYS_SUPLY_NUM
    supply_nonzero = work_df["DAYS_SUPLY_NUM"].replace(0, np.nan)
    work_df["gap_vs_supply_ratio"] = work_df["actual_gap_days"] / supply_nonzero

    # Step 5: Compute patient-level feature aggregations
    records: List[dict] = []
    for pid, group in work_df.groupby("DESYNPUF_ID", sort=False):
        n_fills = len(group)

        # Gap ratios (excluding the last fill which has actual_gap_days = NaN)
        gap_ratios = group["gap_vs_supply_ratio"].dropna().values

        if len(gap_ratios) > 0:
            mean_gap = float(np.mean(gap_ratios))
            std_gap = (
                float(np.std(gap_ratios, ddof=1)) if len(gap_ratios) > 1 else 0.0
            )
        else:
            mean_gap = np.nan
            std_gap = np.nan

        # Linear regression of gap_vs_supply_ratio over fill sequence index
        if len(gap_ratios) >= 2:
            x_seq = np.arange(len(gap_ratios), dtype=float)
            res_gap = linregress(x_seq, gap_ratios)
            gap_slope = (
                float(res_gap.slope) if not np.isnan(res_gap.slope) else 0.0
            )
        else:
            gap_slope = 0.0

        # Linear regression of PTNT_PAY_AMT over time (days elapsed since 1st fill)
        pay_amts = group["PTNT_PAY_AMT"].values
        srvc_dts = group["SRVC_DT"]
        days_from_start = (
            (srvc_dts - srvc_dts.iloc[0]).dt.total_seconds().values / 86400.0
        )

        if len(pay_amts) >= 2 and float(np.var(days_from_start)) > 0:
            res_pay = linregress(days_from_start, pay_amts)
            pay_slope = (
                float(res_pay.slope) if not np.isnan(res_pay.slope) else 0.0
            )
        else:
            pay_slope = 0.0

        records.append(
            {
                "DESYNPUF_ID": pid,
                "num_fills": n_fills,
                "mean_gap_ratio": mean_gap,
                "std_gap_ratio": std_gap,
                "gap_trend_slope": gap_slope,
                "pay_amt_trend": pay_slope,
            }
        )

    return pd.DataFrame(records, columns=FEATURE_COLUMNS)
