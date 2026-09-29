"""
data_loader.py
==============
Loads and filters CMS DE-SynPUF data for the insulin-rationing analysis.

Typical usage::

    from src.data_loader import load_cohort

    df = load_cohort()          # uses default paths
    df = load_cohort(           # explicit override
        data_dir="data/raw",
        ndc_path="data/reference/insulin_ndc_list.csv",
    )

Pipeline summary
----------------
1. Read the Beneficiary Summary CSV; keep only beneficiaries with
   ``SP_DIABETES == 1`` (chronic condition flag for diabetes).
2. Read the Prescription Drug Event (PDE) CSV.
3. Read the insulin NDC reference list and filter PDE rows to those
   whose ``PROD_SRVC_ID`` matches an insulin NDC code.
4. Further restrict PDE rows to ``DESYNPUF_ID`` values present in the
   diabetic cohort identified in step 1.
5. Parse ``SRVC_DT`` (format ``%Y%m%d``) to a proper ``datetime.date``.
6. Return the merged DataFrame, retaining all beneficiary columns plus
   the filtered PDE columns joined on ``DESYNPUF_ID``.

NDC normalisation
-----------------
The FDA NDC reference CSV stores codes in dashed ``5-4-2`` format
(e.g. ``0088-2220-33``).  CMS PDE stores ``PROD_SRVC_ID`` as an
11-digit run-together string (e.g. ``00882220033``).  Both are
normalised to zero-padded 11-digit strings before comparison so that
no codes are missed due to formatting differences.

File discovery
--------------
The function looks for the *first* file in ``data_dir`` whose name
matches the patterns below (case-insensitive):

* Beneficiary Summary  → ``*beneficiary*summary*.csv``
* PDE                  → ``*prescription*drug*.csv``  (or ``*pde*.csv``)

This avoids hard-coding the exact DE-SynPUF sample filename so the
loader works across multiple sample files.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BENE_PATTERN = re.compile(r"beneficiary.+summary", re.IGNORECASE)
_PDE_PATTERN = re.compile(r"(prescription.+drug|pde)", re.IGNORECASE)


def _find_csv(directory: Path, pattern: re.Pattern[str], label: str) -> Path:
    """Return the first CSV in *directory* whose stem matches *pattern*.

    Parameters
    ----------
    directory:
        Directory to search (non-recursive).
    pattern:
        Compiled regex tested against the filename (stem + suffix).
    label:
        Human-readable name used in error messages.

    Returns
    -------
    Path
        Path to the matching file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No {label} CSV found: directory does not exist: {directory}"
        )
    candidates = [
        p for p in directory.iterdir()
        if p.suffix.lower() == ".csv" and pattern.search(p.name)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No {label} CSV found in {directory}. "
            f"Expected a file matching /{pattern.pattern}/i."
        )
    if len(candidates) > 1:
        logger.warning(
            "Multiple %s CSVs found in %s; using %s",
            label,
            directory,
            candidates[0].name,
        )
    return candidates[0]


def _normalise_ndc(series: pd.Series) -> pd.Series:
    """Strip dashes and zero-pad NDC codes to 11 digits.

    Handles both the dashed FDA format (``0088-2220-33``) and the
    CMS run-together format (``00882220033``) so that they compare
    equal.

    Parameters
    ----------
    series:
        String Series of NDC codes.

    Returns
    -------
    pd.Series
        Normalised 11-digit NDC strings (dtype ``object``).
    """
    return series.astype(str).str.replace("-", "", regex=False).str.zfill(11)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_cohort(
    data_dir: str | Path = "data/raw",
    ndc_path: str | Path = "data/reference/insulin_ndc_list.csv",
    bene_dtype: Optional[dict] = None,
    pde_dtype: Optional[dict] = None,
) -> pd.DataFrame:
    """Load, filter, and merge CMS DE-SynPUF beneficiary and PDE data.

    The function follows a strict filtering pipeline and logs row counts
    at every step so that the caller can sanity-check data loss.

    Parameters
    ----------
    data_dir:
        Directory containing the Beneficiary Summary CSV and the PDE CSV.
        File names are discovered by pattern matching so the exact
        DE-SynPUF sample number need not be hard-coded.
    ndc_path:
        Path to the insulin NDC reference CSV produced by
        ``src/reference/build_insulin_ndc_list.py``.  Must contain at
        least a column named ``ndc_code``.
    bene_dtype:
        Optional dtype override dict forwarded to ``pd.read_csv`` when
        reading the Beneficiary Summary file.  Useful in tests.
    pde_dtype:
        Optional dtype override dict forwarded to ``pd.read_csv`` when
        reading the PDE file.  Useful in tests.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with one row per insulin PDE event for
        diabetic beneficiaries.  Columns include all PDE fields plus all
        Beneficiary Summary fields (joined on ``DESYNPUF_ID``).
        ``SRVC_DT`` is cast to ``datetime.date``.

    Raises
    ------
    FileNotFoundError
        If any of the three required input files cannot be located.
    KeyError
        If expected columns (``SP_DIABETES``, ``DESYNPUF_ID``,
        ``PROD_SRVC_ID``, ``SRVC_DT``) are absent from the loaded files.
    """
    data_dir = Path(data_dir)
    ndc_path = Path(ndc_path)

    # ------------------------------------------------------------------
    # Step 1 – Load Beneficiary Summary and filter to diabetic cohort
    # ------------------------------------------------------------------
    bene_path = _find_csv(data_dir, _BENE_PATTERN, "Beneficiary Summary")
    logger.info("Loading Beneficiary Summary from %s", bene_path)

    bene_df = pd.read_csv(bene_path, dtype=bene_dtype)
    logger.info("  Beneficiary rows loaded       : %d", len(bene_df))

    if "SP_DIABETES" not in bene_df.columns:
        raise KeyError(
            f"Column 'SP_DIABETES' not found in {bene_path.name}. "
            f"Available columns: {bene_df.columns.tolist()}"
        )

    diabetic_df = bene_df[bene_df["SP_DIABETES"] == 1].copy()
    logger.info(
        "  After SP_DIABETES == 1 filter : %d  (dropped %d)",
        len(diabetic_df),
        len(bene_df) - len(diabetic_df),
    )

    diabetic_ids: set[str] = set(diabetic_df["DESYNPUF_ID"].astype(str))

    # ------------------------------------------------------------------
    # Step 2 – Load PDE
    # ------------------------------------------------------------------
    pde_path = _find_csv(data_dir, _PDE_PATTERN, "PDE")
    logger.info("Loading PDE from %s", pde_path)

    pde_df = pd.read_csv(pde_path, dtype=pde_dtype)
    logger.info("  PDE rows loaded               : %d", len(pde_df))

    # ------------------------------------------------------------------
    # Step 3 – Load insulin NDC list and filter PDE to insulin events
    # ------------------------------------------------------------------
    if not ndc_path.exists():
        raise FileNotFoundError(
            f"Insulin NDC reference file not found: {ndc_path}. "
            "Run src/reference/build_insulin_ndc_list.py first."
        )

    logger.info("Loading insulin NDC reference from %s", ndc_path)
    ndc_df = pd.read_csv(ndc_path, usecols=["ndc_code"])
    insulin_ndcs: set[str] = set(_normalise_ndc(ndc_df["ndc_code"]))
    logger.info("  Insulin NDC codes in reference: %d unique codes", len(insulin_ndcs))

    if "PROD_SRVC_ID" not in pde_df.columns:
        raise KeyError(
            f"Column 'PROD_SRVC_ID' not found in {pde_path.name}. "
            f"Available columns: {pde_df.columns.tolist()}"
        )

    pde_df["_ndc_norm"] = _normalise_ndc(pde_df["PROD_SRVC_ID"])
    insulin_pde = pde_df[pde_df["_ndc_norm"].isin(insulin_ndcs)].copy()
    insulin_pde.drop(columns=["_ndc_norm"], inplace=True)
    logger.info(
        "  PDE rows after insulin NDC filter : %d  (dropped %d)",
        len(insulin_pde),
        len(pde_df) - len(insulin_pde),
    )

    # ------------------------------------------------------------------
    # Step 4 – Restrict to diabetic beneficiaries
    # ------------------------------------------------------------------
    if "DESYNPUF_ID" not in pde_df.columns:
        raise KeyError(
            f"Column 'DESYNPUF_ID' not found in {pde_path.name}. "
            f"Available columns: {pde_df.columns.tolist()}"
        )

    insulin_diabetic = insulin_pde[
        insulin_pde["DESYNPUF_ID"].astype(str).isin(diabetic_ids)
    ].copy()
    logger.info(
        "  PDE rows after diabetic ID filter : %d  (dropped %d)",
        len(insulin_diabetic),
        len(insulin_pde) - len(insulin_diabetic),
    )

    # ------------------------------------------------------------------
    # Step 5 – Parse SRVC_DT
    # ------------------------------------------------------------------
    if "SRVC_DT" not in insulin_diabetic.columns:
        raise KeyError(
            f"Column 'SRVC_DT' not found in {pde_path.name}. "
            f"Available columns: {pde_df.columns.tolist()}"
        )

    insulin_diabetic["SRVC_DT"] = pd.to_datetime(
        insulin_diabetic["SRVC_DT"].astype(str),
        format="%Y%m%d",
    ).dt.date
    logger.info("  SRVC_DT parsed to date format.")

    # ------------------------------------------------------------------
    # Step 6 – Merge with beneficiary columns and return
    # ------------------------------------------------------------------
    merged = insulin_diabetic.merge(
        diabetic_df,
        on="DESYNPUF_ID",
        how="left",
        suffixes=("", "_bene"),
    )
    logger.info(
        "  Final merged rows             : %d  (beneficiary cols added via left join)",
        len(merged),
    )

    return merged
