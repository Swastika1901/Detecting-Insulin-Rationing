"""
tests/test_data_loader.py
=========================
Unit tests for :func:`src.data_loader.load_cohort` using small, fully
synthetic CSV fixtures written to a temporary directory.

These tests do **not** touch the real data files in ``data/raw/``; every
CSV is fabricated in-process so the suite runs offline and instantly.

Fixture design
--------------
* Two beneficiaries: ``BENE_01`` (diabetic, SP_DIABETES=1) and
  ``BENE_02`` (non-diabetic, SP_DIABETES=2).
* Four PDE events:
    - BENE_01 + insulin NDC      → should appear in the result  ✓
    - BENE_01 + non-insulin NDC  → filtered out at NDC step     ✗
    - BENE_02 + insulin NDC      → filtered out at ID step      ✗
    - BENE_01 + insulin NDC (dashed format in PDE to test normalisation) ✓
* NDC reference: one code in dashed format, one in run-together format,
  both corresponding to the same real 11-digit value.
"""

from __future__ import annotations

import datetime
import textwrap
from pathlib import Path

import pandas as pd
import pytest

# Make ``src`` importable regardless of how pytest is invoked.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_cohort, _normalise_ndc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INSULIN_NDC_DASHED = "00169-7501-11"          # NovoLog – dashed format
INSULIN_NDC_PLAIN   = "00169750111"            # same code, run-together (11-digit)
NON_INSULIN_NDC     = "99999999999"            # clearly not insulin


_BENE_HEADER = (
    "DESYNPUF_ID,BENE_BIRTH_DT,BENE_DEATH_DT,BENE_SEX_IDENT_CD,"
    "BENE_RACE_CD,BENE_ESRD_IND,SP_STATE_CODE,BENE_COUNTY_CD,"
    "BENE_HI_CVRAGE_TOT_MONS,BENE_SMI_CVRAGE_TOT_MONS,"
    "BENE_HMO_CVRAGE_TOT_MONS,PLAN_CVRG_MOS_NUM,"
    "SP_ALZHDMTA,SP_CHF,SP_CHRNKIDN,SP_CNCR,SP_COPD,SP_DEPRESSN,"
    "SP_DIABETES,SP_ISCHMCHT,SP_OSTEOPRS,SP_RA_OA,SP_STRKETIA,"
    "MEDREIMB_IP,BENRES_IP,PPPYMT_IP,MEDREIMB_OP,BENRES_OP,PPPYMT_OP,"
    "MEDREIMB_CAR,BENRES_CAR,PPPYMT_CAR"
)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Populate *tmp_path* with synthetic Beneficiary and PDE CSVs."""

    # ---- Beneficiary Summary ----------------------------------------
    bene_csv = tmp_path / "Beneficiary_Summary_Sample.csv"
    bene_content = (
        _BENE_HEADER + "\n"
        "BENE_01,19500101,,1,1,0,10,100,12,12,0,12,2,2,2,2,2,2,1,2,2,2,2,0,0,0,50,10,0,100,20,0\n"
        "BENE_02,19600202,,2,1,0,10,200,12,12,0,12,2,2,2,2,2,2,2,2,2,2,2,0,0,0,30,5,0,80,10,0\n"
    )
    bene_csv.write_text(bene_content, encoding="utf-8")

    # ---- PDE -----------------------------------------------------------
    # PROD_SRVC_ID purposely mixes formats to exercise normalisation:
    #   row 1 – insulin, dashed format (same as reference CSV)  → survives
    #   row 2 – insulin, run-together 11-digit (CMS native)     → survives
    #   row 3 – non-insulin NDC                                  → filtered at NDC step
    #   row 4 – insulin NDC but BENE_02 (non-diabetic)          → filtered at ID step
    pde_csv = tmp_path / "Prescription_Drug_Events_Sample.csv"
    pde_content = (
        "DESYNPUF_ID,PDE_ID,SRVC_DT,PROD_SRVC_ID,QTY_DSPNSD_NUM,"
        "DAYS_SUPLY_NUM,PTNT_PAY_AMT,TOT_RX_CST_AMT\n"
        f"BENE_01,PDE001,20090315,{INSULIN_NDC_DASHED},30.000,30,10.00,120.00\n"
        f"BENE_01,PDE002,20090615,{INSULIN_NDC_PLAIN},60.000,30,10.00,120.00\n"
        f"BENE_01,PDE003,20090901,{NON_INSULIN_NDC},30.000,30,5.00,60.00\n"
        f"BENE_02,PDE004,20090401,{INSULIN_NDC_PLAIN},30.000,30,10.00,120.00\n"
    )
    pde_csv.write_text(pde_content, encoding="utf-8")

    return tmp_path


@pytest.fixture()
def ndc_path(tmp_path: Path) -> Path:
    """Write a minimal insulin NDC reference CSV to a dedicated subdirectory.

    Using a subdirectory (``ndc/``) keeps the NDC file out of the same flat
    directory as the bene/PDE CSVs so the file-discovery patterns in
    :func:`~src.data_loader._find_csv` do not accidentally pick it up.
    """
    ndc_dir = tmp_path / "ndc"
    ndc_dir.mkdir()
    ndc_csv = ndc_dir / "insulin_ndc_list.csv"
    ndc_csv.write_text(
        "ndc_code,drug_name,brand_name,labeler,source_url\n"
        f"{INSULIN_NDC_DASHED},NovoLog (insulin aspart) 100 units/mL vial,"
        "NovoLog,Novo Nordisk,https://api.fda.gov/drug/ndc.json\n",
        encoding="utf-8",
    )
    return ndc_csv


@pytest.fixture()
def cohort(data_dir: Path, ndc_path: Path) -> pd.DataFrame:
    """Run load_cohort() against the synthetic fixtures."""
    return load_cohort(data_dir=data_dir, ndc_path=ndc_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormaliseNdc:
    """Unit tests for the internal NDC normalisation helper."""

    def test_strips_dashes(self):
        # "0088-2220-33" → strip dashes → "008822033" (9 digits, not 11)
        # wait: 0+0+8+8+2+2+2+0+3+3 = 10 digits → "0088222033" → zfill(11) = "00088222033"
        # Actual: strip "0088-2220-33" → "008822033"? No:
        # "0088" + "2220" + "33" = "0088" "2220" "33" = "008822033"
        # Count: 0,0,8,8,2,2,2,0,3,3 = 10 digits → zfill(11) → "00088222033"
        s = pd.Series(["0088-2220-33"])
        assert _normalise_ndc(s).iloc[0] == "00088222033"

    def test_zero_pads_short_code(self):
        # 9-digit product NDC (no package code) — must pad to 11
        s = pd.Series(["169-7501"])
        result = _normalise_ndc(s).iloc[0]
        assert len(result) == 11

    def test_plain_11_digit_unchanged(self):
        s = pd.Series(["00169750111"])
        assert _normalise_ndc(s).iloc[0] == "00169750111"

    def test_vectorised(self):
        s = pd.Series(["0088-2220-33", "00169750111"])
        result = _normalise_ndc(s)
        assert result.tolist() == ["00088222033", "00169750111"]


class TestLoadCohortShape:
    """High-level shape and content tests for load_cohort()."""

    def test_returns_dataframe(self, cohort: pd.DataFrame):
        assert isinstance(cohort, pd.DataFrame)

    def test_exactly_two_rows_survive(self, cohort: pd.DataFrame):
        """PDE001 and PDE002 survive; PDE003 (non-insulin) and PDE004
        (non-diabetic) are both filtered out."""
        assert len(cohort) == 2

    def test_only_diabetic_beneficiary_present(self, cohort: pd.DataFrame):
        assert set(cohort["DESYNPUF_ID"].unique()) == {"BENE_01"}
        assert "BENE_02" not in cohort["DESYNPUF_ID"].values

    def test_non_insulin_ndc_excluded(self, cohort: pd.DataFrame):
        """NON_INSULIN_NDC must not appear in the result."""
        normalised_non_insulin = NON_INSULIN_NDC.zfill(11)
        prod_ids_norm = _normalise_ndc(cohort["PROD_SRVC_ID"])
        assert normalised_non_insulin not in prod_ids_norm.values

    def test_srvc_dt_parsed_as_date(self, cohort: pd.DataFrame):
        """SRVC_DT must be Python date objects, not raw integers or strings."""
        assert cohort["SRVC_DT"].iloc[0] == datetime.date(2009, 3, 15)
        assert all(isinstance(d, datetime.date) for d in cohort["SRVC_DT"])

    def test_beneficiary_columns_merged(self, cohort: pd.DataFrame):
        """Beneficiary Summary columns (e.g. SP_DIABETES) must be present
        after the left join."""
        assert "SP_DIABETES" in cohort.columns
        assert (cohort["SP_DIABETES"] == 1).all()

    def test_pde_id_column_present(self, cohort: pd.DataFrame):
        assert "PDE_ID" in cohort.columns


class TestLoadCohortNdcNormalisation:
    """Verify that dashed and run-together NDC formats both match."""

    def test_dashed_ndc_in_pde_is_matched(self, cohort: pd.DataFrame):
        """PDE001 has a dashed PROD_SRVC_ID; it must survive the filter."""
        assert "PDE001" in cohort["PDE_ID"].values

    def test_plain_ndc_in_pde_is_matched(self, cohort: pd.DataFrame):
        """PDE002 has a plain 11-digit PROD_SRVC_ID; it must survive."""
        assert "PDE002" in cohort["PDE_ID"].values


class TestLoadCohortErrors:
    """Edge-case and error-handling tests."""

    def test_missing_data_dir_raises(self, tmp_path: Path, ndc_path: Path):
        """Passing a non-existent data directory must raise FileNotFoundError."""
        bad_dir = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Beneficiary Summary"):
            load_cohort(data_dir=bad_dir, ndc_path=ndc_path)

    def test_missing_ndc_file_raises(self, data_dir: Path, tmp_path: Path):
        """Pointing to a non-existent NDC file must raise FileNotFoundError."""
        bad_ndc = tmp_path / "nonexistent" / "ndc.csv"
        with pytest.raises(FileNotFoundError, match="insulin_ndc_list"):
            load_cohort(data_dir=data_dir, ndc_path=bad_ndc)

    def test_missing_sp_diabetes_column_raises(
        self, tmp_path: Path, ndc_path: Path
    ):
        """A Beneficiary Summary without SP_DIABETES must raise KeyError."""
        bad_bene = tmp_path / "Beneficiary_Summary_bad.csv"
        bad_bene.write_text(
            "DESYNPUF_ID,BENE_BIRTH_DT\nBENE_01,19500101\n",
            encoding="utf-8",
        )
        pde = tmp_path / "Prescription_Drug_Events_bad.csv"
        pde.write_text(
            "DESYNPUF_ID,PDE_ID,SRVC_DT,PROD_SRVC_ID,"
            "QTY_DSPNSD_NUM,DAYS_SUPLY_NUM,PTNT_PAY_AMT,TOT_RX_CST_AMT\n",
            encoding="utf-8",
        )
        with pytest.raises(KeyError, match="SP_DIABETES"):
            load_cohort(data_dir=tmp_path, ndc_path=ndc_path)

    def test_empty_diabetic_cohort_returns_empty_df(
        self, tmp_path: Path, ndc_path: Path
    ):
        """If no beneficiary has SP_DIABETES == 1, the result must be empty."""
        nodm_dir = tmp_path / "nodm"
        nodm_dir.mkdir()
        bene = nodm_dir / "Beneficiary_Summary_nodm.csv"
        bene.write_text(
            _BENE_HEADER + "\n"
            "BENE_01,19500101,,1,1,0,10,100,12,12,0,12,2,2,2,2,2,2,"
            "2,2,2,2,2,0,0,0,50,10,0,100,20,0\n",
            encoding="utf-8",
        )
        pde = nodm_dir / "Prescription_Drug_Events_nodm.csv"
        pde.write_text(
            "DESYNPUF_ID,PDE_ID,SRVC_DT,PROD_SRVC_ID,"
            "QTY_DSPNSD_NUM,DAYS_SUPLY_NUM,PTNT_PAY_AMT,TOT_RX_CST_AMT\n"
            f"BENE_01,PDE001,20090315,{INSULIN_NDC_PLAIN},30,30,10,120\n",
            encoding="utf-8",
        )
        result = load_cohort(data_dir=nodm_dir, ndc_path=ndc_path)
        assert len(result) == 0
