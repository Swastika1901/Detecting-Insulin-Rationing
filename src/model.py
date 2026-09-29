"""
model.py
========
Anomaly detection and risk scoring model for insulin rationing identification.

Why IsolationForest?
--------------------
In CMS DE-SynPUF prescription claims data, there are no ground-truth labels
indicating whether a beneficiary actually rationed their insulin.
Consequently, supervised classification algorithms cannot be directly trained.

IsolationForest is an unsupervised anomaly detection algorithm that explicitly
isolates anomalies by randomly selecting feature split points. Patients who
exhibit extreme refill-gap behaviors (high mean/std gap ratio, steep positive
gap trend slope) and increasing financial burden (steep copay growth) will be
isolated near the root of decision trees, receiving higher anomaly and risk
scores without requiring historical ground-truth labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "gap_trend_slope",
    "pay_amt_trend",
    "mean_gap_ratio",
    "std_gap_ratio",
]


def train_anomaly_model(
    features_df: pd.DataFrame,
    contamination: float = 0.05,
    save_dir: Optional[str | Path] = "models",
    random_state: int = 42,
) -> Tuple[IsolationForest, StandardScaler]:
    """Fit a StandardScaler and IsolationForest on patient-level features.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing patient features. Must include the columns:
        ``gap_trend_slope``, ``pay_amt_trend``, ``mean_gap_ratio``, and
        ``std_gap_ratio``.
    contamination : float, default=0.05
        The proportion of anomalies in the data set (passed to IsolationForest).
    save_dir : str | Path | None, default="models"
        Directory where fitted ``isolation_forest.joblib`` and ``scaler.joblib``
        will be saved. If None, saving to disk is skipped.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[IsolationForest, StandardScaler]
        The fitted IsolationForest model and StandardScaler.

    Raises
    ------
    KeyError
        If required feature columns are missing from *features_df*.
    ValueError
        If *features_df* is empty.
    """
    missing = set(FEATURE_COLS) - set(features_df.columns)
    if missing:
        raise KeyError(
            f"features_df is missing required feature column(s): {sorted(missing)}"
        )

    if features_df.empty:
        raise ValueError("Cannot train anomaly model on an empty DataFrame.")

    X = features_df[FEATURE_COLS].copy()

    # Fit scaler and transform features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit IsolationForest
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    logger.info(
        "Fitted IsolationForest on %d patients (contamination=%.3f)",
        len(features_df),
        contamination,
    )

    # Save fitted artifacts to disk if save_dir is specified
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        model_path = save_path / "isolation_forest.joblib"
        scaler_path = save_path / "scaler.joblib"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info("Saved model to %s and scaler to %s", model_path, scaler_path)

    return model, scaler


def score_patients(
    model: IsolationForest,
    scaler: StandardScaler,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score patient features using a fitted IsolationForest and StandardScaler.

    Adds three columns to the input DataFrame:
    1. ``anomaly_score``: Raw inverted decision score (higher = more anomalous).
    2. ``risk_score``: Min-max normalized score in range [0, 100].
    3. ``flagged``: Boolean flag indicating if patient is predicted as an anomaly.

    Parameters
    ----------
    model : IsolationForest
        Fitted IsolationForest model.
    scaler : StandardScaler
        Fitted StandardScaler.
    features_df : pd.DataFrame
        Patient features DataFrame containing ``FEATURE_COLS``.

    Returns
    -------
    pd.DataFrame
        Copy of *features_df* with ``anomaly_score``, ``risk_score``, and
        ``flagged`` columns appended.

    Raises
    ------
    KeyError
        If required feature columns are missing from *features_df*.
    """
    missing = set(FEATURE_COLS) - set(features_df.columns)
    if missing:
        raise KeyError(
            f"features_df is missing required feature column(s): {sorted(missing)}"
        )

    scored_df = features_df.copy()

    if scored_df.empty:
        scored_df["anomaly_score"] = pd.Series(dtype=float)
        scored_df["risk_score"] = pd.Series(dtype=float)
        scored_df["flagged"] = pd.Series(dtype=bool)
        return scored_df

    X = scored_df[FEATURE_COLS].copy()
    X_scaled = scaler.transform(X)

    # IsolationForest.score_samples returns negative scores where lower = more anomalous.
    # Inverting (-score_samples) makes higher values represent higher anomaly severity.
    raw_anomaly_scores = -model.score_samples(X_scaled)
    predictions = model.predict(X_scaled)  # -1 for anomaly, 1 for inlier

    scored_df["anomaly_score"] = raw_anomaly_scores
    scored_df["flagged"] = predictions == -1

    # Min-max normalization to [0, 100] scale
    min_score = np.min(raw_anomaly_scores)
    max_score = np.max(raw_anomaly_scores)

    if max_score > min_score:
        risk = 100.0 * (raw_anomaly_scores - min_score) / (max_score - min_score)
    else:
        risk = np.zeros_like(raw_anomaly_scores, dtype=float)

    scored_df["risk_score"] = np.round(risk, 2)

    return scored_df
