"""
scripts/run_validation.py
==========================
Runs synthetic rationing validation & robustness experiments:
1. Benchmark IsolationForest vs Supervised Classifiers (metrics.csv).
2. Robustness sweep over multiplier_step, injection_fraction, Gaussian noise,
   confounder hospitalizations, and leakage checks across 20 random seeds (robustness.csv).
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cohort  # noqa: E402
from src.feature_engineering import build_patient_features  # noqa: E402
from src.model import FEATURE_COLS, score_patients, train_anomaly_model  # noqa: E402
from src.synthetic import generate_synthetic_cohort  # noqa: E402
from src.validation import (  # noqa: E402
    apply_confounder_gaps,
    apply_gap_noise,
    inject_synthetic_rationing,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_supervised_baselines(
    features_df: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[dict]:
    """Evaluate supervised classifiers using Stratified N-Fold CV on features_df."""
    X = features_df[FEATURE_COLS].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
    }

    results = []
    for name, clf_cls in classifiers.items():
        auc_scores = []
        prec_scores = []
        rec_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            clf = clf_cls
            clf.fit(X_train_scaled, y_train)

            y_proba = clf.predict_proba(X_val_scaled)[:, 1]
            y_pred = clf.predict(X_val_scaled)

            auc_scores.append(roc_auc_score(y_val, y_proba))
            prec_scores.append(precision_score(y_val, y_pred, zero_division=0))
            rec_scores.append(recall_score(y_val, y_pred, zero_division=0))

        results.append(
            {
                "model": f"{name} (5-fold CV)",
                "contamination": "N/A",
                "roc_auc": round(float(np.mean(auc_scores)), 4),
                "precision": round(float(np.mean(prec_scores)), 4),
                "recall": round(float(np.mean(rec_scores)), 4),
            }
        )

    return results


def run_robustness_experiment(
    multiplier_steps: list[float] = [0.05, 0.10, 0.20, 0.40],
    injection_fractions: list[float] = [0.05, 0.10, 0.15],
    n_seeds: int = 20,
    n_patients: int = 150,
) -> pd.DataFrame:
    """Run robustness sweep experiment over multiplier_step, injection_fraction, and 20 random seeds."""
    logger.info(
        "Starting robustness experiment sweep (%d configs x %d seeds)...",
        len(multiplier_steps) * len(injection_fractions),
        n_seeds,
    )

    rows = []

    for mult_step in multiplier_steps:
        for inj_frac in injection_fractions:
            iforest_aucs = []
            logreg_aucs = []
            conf_flagged_pcts = []
            auc_drops = []

            for seed_idx in range(n_seeds):
                seed = 42 + seed_idx
                df_base = generate_synthetic_cohort(n_patients=n_patients, random_state=seed)

                # 1. Synthetic rationing injection
                df_inj, inj_pids = inject_synthetic_rationing(
                    df_base, fraction=inj_frac, random_state=seed, multiplier_step=mult_step
                )

                # 2. Confounder group (single long gap for 10% of control patients)
                df_conf, conf_pids = apply_confounder_gaps(
                    df_inj, injected_pids=inj_pids, fraction=0.10, random_state=seed
                )

                # 3. Add Gaussian noise (sigma = 10% of patient mean gap)
                df_noisy = apply_gap_noise(df_conf, noise_ratio=0.10, random_state=seed)

                # 4. Extract features
                feats_df = build_patient_features(df_noisy, min_fills=4)

                if feats_df.empty or feats_df["DESYNPUF_ID"].nunique() < 5:
                    continue

                feats_df["is_rationer"] = feats_df["DESYNPUF_ID"].isin(inj_pids)
                feats_df["is_confounder"] = feats_df["DESYNPUF_ID"].isin(conf_pids)
                y = feats_df["is_rationer"].astype(int)

                if y.nunique() < 2:
                    continue

                # 5. Isolation Forest
                model_if, scaler_if = train_anomaly_model(
                    feats_df, contamination=inj_frac, save_dir=None, random_state=seed
                )
                scored_if = score_patients(model_if, scaler_if, feats_df)

                auc_if = roc_auc_score(y, scored_if["anomaly_score"])
                iforest_aucs.append(auc_if)

                # Confounder false positive check
                conf_df = scored_if[scored_if["is_confounder"]]
                if not conf_df.empty:
                    pct = float(conf_df["flagged"].sum() / len(conf_df) * 100.0)
                    conf_flagged_pcts.append(pct)

                # 6. Logistic Regression Full vs Reduced (Leakage Check)
                X_full = feats_df[FEATURE_COLS].values
                X_red = feats_df[["pay_amt_trend", "std_gap_ratio"]].values

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                auc_full_folds = []
                auc_red_folds = []

                for train_idx, val_idx in skf.split(X_full, y):
                    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
                    if y_tr.nunique() < 2 or y_va.nunique() < 2:
                        continue

                    # Full model
                    s_f = StandardScaler()
                    X_tr_f = s_f.fit_transform(X_full[train_idx])
                    X_va_f = s_f.transform(X_full[val_idx])

                    clf_f = LogisticRegression(random_state=seed)
                    clf_f.fit(X_tr_f, y_tr)
                    p_f = clf_f.predict_proba(X_va_f)[:, 1]
                    auc_full_folds.append(roc_auc_score(y_va, p_f))

                    # Reduced model (leakage check)
                    s_r = StandardScaler()
                    X_tr_r = s_r.fit_transform(X_red[train_idx])
                    X_va_r = s_r.transform(X_red[val_idx])

                    clf_r = LogisticRegression(random_state=seed)
                    clf_r.fit(X_tr_r, y_tr)
                    p_r = clf_r.predict_proba(X_va_r)[:, 1]
                    auc_red_folds.append(roc_auc_score(y_va, p_r))

                if auc_full_folds and auc_red_folds:
                    mean_full = np.mean(auc_full_folds)
                    mean_red = np.mean(auc_red_folds)
                    logreg_aucs.append(mean_full)
                    auc_drops.append(mean_full - mean_red)

            rows.append(
                {
                    "multiplier_step": mult_step,
                    "injection_fraction": inj_frac,
                    "iforest_auc_mean": round(float(np.mean(iforest_aucs)), 4),
                    "iforest_auc_std": round(float(np.std(iforest_aucs)), 4),
                    "logreg_auc_mean": round(float(np.mean(logreg_aucs)), 4),
                    "logreg_auc_std": round(float(np.std(logreg_aucs)), 4),
                    "confounder_flagged_pct": round(float(np.mean(conf_flagged_pcts)), 2) if conf_flagged_pcts else 0.0,
                    "leakage_auc_drop": round(float(np.mean(auc_drops)), 4) if auc_drops else 0.0,
                }
            )

    robustness_df = pd.DataFrame(rows)
    return robustness_df


def main():
    print("=" * 70)
    print("  INSULIN RATIONING DETECTION – SYNTHETIC VALIDATION EXPERIMENT")
    print("=" * 70)

    # 1. Load or generate cohort data
    df_cohort = None
    try:
        logger.info("Attempting to load cohort from data/raw ...")
        df_cohort = load_cohort()
        logger.info("Successfully loaded %d cohort rows from raw data.", len(df_cohort))

        # Check if enough patients have >= 4 fills in raw slice
        feats_test = build_patient_features(df_cohort, min_fills=4)
        if len(feats_test) < 10:
            logger.warning(
                "Raw sample slice has only %d beneficiaries with >= 4 fills. Generating synthetic cohort...",
                len(feats_test),
            )
            df_cohort = generate_synthetic_cohort(n_patients=150)
    except (FileNotFoundError, KeyError) as exc:
        logger.warning(
            "Raw CMS data unavailable (%s). Generating synthetic baseline cohort ...",
            exc,
        )
        df_cohort = generate_synthetic_cohort(n_patients=150)

    # 2. Inject synthetic rationing into 15% of beneficiaries
    injection_fraction = 0.15
    df_injected, injected_pids = inject_synthetic_rationing(
        df_cohort, fraction=injection_fraction, random_state=42, multiplier_step=0.40
    )
    injected_set = set(injected_pids)

    # 3. Extract patient features
    logger.info("Extracting patient-level features...")
    features_df = build_patient_features(df_injected, min_fills=4)

    if features_df.empty:
        logger.warning("features_df is empty. Using synthetic cohort fallback.")
        df_cohort = generate_synthetic_cohort(n_patients=150)
        df_injected, injected_pids = inject_synthetic_rationing(
            df_cohort, fraction=injection_fraction, random_state=42, multiplier_step=0.40
        )
        injected_set = set(injected_pids)
        features_df = build_patient_features(df_injected, min_fills=4)

    # Label rows with synthetic rationing status
    features_df["is_synthetic_rationer"] = features_df["DESYNPUF_ID"].isin(injected_set)
    y_true = features_df["is_synthetic_rationer"].astype(int)

    results = []

    # 4. Evaluate IsolationForest at contamination levels 5%, 10%, 20%
    logger.info("Evaluating IsolationForest at contamination levels 5%%, 10%%, 20%%...")
    contamination_levels = [0.05, 0.10, 0.20]
    scored_df_10 = None

    for cont in contamination_levels:
        save_dir = PROJECT_ROOT / "models" / "validation" if cont == 0.10 else None
        model, scaler = train_anomaly_model(
            features_df, contamination=cont, save_dir=save_dir, random_state=42
        )
        scored_df = score_patients(model, scaler, features_df)
        if cont == 0.10:
            scored_df_10 = scored_df

        auc_val = roc_auc_score(y_true, scored_df["anomaly_score"])
        prec_val = precision_score(y_true, scored_df["flagged"], zero_division=0)
        rec_val = recall_score(y_true, scored_df["flagged"], zero_division=0)

        results.append(
            {
                "model": f"IsolationForest (cont={int(cont*100)}%)",
                "contamination": f"{cont:.2f}",
                "roc_auc": round(float(auc_val), 4),
                "precision": round(float(prec_val), 4),
                "recall": round(float(rec_val), 4),
            }
        )

    # 5. Evaluate Supervised Baselines using Stratified 5-Fold CV
    logger.info("Evaluating Supervised Baselines (Logistic Regression, Random Forest, Decision Tree) with 5-Fold CV...")
    supervised_results = evaluate_supervised_baselines(features_df, y_true, n_splits=5, random_state=42)
    results.extend(supervised_results)

    # 6. Save metrics to report/metrics.csv
    report_dir = PROJECT_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = report_dir / "metrics.csv"
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info("Saved validation metrics to %s", metrics_csv_path)

    # Print summary metrics table
    print("\n" + "=" * 70)
    print("                    VALIDATION METRICS COMPARISON")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("=" * 70 + "\n")

    # 7. Robustness Experiment Sweep (20 Seeds x 12 Configs)
    robustness_df = run_robustness_experiment()
    robustness_csv_path = report_dir / "robustness.csv"
    robustness_df.to_csv(robustness_csv_path, index=False)
    logger.info("Saved robustness experiment results to %s", robustness_csv_path)

    print("-" * 70)
    print("                    ROBUSTNESS EXPERIMENT RESULTS")
    print("-" * 70)
    print(robustness_df.to_string(index=False))
    print("-" * 70 + "\n")

    # 8. Detailed breakdown for default 10% IsolationForest
    rationers_df = scored_df_10[scored_df_10["is_synthetic_rationer"]]
    rest_df = scored_df_10[~scored_df_10["is_synthetic_rationer"]]

    total_rationers = len(rationers_df)
    flagged_rationers = int(rationers_df["flagged"].sum())
    rate_rationers = (flagged_rationers / total_rationers * 100.0) if total_rationers > 0 else 0.0

    total_rest = len(rest_df)
    flagged_rest = int(rest_df["flagged"].sum())
    rate_rest = (flagged_rest / total_rest * 100.0) if total_rest > 0 else 0.0

    mean_risk_rationers = float(rationers_df["risk_score"].mean()) if total_rationers > 0 else 0.0
    mean_risk_rest = float(rest_df["risk_score"].mean()) if total_rest > 0 else 0.0

    # 9. Generate bar chart visualization
    chart_path = report_dir / "validation_chart.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Synthetic Validation: Insulin Rationing Detection Performance",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    categories = ["Injected Rationers", "Control Group"]
    rates = [rate_rationers, rate_rest]
    colors = ["#E63946", "#457B9D"]

    bars1 = ax1.bar(categories, rates, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Flagged Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Model Flagged Rate (%) by Group (10% Contamination)", fontsize=11)
    ax1.set_ylim(0, max(100, max(rates) * 1.15))
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    mean_scores = [mean_risk_rationers, mean_risk_rest]
    bars2 = ax2.bar(categories, mean_scores, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel("Mean Risk Score (0-100)", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Composite Risk Score (0-100)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(chart_path, dpi=300)
    plt.close()

    logger.info("Saved validation chart to %s", chart_path)
    print(f"Validation chart saved successfully to: {chart_path}\n")


if __name__ == "__main__":
    main()
