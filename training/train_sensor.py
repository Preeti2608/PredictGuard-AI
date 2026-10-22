"""
Machine Telemetry Failure Prediction - Model Comparison, Calibration & Training Script.
Compares 7 algorithms (Logistic Regression, Decision Tree, SVM, KNN, Naive Bayes, Random Forest, Gradient Boosting)
using stratified train/test split. Primary optimization metric is Recall on the Failure class (class 1).
Calibrates the winning classifier via CalibratedClassifierCV and writes full benchmark metrics to docs/metrics/.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMN = "Machine failure"


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Compare Machine Failure Classifiers")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(BASE_DIR / "data" / "ai4i2020.csv"),
        help="Path to AI4I 2020 predictive maintenance CSV",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(BASE_DIR / "models" / "sensor_failure_calibrated.joblib"),
        help="Path to save calibrated winning model",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=str(BASE_DIR / "docs" / "metrics" / "sensor_metrics.json"),
        help="Path to save benchmark comparison metrics JSON",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Reproducibility random seed")
    return parser.parse_args()


def build_preprocessor():
    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    categorical_features = ["Type"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
        ]
    )


def main():
    args = parse_args()
    data_path = Path(args.data_path)

    print("=" * 60)
    print("MACHINE TELEMETRY FAILURE PREDICTOR - MULTI-MODEL BENCHMARK")
    print("=" * 60)

    if not data_path.exists():
        print(f"\n[TODO - DATA NEEDED] Dataset not found at: {data_path}")
        print("Please provide the AI4I 2020 Predictive Maintenance Dataset CSV at data/ai4i2020.csv.")
        print("Dataset source: UCI Machine Learning Repository / Kaggle AI4I 2020.")

        metrics_file = Path(args.output_metrics)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        if not metrics_file.exists():
            todo_metrics = {
                "status": "PENDING_DATASET",
                "message": "TODO: Provide data/ai4i2020.csv and execute training/train_sensor.py",
                "primary_metric": "Recall (Failure Class 1)",
                "models_evaluated": [
                    "Logistic Regression",
                    "Decision Tree",
                    "Support Vector Machine (SVC)",
                    "K-Nearest Neighbors",
                    "Gaussian Naive Bayes",
                    "Random Forest",
                    "Gradient Boosting",
                ],
                "leaderboard": [],
                "winner": None,
                "calibrated_metrics": None,
            }
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(todo_metrics, f, indent=2)
            print(f"Recorded TODO marker in {metrics_file}")
        sys.exit(0)

    print(f"Loading telemetry data from {data_path}...")
    df = pd.read_csv(data_path)

    # Validate required columns
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        print(f"Error: Dataset missing columns: {missing}")
        sys.exit(1)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].values

    print(f"Dataset shape: {X.shape}. Target distribution: {np.bincount(y)}")

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.random_state, stratify=y
    )

    models_to_evaluate = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=args.random_state, max_iter=1000
        ),
        "Decision Tree": RandomForestClassifier(
            n_estimators=1, max_depth=10, class_weight="balanced", random_state=args.random_state
        ),
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=args.random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=args.random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=args.random_state),
    }

    leaderboard = []

    print("\nRunning multi-model stratified evaluation...")
    for model_name, clf in models_to_evaluate.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        acc = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None

        entry = {
            "model": model_name,
            "failure_recall": round(rec, 4),
            "failure_precision": round(prec, 4),
            "failure_f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4) if auc is not None else None,
            "pipeline": pipe,
        }
        leaderboard.append(entry)
        print(
            f"  {model_name:<24} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f} | AUC: {auc if auc else 0:.4f}"
        )

    # Primary metric: Recall on failure class (class 1)
    leaderboard.sort(key=lambda x: (x["failure_recall"], x["failure_f1"]), reverse=True)
    winner_entry = leaderboard[0]
    winner_name = winner_entry["model"]
    print(f"\nWinning Model by Failure Class Recall: {winner_name} (Recall={winner_entry['failure_recall']})")

    # Calibrate winner using CalibratedClassifierCV
    print(f"Calibrating {winner_name} using CalibratedClassifierCV (method='sigmoid', 5-fold CV)...")
    raw_winner_pipeline = winner_entry["pipeline"]

    calibrated_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=raw_winner_pipeline.named_steps["classifier"],
                    method="sigmoid",
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state),
                ),
            ),
        ]
    )
    calibrated_pipeline.fit(X_train, y_train)

    y_cal_pred = calibrated_pipeline.predict(X_test)
    y_cal_prob = calibrated_pipeline.predict_proba(X_test)[:, 1]

    cal_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_cal_pred)), 4),
        "failure_recall": round(float(recall_score(y_test, y_cal_pred, pos_label=1)), 4),
        "failure_precision": round(float(precision_score(y_test, y_cal_pred, pos_label=1)), 4),
        "failure_f1": round(float(f1_score(y_test, y_cal_pred, pos_label=1)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_cal_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_cal_pred).tolist(),
    }

    # Save calibrated model
    out_model_path = Path(args.output_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_pipeline, out_model_path)
    print(f"Calibrated winning model saved to {out_model_path}")

    # Write metrics JSON
    metrics_summary = {
        "status": "COMPLETED",
        "dataset": "AI4I 2020 Predictive Maintenance",
        "primary_metric": "Recall on Machine Failure (Class 1)",
        "winner": winner_name,
        "calibrated_model_file": str(out_model_path.name),
        "calibrated_test_metrics": cal_metrics,
        "leaderboard": [{k: v for k, v in item.items() if k != "pipeline"} for item in leaderboard],
    }

    out_metrics_path = Path(args.output_metrics)
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Metrics written to {out_metrics_path}")


if __name__ == "__main__":
    main()
