"""
Tool Wear Degradation Forecasting - Regression Model Benchmark & Training.
Predicts continuous 'Tool wear [min]' using Linear Regression, Polynomial Regression,
and Multivariate Ridge / Random Forest Regressors on AI4I telemetry.
Reports MSE, RMSE, and R2 score; saves best regressor to models/tool_wear_regressor.joblib.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
]
TARGET_COLUMN = "Tool wear [min]"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tool Wear Degradation Regressor")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(BASE_DIR / "data" / "ai4i2020.csv"),
        help="Path to AI4I 2020 predictive maintenance CSV",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(BASE_DIR / "models" / "tool_wear_regressor.joblib"),
        help="Path to save best regression model",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=str(BASE_DIR / "docs" / "metrics" / "regression_metrics.json"),
        help="Path to save regression metrics JSON",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def build_preprocessor(poly_degree: int = 1):
    num_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
    ]
    cat_features = ["Type"]

    if poly_degree > 1:
        num_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ]
        )
    else:
        num_pipeline = Pipeline([("scaler", StandardScaler())])

    return ColumnTransformer(
        [
            ("num", num_pipeline, num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features),
        ]
    )


def main():
    args = parse_args()
    data_path = Path(args.data_path)

    print("=" * 60)
    print("TOOL WEAR DEGRADATION - REGRESSION BENCHMARK")
    print("=" * 60)

    if not data_path.exists():
        print(f"\n[TODO - DATA NEEDED] Dataset not found at: {data_path}")
        print("Please provide the AI4I 2020 dataset CSV at data/ai4i2020.csv.")

        metrics_file = Path(args.output_metrics)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        if not metrics_file.exists():
            todo_metrics = {
                "status": "PENDING_DATASET",
                "message": "TODO: Provide data/ai4i2020.csv and execute training/train_regression.py",
                "target": "Tool wear [min]",
                "models_evaluated": [
                    "Linear Regression",
                    "Polynomial Regression (Degree 2)",
                    "Multivariate Ridge Regressor",
                    "Random Forest Regressor",
                ],
                "results": [],
                "best_model": None,
                "best_metrics": None,
            }
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(todo_metrics, f, indent=2)
            print(f"Recorded TODO marker in {metrics_file}")
        sys.exit(0)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=args.random_state)

    models_dict = {
        "Linear Regression": Pipeline(
            [("preprocessor", build_preprocessor(poly_degree=1)), ("regressor", LinearRegression())]
        ),
        "Polynomial Regression (Degree 2)": Pipeline(
            [("preprocessor", build_preprocessor(poly_degree=2)), ("regressor", Ridge(alpha=1.0))]
        ),
        "Multivariate Ridge": Pipeline(
            [("preprocessor", build_preprocessor(poly_degree=1)), ("regressor", Ridge(alpha=10.0))]
        ),
        "Random Forest Regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor(poly_degree=1)),
                ("regressor", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=args.random_state)),
            ]
        ),
    }

    results = []
    print("\nEvaluating regression candidates...")
    for name, pipe in models_dict.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        entry = {
            "model": name,
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
            "pipeline": pipe,
        }
        results.append(entry)
        print(f"  {name:<32} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f}")

    # Best model by lowest RMSE
    results.sort(key=lambda x: x["rmse"])
    best = results[0]
    print(f"\nBest Model: {best['model']} (RMSE={best['rmse']}, R2={best['r2_score']})")

    # Save model artifact
    out_model_path = Path(args.output_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], out_model_path)
    print(f"Saved best regression model to {out_model_path}")

    # Save metrics JSON
    metrics_summary = {
        "status": "COMPLETED",
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "target": "Tool wear [min]",
        "best_model": best["model"],
        "best_metrics": {
            "mse": best["mse"],
            "rmse": best["rmse"],
            "mae": best["mae"],
            "r2_score": best["r2_score"],
        },
        "all_models": [{k: v for k, v in r.items() if k != "pipeline"} for r in results],
    }

    out_metrics_path = Path(args.output_metrics)
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Regression metrics saved to {out_metrics_path}")


if __name__ == "__main__":
    main()
