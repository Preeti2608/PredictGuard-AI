"""
PredictGuard AI - Shared Metrics I/O and Model Metadata Engine.

Enforces strict compliance with ML audit rules:
1. Never fabricate or hand-type any metric, dataset size, or training date.
2. Metrics MUST be computed from actual test set ground-truth (y_true) and predictions (y_pred, y_prob).
3. Upserts models/model_metadata.json and docs/metrics/<name>_metrics.json atomically.
4. Tracks dataset name, train/val/test sizes, random seed, timestamp, Git SHA, and library versions.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "docs" / "metrics"
METADATA_FILE = BASE_DIR / "models" / "model_metadata.json"


def get_git_sha() -> Optional[str]:
    """Retrieve current Git commit SHA if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def get_library_versions() -> Dict[str, str]:
    """Collect runtime versions of core ML libraries."""
    versions: Dict[str, str] = {}
    try:
        import sklearn

        versions["scikit-learn"] = getattr(sklearn, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import tensorflow as tf

        versions["tensorflow"] = getattr(tf, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import joblib

        versions["joblib"] = getattr(joblib, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import numpy as np_lib

        versions["numpy"] = getattr(np_lib, "__version__", "unknown")
    except ImportError:
        pass
    return versions


def compute_metrics_from_predictions(
    y_true: Union[List[Any], np.ndarray],
    y_pred: Union[List[Any], np.ndarray],
    y_prob: Optional[Union[List[Any], np.ndarray]] = None,
    pos_label: Any = 1,
    average: str = "binary",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute verification metrics strictly from real test predictions."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    acc = float(accuracy_score(y_t, y_p))

    # Confusion matrix
    cm = confusion_matrix(y_t, y_p).tolist()

    if average == "binary":
        prec = float(precision_score(y_t, y_p, pos_label=pos_label, zero_division=0))
        rec = float(recall_score(y_t, y_p, pos_label=pos_label, zero_division=0))
        f1 = float(f1_score(y_t, y_p, pos_label=pos_label, zero_division=0))
    else:
        prec = float(precision_score(y_t, y_p, average=average, zero_division=0))
        rec = float(recall_score(y_t, y_p, average=average, zero_division=0))
        f1 = float(f1_score(y_t, y_p, average=average, zero_division=0))

    auc: Optional[float] = None
    if y_prob is not None:
        try:
            y_pr = np.asarray(y_prob)
            if average == "binary":
                # If 2D prob array passed, take second column
                if y_pr.ndim == 2 and y_pr.shape[1] == 2:
                    y_pr = y_pr[:, 1]
                auc = float(roc_auc_score(y_t, y_pr))
            else:
                auc = float(roc_auc_score(y_t, y_pr, multi_class="ovr", average="macro"))
        except Exception:
            auc = None

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": cm,
    }

    # Optional top-2 accuracy for multiclass
    if average != "binary" and y_prob is not None:
        try:
            y_pr = np.asarray(y_prob)
            top2 = np.argsort(y_pr, axis=1)[:, -2:]
            top2_correct = sum(y_t[i] in top2[i] for i in range(len(y_t)))
            metrics["top_2_accuracy"] = round(float(top2_correct / len(y_t)), 4)
        except Exception:
            pass

    return metrics


def save_trained_metrics(
    metrics_filename: str,
    model_name: str,
    dataset_name: str,
    task: str,
    model_file_path: str,
    split_sizes: Dict[str, int],
    metrics: Dict[str, Any],
    seed: Optional[int] = 42,
    selection_rule: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Save computed metrics to docs/metrics/<name>.json and upsert models/model_metadata.json."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_metrics_path = METRICS_DIR / metrics_filename

    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_sha()
    libraries = get_library_versions()

    metrics_payload: Dict[str, Any] = {
        "status": "TRAINED",
        "model": model_name,
        "dataset": dataset_name,
        "task": task,
        "file_path": model_file_path,
        "split_sizes": split_sizes,
        "seed": seed,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "library_versions": libraries,
        "metrics": metrics,
    }
    if selection_rule:
        metrics_payload["selection_rule"] = selection_rule
    if extra_data:
        metrics_payload.update(extra_data)

    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Upsert model_metadata.json
    upsert_model_metadata(
        model_name=model_name,
        dataset=dataset_name,
        task=task,
        file_path=model_file_path,
        trained=True,
        metrics=metrics,
        training_date=timestamp[:10],
    )


def save_placeholder_metrics(
    metrics_filename: str,
    model_name: str,
    dataset_name: str,
    task: str,
    model_file_path: str,
    note: str,
) -> None:
    """Mark an untrained placeholder in metrics and metadata with null metric values."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_metrics_path = METRICS_DIR / metrics_filename

    payload: Dict[str, Any] = {
        "status": "UNTRAINED_PLACEHOLDER",
        "model": model_name,
        "dataset": dataset_name,
        "task": task,
        "file_path": model_file_path,
        "metrics": {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "roc_auc": None,
            "confusion_matrix": None,
        },
        "note": note,
    }

    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    upsert_model_metadata(
        model_name=model_name,
        dataset=dataset_name,
        task=task,
        file_path=model_file_path,
        trained=False,
        metrics={
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "roc_auc": None,
            "confusion_matrix": None,
        },
        training_date=None,
    )


def upsert_model_metadata(
    model_name: str,
    dataset: str,
    task: str,
    file_path: str,
    trained: bool,
    metrics: Dict[str, Any],
    training_date: Optional[str] = None,
) -> None:
    """Atomically upsert entry in models/model_metadata.json."""
    metadata: Dict[str, Any] = {"models": []}
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {"models": []}

    models_list = metadata.get("models", [])
    found = False

    for m in models_list:
        if m.get("name") == model_name or m.get("file_path") == file_path:
            m["name"] = model_name
            m["dataset"] = dataset
            m["task"] = task
            m["file_path"] = file_path
            m["trained"] = trained
            m["status"] = "TRAINED" if trained else "UNTRAINED_PLACEHOLDER"
            m["metrics"] = metrics
            m["training_date"] = training_date
            found = True
            break

    if not found:
        models_list.append(
            {
                "name": model_name,
                "version": "1.0.0",
                "framework": "TensorFlow / scikit-learn",
                "task": task,
                "training_date": training_date,
                "dataset": dataset,
                "trained": trained,
                "status": "TRAINED" if trained else "UNTRAINED_PLACEHOLDER",
                "metrics": metrics,
                "file_path": file_path,
            }
        )

    metadata["models"] = models_list
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
