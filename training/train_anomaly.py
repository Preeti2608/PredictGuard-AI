"""
PredictGuard AI - MVTec Anomaly Detection (AD) Training Engine.

Trains unsupervised anomaly detectors on MobileNetV2 dense image embeddings:
- Categories supported: metal_nut, screw, transistor.
- Unsupervised fit on defect-free nominal images (train/good).
- Model architectures:
  1. Isolation Forest (Ensemble of isolation trees identifying anomalous densities)
  2. kNN Distance Scorer (k=5 nearest-neighbor feature distance metric)
- Held-out evaluation: image-level ROC-AUC across good and anomalous test samples.

License notice: CC BY-NC-SA 4.0 (Non-Commercial Academic & Research Use Only).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.metrics_io import get_git_sha, get_library_versions

DATA_DIR = BASE_DIR / "data" / "mvtec_ad"
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = BASE_DIR / "docs" / "metrics" / "mvtec_metrics.json"

CATEGORIES = ["metal_nut", "screw", "transistor"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train MVTec AD Anomaly Detectors")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to MVTec AD root directory")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES, help="Categories to train")
    return parser.parse_args()


def print_blocked_instructions(data_dir: Path):
    print("\n" + "=" * 78)
    print("STATUS: BLOCKED - MISSING MVTEC AD DATASET (TERMS ACCEPTANCE REQUIRED)")
    print("=" * 78)
    print(f"The MVTec AD dataset directory was not found at: {data_dir}")
    print("\nMVTec AD is distributed under the Creative Commons CC BY-NC-SA 4.0 license")
    print("(Non-commercial research / academic use only).")
    print("Terms must be accepted by the user at:")
    print("  https://www.mvtec.com/research-teaching/datasets/mvtec-ad")
    print("\nPlease download 2-3 categories (e.g. metal_nut.tar.xz, screw.tar.xz) and extract:")
    print("  data/mvtec_ad/")
    print("  |-- metal_nut/")
    print("  |   |-- train/good/           (Nominal defect-free training images)")
    print("  |   +-- test/                 (Subdirectories: good, bent, scratch, etc.)")
    print("  |-- screw/")
    print("  |   |-- train/good/")
    print("  |   +-- test/")
    print("  +-- transistor/")
    print("      |-- train/good/")
    print("      +-- test/")
    print("\nOnce placed, re-run:")
    print("  python training/train_anomaly.py")
    print("=" * 78 + "\n")


def extract_mobilenet_embeddings(paths: List[str], batch_size: int = 32) -> np.ndarray:
    feature_extractor = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    def load_img(path_str):
        b = tf.io.read_file(path_str)
        img = tf.image.decode_image(b, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224], method="nearest")
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    embeddings = feature_extractor.predict(ds, verbose=0)
    return np.asarray(embeddings)


def train_category(category_dir: Path) -> Dict[str, float]:
    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"

    valid_exts = {".png", ".jpg", ".jpeg"}
    train_paths = [str(p) for p in train_good_dir.glob("*.*") if p.suffix.lower() in valid_exts]

    # Test paths and labels (0 = good, 1 = defect)
    test_paths: List[str] = []
    test_labels: List[int] = []

    for sub in test_dir.iterdir():
        if not sub.is_dir():
            continue
        is_defect = 0 if sub.name.lower() == "good" else 1
        sub_images = [str(p) for p in sub.glob("*.*") if p.suffix.lower() in valid_exts]
        test_paths.extend(sub_images)
        test_labels.extend([is_defect] * len(sub_images))

    print(f"[{category_dir.name}] Train Good: {len(train_paths)} | Test Total: {len(test_paths)} (Defects: {sum(test_labels)})")

    X_train = extract_mobilenet_embeddings(train_paths)
    X_test = extract_mobilenet_embeddings(test_paths)
    y_test = np.array(test_labels)

    # 1. Isolation Forest
    iso_forest = IsolationForest(n_estimators=150, contamination=0.1, random_state=42)
    iso_forest.fit(X_train)
    # Higher score = more anomalous (decision_function returns negative anomaly score)
    iso_scores = -iso_forest.decision_function(X_test)
    iso_auc = float(roc_auc_score(y_test, iso_scores))

    # 2. kNN Distance Scorer
    knn = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn.fit(X_train)
    distances, _ = knn.kneighbors(X_test)
    knn_scores = distances.mean(axis=1)
    knn_auc = float(roc_auc_score(y_test, knn_scores))

    # Save model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "category": category_dir.name,
        "isolation_forest": iso_forest,
        "knn_scorer": knn,
        "iso_auc": iso_auc,
        "knn_auc": knn_auc,
    }
    artifact_path = MODELS_DIR / f"mvtec_{category_dir.name}_anomaly.joblib"
    joblib.dump(model_artifact, artifact_path)
    print(f"[{category_dir.name}] IsolationForest ROC-AUC: {iso_auc:.4f} | kNN ROC-AUC: {knn_auc:.4f}")
    print(f"[{category_dir.name}] Saved artifact to: {artifact_path}")

    return {
        "isolation_forest_roc_auc": round(iso_auc, 4),
        "knn_roc_auc": round(knn_auc, 4),
        "train_samples": len(train_paths),
        "test_samples": len(test_paths),
        "test_defects": int(sum(test_labels)),
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("MVTEC AD ANOMALY DETECTION TRAINING ENGINE (CC BY-NC-SA 4.0)")
    print("=" * 70)

    if not data_dir.exists():
        print_blocked_instructions(data_dir)
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not METRICS_PATH.exists():
            with open(METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "status": "BLOCKED_PENDING_DATASET",
                        "license": "CC BY-NC-SA 4.0 (Non-commercial research only)",
                        "url": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
                        "message": "Terms must be accepted by user. Dataset not yet present in data/mvtec_ad/.",
                        "categories": {},
                    },
                    f,
                    indent=2,
                )
        sys.exit(2)

    categories_evaluated: Dict[str, Dict[str, float]] = {}
    found_any = False

    for cat in args.categories:
        cat_dir = data_dir / cat
        if cat_dir.exists() and (cat_dir / "train" / "good").exists():
            found_any = True
            categories_evaluated[cat] = train_category(cat_dir)
        else:
            print(f"[{cat}] Missing directory at {cat_dir}; skipping.")

    if not found_any:
        print_blocked_instructions(data_dir)
        sys.exit(2)

    payload = {
        "status": "COMPLETED",
        "license": "CC BY-NC-SA 4.0 (Non-Commercial Academic & Research Use Only)",
        "git_sha": get_git_sha(),
        "library_versions": get_library_versions(),
        "categories": categories_evaluated,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved MVTec metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
