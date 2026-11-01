"""
PredictGuard AI - Casting Defect Classifier Real Evaluation Engine.

Loads models/casting_defect_classifier.h5 and computes held-out test metrics on data/casting_data/test/
using the exact canonical preprocessor (app.services.image_service.preprocess_image).

Generates:
1. docs/metrics/image_metrics.json (via training.metrics_io)
2. docs/figures/casting_confusion_matrix.png
3. docs/figures/casting_roc_curve.png
4. Upserts models/model_metadata.json with verified test metrics.
"""

import argparse
import sys
from pathlib import Path
from typing import List

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.services.image_service import preprocess_image
from training.metrics_io import compute_metrics_from_predictions, save_trained_metrics

MODEL_PATH = BASE_DIR / "models" / "casting_defect_classifier.h5"
DATA_DIR = BASE_DIR / "data" / "casting_data"
TEST_DIR = DATA_DIR / "test"
FIGURES_DIR = BASE_DIR / "docs" / "figures"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Casting Quality Classifier on Held-Out Test Data")
    parser.add_argument("--test-dir", type=str, default=str(TEST_DIR), help="Path to held-out test directory")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH), help="Path to trained Keras/H5 model")
    parser.add_argument("--interpolation", type=str, default="nearest", choices=["nearest", "bicubic", "bilinear"])
    return parser.parse_args()


def check_dataset_exists(test_dir: Path) -> bool:
    if not test_dir.exists():
        return False
    def_dir = test_dir / "def_front"
    ok_dir = test_dir / "ok_front"
    if not def_dir.exists() or not ok_dir.exists():
        return False
    return True


def print_blocked_instructions():
    print("\n" + "=" * 78)
    print("STATUS: BLOCKED - MISSING HELD-OUT TEST DATASET")
    print("=" * 78)
    print("The Casting Quality test images are not present at: data/casting_data/test/")
    print("\nPlease download the dataset archive from Kaggle:")
    print("  Kaggle Dataset Slug: ravirajsinh45/real-life-industrial-dataset-of-casting-product")
    print("  URL: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product")
    print("\nExtract the archive so that the folder layout matches:")
    print("  data/casting_data/")
    print("  |-- train/")
    print("  |   |-- def_front/    (Defective casting images)")
    print("  |   +-- ok_front/     (Normal defect-free casting images)")
    print("  +-- test/")
    print("      |-- def_front/    (Held-out defective test images)")
    print("      +-- ok_front/     (Held-out normal test images)")
    print("\nOnce placed, re-run:")
    print("  python training/evaluate_image.py")
    print("=" * 78 + "\n")


def evaluate_casting_model(model_path: Path, test_dir: Path, interpolation: str) -> None:
    print(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    def_images = list((test_dir / "def_front").glob("*.*"))
    ok_images = list((test_dir / "ok_front").glob("*.*"))
    # Filter to image extensions
    valid_exts = {".jpg", ".jpeg", ".png"}
    def_images = [p for p in def_images if p.suffix.lower() in valid_exts]
    ok_images = [p for p in ok_images if p.suffix.lower() in valid_exts]

    total_count = len(def_images) + len(ok_images)
    print(f"Found {len(def_images)} defective images and {len(ok_images)} normal images (Total: {total_count})")

    # Mapping: def_front=0 (Defect), ok_front=1 (Normal)
    # The sigmoid output represents P(ok_front). Defect probability is 1.0 - score.
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob_defective: List[float] = []

    print("Executing batch inference using production preprocessor...")
    # Evaluate def_front (true label = 1 for defect)
    for p in def_images:
        with open(p, "rb") as f:
            content = f.read()
        tensor = preprocess_image(content, target_size=(224, 224), interpolation=interpolation)
        raw_score = float(model.predict(tensor, verbose=0)[0][0])
        defect_prob = 1.0 - raw_score
        y_true.append(1)  # Defect
        y_prob_defective.append(defect_prob)
        y_pred.append(1 if defect_prob >= 0.5 else 0)

    # Evaluate ok_front (true label = 0 for defect)
    for p in ok_images:
        with open(p, "rb") as f:
            content = f.read()
        tensor = preprocess_image(content, target_size=(224, 224), interpolation=interpolation)
        raw_score = float(model.predict(tensor, verbose=0)[0][0])
        defect_prob = 1.0 - raw_score
        y_true.append(0)  # Normal
        y_prob_defective.append(defect_prob)
        y_pred.append(1 if defect_prob >= 0.5 else 0)

    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    y_probs = np.array(y_prob_defective)

    metrics = compute_metrics_from_predictions(
        y_true=y_t,
        y_pred=y_p,
        y_prob=y_probs,
        pos_label=1,  # Defect is positive class
        average="binary",
        class_names=["ok_front", "def_front"],
    )

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET EVALUATION METRICS (DEFECTIVE CLASS = 1)")
    print("=" * 60)
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Defect Precision: {metrics['precision']:.4f}")
    print(f"  Defect Recall:    {metrics['recall']:.4f}")
    print(f"  Defect F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:          {metrics['roc_auc']:.4f if metrics['roc_auc'] else 'N/A'}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    print("=" * 60)

    # Generate and save figures if matplotlib is installed
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if HAS_MATPLOTLIB:
        # 1. Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=np.array(metrics["confusion_matrix"]),
            display_labels=["Normal (ok)", "Defective (def)"],
        )
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        ax.set_title("Casting Defect Classifier - Confusion Matrix")
        cm_path = FIGURES_DIR / "casting_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"Saved confusion matrix figure to: {cm_path}")

        # 2. ROC Curve
        if metrics["roc_auc"] is not None:
            fpr, tpr, _ = roc_curve(y_t, y_probs)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {metrics['roc_auc']:.4f})")
            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate (Normal flagged as Defect)")
            ax.set_ylabel("True Positive Rate (Defect Recall)")
            ax.set_title("Casting Defect Classifier - ROC Curve")
            ax.legend(loc="lower right")
            roc_path = FIGURES_DIR / "casting_roc_curve.png"
            plt.tight_layout()
            plt.savefig(roc_path, dpi=200)
            plt.close()
            print(f"Saved ROC curve figure to: {roc_path}")
    else:
        print("Note: matplotlib not installed; skipping confusion matrix and ROC curve plot generation.")


    # Save metrics via shared metrics_io engine
    save_trained_metrics(
        metrics_filename="image_metrics.json",
        model_name="Casting Defect Classifier",
        dataset_name="Casting Product Image Quality Dataset (ravirajsinh45/real-life-industrial-dataset-of-casting-product)",
        task="Binary Defect Inspection (ok_front vs def_front)",
        model_file_path=str(model_path.relative_to(BASE_DIR)),
        split_sizes={"test_defective": len(def_images), "test_normal": len(ok_images), "total_test": total_count},
        metrics=metrics,
        selection_rule="Evaluation on held-out test split using production nearest interpolation",
    )
    print("Successfully recorded verified test metrics to docs/metrics/image_metrics.json and models/model_metadata.json.")


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    model_path = Path(args.model_path)

    if not check_dataset_exists(test_dir):
        print_blocked_instructions()
        sys.exit(2)

    evaluate_casting_model(model_path, test_dir, args.interpolation)


if __name__ == "__main__":
    main()
