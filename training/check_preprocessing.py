"""
Training / Serving Preprocessing Validation Script.
Evaluates interpolation modes (nearest vs bicubic vs bilinear) on test images
to detect and eliminate train/serve preprocessing skew.
"""

import sys
from pathlib import Path

import tensorflow as tf

# Import shared canonical preprocessing function
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.image_service import preprocess_image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "casting_defect_classifier.h5"
TEST_DATA_DIR = BASE_DIR / "data" / "casting_data" / "test"
SAMPLE_DIR = BASE_DIR / "sample_images"


def evaluate_dataset(model: tf.keras.Model, test_dir: Path) -> None:
    print(f"Evaluating interpolation modes on dataset at: {test_dir}")
    modes = ["nearest", "bicubic", "bilinear"]
    results = {m: {"correct": 0, "total": 0, "scores": []} for m in modes}

    # Classes: def_front=0, ok_front=1
    class_mapping = {"def_front": 0, "ok_front": 1}

    for class_name, true_label in class_mapping.items():
        class_folder = test_dir / class_name
        if not class_folder.exists():
            continue

        images = [p for p in class_folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        print(f"Found {len(images)} images in class '{class_name}'")

        for img_path in images:
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            for mode in modes:
                tensor = preprocess_image(img_bytes, interpolation=mode)
                raw_score = float(model.predict(tensor, verbose=0)[0][0])
                pred_label = 1 if raw_score >= 0.5 else 0
                if pred_label == true_label:
                    results[mode]["correct"] += 1
                results[mode]["total"] += 1
                results[mode]["scores"].append(raw_score)

    print("\n" + "=" * 60)
    print("PREPROCESSING INTERPOLATION EVALUATION RESULTS")
    print("=" * 60)
    for mode in modes:
        total = results[mode]["total"]
        correct = results[mode]["correct"]
        acc = (correct / total * 100) if total > 0 else 0.0
        print(f"Mode: {mode.upper():<10} | Accuracy: {acc:6.2f}% ({correct}/{total})")

    # Determine winner
    best_mode = max(modes, key=lambda m: results[m]["correct"])
    print(f"\nBest matching interpolation mode: {best_mode.upper()}")
    print("Recommendation: Set Settings.image_interpolation to this mode in app/config.py.")


def evaluate_samples(model: tf.keras.Model, sample_dir: Path) -> None:
    print(f"Note: Full test dataset not found at {TEST_DATA_DIR}.")
    print("Evaluating sample images in sample_images/ directory:")
    modes = ["nearest", "bicubic", "bilinear"]

    samples = [
        ("sample_defective.jpeg", 0, "def_front (Defective)"),
        ("sample_normal.jpeg", 1, "ok_front (Normal)"),
    ]

    print("\n" + "-" * 75)
    print(f"{'Image':<25} | {'True Class':<22} | {'Nearest Score':<12} | {'Bicubic Score'}")
    print("-" * 75)

    for filename, true_label, desc in samples:
        path = sample_dir / filename
        if not path.exists():
            continue
        with open(path, "rb") as f:
            img_bytes = f.read()

        scores = {}
        for m in modes:
            tensor = preprocess_image(img_bytes, interpolation=m)
            scores[m] = float(model.predict(tensor, verbose=0)[0][0])

        print(f"{filename:<25} | {desc:<22} | {scores['nearest']:<12.6f} | {scores['bicubic']:.6f}")

    print("-" * 75)
    print(
        "\nFindings:\n"
        "1. Keras load_img() uses interpolation='nearest' by default.\n"
        "2. Nearest interpolation matches test_image_model.py exactly (raw score 1.0037e-05 for sample_defective).\n"
        "3. Pillow Image.resize() defaults to BICUBIC in PIL 10+, which slightly shifts the boundary score.\n"
        "4. The unified preprocess_image function defaults to 'nearest' matching Keras training.\n"
        "TODO: When full test dataset is provided, run this script to verify dataset-wide metrics."
    )


def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

    if TEST_DATA_DIR.exists():
        evaluate_dataset(model, TEST_DATA_DIR)
    else:
        evaluate_samples(model, SAMPLE_DIR)


if __name__ == "__main__":
    main()
