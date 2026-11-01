"""
PredictGuard AI - NEU 6-Class Surface Defect Training Engine.

Trains MobileNetV2 on hot-rolled steel strip surface defect images.
Classes (6 Defect Types):
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled_in_scale
  5: scratches

Strict ML Audit Compliance:
1. No synthetic or placeholder substitution.
2. If data/neu_data is missing: stops and marks task BLOCKED.
3. Saves models/neu_defect_classifier.keras ONLY upon successful training on real data.
4. Saves stratified split file lists to data/neu_splits.json for reproducibility.
5. Employs two-stage transfer learning (head training, then fine-tuning) with early stopping.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.metrics_io import compute_metrics_from_predictions, save_trained_metrics

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

DATA_DIR = BASE_DIR / "data" / "neu_data"
SPLITS_FILE = BASE_DIR / "data" / "neu_splits.json"
MODEL_PATH = BASE_DIR / "models" / "neu_defect_classifier.keras"
METRICS_PATH = BASE_DIR / "docs" / "metrics" / "neu_metrics.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on NEU Surface Defect Database")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to NEU dataset directory")
    parser.add_argument("--splits-file", type=str, default=str(SPLITS_FILE), help="Path to save train/val/test splits")
    parser.add_argument("--output-model", type=str, default=str(MODEL_PATH), help="Path to save trained Keras model")
    parser.add_argument("--output-metrics", type=str, default=str(METRICS_PATH), help="Path to save metrics JSON")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs-head", type=int, default=10, help="Epochs for classification head training")
    parser.add_argument("--epochs-finetune", type=int, default=15, help="Epochs for fine-tuning top layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def print_blocked_instructions(data_dir: Path):
    print("\n" + "=" * 78)
    print("STATUS: BLOCKED - MISSING NEU SURFACE DEFECT DATASET")
    print("=" * 78)
    print(f"The NEU Surface Defect Database was not found at: {data_dir}")
    print("\nPlease download the dataset archive from Kaggle:")
    print("  Kaggle Dataset Slug: kaustubhdikshit/neu-surface-defect-database")
    print("  URL: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
    print("\nExtract the archive so that class folders exist under data/neu_data/:")
    print("  data/neu_data/")
    print("  |-- crazing/           (300 bmp/jpg images)")
    print("  |-- inclusion/         (300 bmp/jpg images)")
    print("  |-- patches/           (300 bmp/jpg images)")
    print("  |-- pitted_surface/    (300 bmp/jpg images)")
    print("  |-- rolled_in_scale/   (300 bmp/jpg images)")
    print("  +-- scratches/         (300 bmp/jpg images)")
    print("\nOr with pre-split folders (train/val/test).")
    print("\nOnce placed, re-run:")
    print("  python training/train_neu.py")
    print("=" * 78 + "\n")


def scan_neu_images(data_dir: Path) -> Tuple[List[str], List[int]]:
    """Scan NEU data directory and map file paths to class indices."""
    image_paths: List[str] = []
    labels: List[int] = []
    valid_exts = {".bmp", ".jpg", ".jpeg", ".png"}

    for idx, cls_name in enumerate(NEU_CLASSES):
        # Look for folder named cls_name directly or in subdirectories
        candidates = list(data_dir.rglob(cls_name))
        found_images: List[Path] = []
        for cand in candidates:
            if cand.is_dir():
                found_images.extend([p for p in cand.glob("*.*") if p.suffix.lower() in valid_exts])

        if not found_images:
            # Fallback: check files with class name prefix (e.g., crazing_1.jpg)
            found_images = [
                p for p in data_dir.rglob("*.*")
                if p.suffix.lower() in valid_exts and cls_name in p.stem.lower()
            ]

        for p in found_images:
            image_paths.append(str(p.resolve()))
            labels.append(idx)

    return image_paths, labels


def create_or_load_splits(
    image_paths: List[str],
    labels: List[int],
    splits_path: Path,
    seed: int,
) -> Dict[str, List[str]]:
    """Create stratified 70/15/15 train/val/test split and persist to JSON."""
    if splits_path.exists():
        print(f"Loading existing split configuration from {splits_path}...")
        with open(splits_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Creating stratified 70/15/15 train/val/test split...")
    paths_arr = np.array(image_paths)
    labels_arr = np.array(labels)

    # First split: 70% train, 30% temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(paths_arr, labels_arr))

    # Second split: split 30% temp into 15% val and 15% test
    temp_paths = paths_arr[temp_idx]
    temp_labels = labels_arr[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_sub_idx, test_sub_idx = next(sss2.split(temp_paths, temp_labels))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    splits = {
        "train": paths_arr[train_idx].tolist(),
        "val": paths_arr[val_idx].tolist(),
        "test": paths_arr[test_idx].tolist(),
    }

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved stratified split paths to {splits_path}")
    return splits


def build_tf_dataset(
    file_paths: List[str],
    class_indices: Dict[str, int],
    batch_size: int,
    is_training: bool = False,
) -> tf.data.Dataset:
    """Load grayscale/RGB images, resize to 224x224, convert to 3 channels, batch."""
    paths: List[str] = []
    labels: List[int] = []

    for p_str in file_paths:
        p = Path(p_str)
        # Identify class index from path
        cls_idx = None
        for name, idx in class_indices.items():
            if name in str(p).lower():
                cls_idx = idx
                break
        if cls_idx is not None:
            paths.append(p_str)
            labels.append(cls_idx)

    def load_and_preprocess(path_tensor, label_tensor):
        img_bytes = tf.io.read_file(path_tensor)
        # Decode as 3-channel RGB (auto-expands single channel grayscale)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224], method="nearest")
        img = tf.cast(img, tf.float32) / 255.0
        return img, label_tensor

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("NEU STEEL STRIP 6-CLASS SURFACE DEFECT CLASSIFICATION")
    print("=" * 60)

    if not data_dir.exists():
        print_blocked_instructions(data_dir)
        sys.exit(2)

    image_paths, labels = scan_neu_images(data_dir)
    if len(image_paths) < 100:
        print(f"Error: Found only {len(image_paths)} images in {data_dir}. Expected ~1,800 images.")
        print_blocked_instructions(data_dir)
        sys.exit(2)

    print(f"Total verified NEU images found: {len(image_paths)}")
    class_map = {name: i for i, name in enumerate(NEU_CLASSES)}

    # Stratified Splits
    splits = create_or_load_splits(image_paths, labels, Path(args.splits_file), args.seed)
    print(f"Dataset split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    train_ds = build_tf_dataset(splits["train"], class_map, args.batch_size, is_training=True)
    val_ds = build_tf_dataset(splits["val"], class_map, args.batch_size, is_training=False)
    test_ds = build_tf_dataset(splits["test"], class_map, args.batch_size, is_training=False)

    # Build Two-Stage Model
    print("\nBuilding MobileNetV2 architecture with training augmentation...")
    data_aug = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.08),
        ],
        name="neu_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(NEU_CLASSES), activation="softmax", name="neu_probabilities")(x)

    model = models.Model(inputs, outputs, name="NEUSurfaceDefectClassifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Stage 1: Train Head
    print("\n--- STAGE 1: Training Classification Head ---")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
    )

    # Stage 2: Fine-Tuning top layers of MobileNetV2
    print("\n--- STAGE 2: Fine-Tuning MobileNetV2 Backbone ---")
    base_model.trainable = True
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        callbacks=callbacks,
    )

    # Evaluate on held-out test split
    print("\nEvaluating on held-out test split...")
    test_labels_list: List[int] = []
    test_preds_list: List[int] = []
    test_probs_list: List[np.ndarray] = []

    for images, batch_labels in test_ds:
        preds = model.predict(images, verbose=0)
        test_probs_list.append(preds)
        test_preds_list.extend(np.argmax(preds, axis=1).tolist())
        test_labels_list.extend(batch_labels.numpy().tolist())

    y_true_all = np.array(test_labels_list)
    y_pred_all = np.array(test_preds_list)
    y_prob_all = np.vstack(test_probs_list)

    metrics = compute_metrics_from_predictions(
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_prob=y_prob_all,
        average="macro",
        class_names=NEU_CLASSES,
    )

    report_dict = classification_report(y_true_all, y_pred_all, target_names=NEU_CLASSES, output_dict=True)

    print("\n" + "=" * 60)
    print("NEU HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Top-2 Accuracy:   {metrics.get('top_2_accuracy', 'N/A')}")
    print(f"  Macro Precision:  {metrics['precision']:.4f}")
    print(f"  Macro Recall:     {metrics['recall']:.4f}")
    print(f"  Macro F1-Score:   {metrics['f1_score']:.4f}")
    print("=" * 60)

    # Save trained model artifact
    out_model = Path(args.output_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_model))
    print(f"Saved verified trained model to: {out_model}")

    # Save metrics via shared metrics_io engine
    save_trained_metrics(
        metrics_filename="neu_metrics.json",
        model_name="NEU Surface Defect Classifier",
        dataset_name="NEU Surface Defect Database (kaustubhdikshit/neu-surface-defect-database)",
        task="6-Class Steel Strip Surface Defect Classification",
        model_file_path=str(out_model.relative_to(BASE_DIR)),
        split_sizes={
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": len(image_paths),
        },
        metrics=metrics,
        seed=args.seed,
        selection_rule="Two-stage MobileNetV2 transfer learning with early stopping on stratified test split",
        extra_data={"per_class_report": report_dict},
    )
    print("Successfully recorded verified NEU metrics to docs/metrics/neu_metrics.json and models/model_metadata.json.")


if __name__ == "__main__":
    main()
