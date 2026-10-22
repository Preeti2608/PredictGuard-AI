"""
Casting Surface Defect Inspection - Transfer Learning Training Script.
Uses MobileNetV2 with transfer learning, data augmentation, and binary crossentropy.
Outputs trained model artifact and evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure root workspace is on python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from tensorflow.keras import layers, models
except ImportError as err:
    print(f"Required libraries missing: {err}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on Casting Quality Dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(BASE_DIR / "data" / "casting_data"),
        help="Path to casting_data directory containing train/ and test/ subdirectories",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(BASE_DIR / "models" / "casting_defect_mobilenet.keras"),
        help="Destination path for trained model",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=str(BASE_DIR / "docs" / "metrics" / "image_metrics.json"),
        help="Destination path for evaluation metrics JSON",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()


def build_model(input_shape=(224, 224, 3), learning_rate=1e-4):
    """Build MobileNetV2 transfer learning architecture with data augmentation."""
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze pretrained feature extractor

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)  # rescale back for mobilenet
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="CastingDefectMobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    print("=" * 60)
    print("CASTING DEFECT CLASSIFICATION - MOBILENETV2 TRAINING")
    print("=" * 60)

    if not data_dir.exists() or not train_dir.exists() or not test_dir.exists():
        print(f"\n[TODO - DATA NEEDED] Dataset not found at: {data_dir}")
        print("Expected structure:")
        print("  data/casting_data/")
        print("    train/ (ok_front/, def_front/)")
        print("    test/  (ok_front/, def_front/)")
        print("\nPlease provide the Kaggle 'Casting product image data for quality inspection' dataset")
        print("in data/casting_data to execute reproducible training.\n")

        # Ensure metrics dir exists and leave honest TODO placeholder if not present
        metrics_file = Path(args.output_metrics)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        if not metrics_file.exists():
            todo_metrics = {
                "status": "PENDING_DATASET",
                "message": "TODO: Provide data/casting_data dataset and execute training/train_image.py",
                "model": "MobileNetV2 Transfer Learning",
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "roc_auc": None,
                "confusion_matrix": None,
            }
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(todo_metrics, f, indent=2)
            print(f"Recorded TODO marker in {metrics_file}")
        sys.exit(0)

    print(f"Loading training data from {train_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(224, 224), batch_size=args.batch_size, label_mode="binary", interpolation="nearest"
    )

    print(f"Loading test data from {test_dir}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(224, 224),
        batch_size=args.batch_size,
        label_mode="binary",
        interpolation="nearest",
        shuffle=False,
    )

    # Rescale to [0, 1]
    rescale = layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)

    model = build_model(learning_rate=args.learning_rate)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    print("\nStarting model training...")
    model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, callbacks=callbacks)

    # Save trained model
    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Model saved to {out_path}")

    # Evaluate on test set
    y_true = []
    y_pred_probs = []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    metrics = {
        "status": "COMPLETED",
        "model_architecture": "MobileNetV2 (Transfer Learning)",
        "dataset": "Casting Product Image Data",
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_pred_probs)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    metrics_path = Path(args.output_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
