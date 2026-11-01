"""
================================================================================
WARNING: DEVELOPMENT / PLACEHOLDER ARCHITECTURE GENERATOR ONLY
================================================================================
THIS SCRIPT PRODUCES UNTRAINED PLACEHOLDER MODELS FOR ARCHITECTURAL WIRING ONLY.
THESE MODELS ARE NOT TRAINED ON REAL BEARING OR NEU DATASETS AND MUST NEVER BE
FLAGGED AS "TRAINED" OR USED IN PRODUCTION INFERENCE.

ALL METRICS PRODUCED BY OR ASSOCIATED WITH THESE MODELS ARE STRICTLY NULL.
DO NOT USE SYNTHETIC IMAGES GENERATED HERE AS BENCHMARK EVIDENCE.
================================================================================
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.metrics_io import save_placeholder_metrics

MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "sample_images"


def build_models():
    print("=" * 70)
    print("WARNING: Building UNTRAINED architectural placeholder models (dev only)...")
    print("=" * 70)

    base_model_path = MODELS_DIR / "casting_defect_classifier.h5"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Missing {base_model_path}")

    full_base = tf.keras.models.load_model(str(base_model_path), compile=False)
    feature_layer = full_base.get_layer("global_average_pooling2d")
    feature_extractor = tf.keras.Model(inputs=full_base.input, outputs=feature_layer.output)

    # 1. Bearing Model Placeholder (Binary)
    bearing_inputs = tf.keras.Input(shape=(224, 224, 3), name="bearing_image_input")
    x = feature_extractor(bearing_inputs, training=False)
    x = tf.keras.layers.Dense(64, activation="relu", name="bearing_dense")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    bearing_output = tf.keras.layers.Dense(1, activation="sigmoid", name="bearing_defect_prob")(x)
    bearing_model = tf.keras.Model(inputs=bearing_inputs, outputs=bearing_output, name="BearingDefectClassifier")
    bearing_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    bearing_model_path = MODELS_DIR / "bearing_defect_classifier.keras"
    bearing_model.save(str(bearing_model_path))
    print(f"Saved bearing placeholder model to: {bearing_model_path}")

    save_placeholder_metrics(
        metrics_filename="bearing_metrics.json",
        model_name="Bearing Defect Classifier",
        dataset_name="Bearing Defect Dataset (Figshare doi: 10.6084/m9.figshare.25335076)",
        task="Binary Bearing Inspection (Normal Bearing vs Defective Raceway)",
        model_file_path="models/bearing_defect_classifier.keras",
        note="Untrained architectural placeholder. No held-out test evaluation has been executed.",
    )

    # 2. NEU Model Placeholder (6-Class Softmax)
    neu_inputs = tf.keras.Input(shape=(224, 224, 3), name="neu_image_input")
    x = feature_extractor(neu_inputs, training=False)
    x = tf.keras.layers.Dense(128, activation="relu", name="neu_dense")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    neu_output = tf.keras.layers.Dense(6, activation="softmax", name="neu_classes_prob")(x)
    neu_model = tf.keras.Model(inputs=neu_inputs, outputs=neu_output, name="NEUSurfaceDefectClassifier")
    neu_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    neu_model_path = MODELS_DIR / "neu_defect_classifier.keras"
    neu_model.save(str(neu_model_path))
    print(f"Saved NEU placeholder model to: {neu_model_path}")

    save_placeholder_metrics(
        metrics_filename="neu_metrics.json",
        model_name="NEU Surface Defect Classifier",
        dataset_name="NEU Surface Defect Database (kaustubhdikshit/neu-surface-defect-database)",
        task="6-Class Steel Strip Surface Defect Classification",
        model_file_path="models/neu_defect_classifier.keras",
        note="Untrained architectural placeholder. No held-out test evaluation has been executed.",
    )


def generate_synthetic_samples():
    """Generates synthetic drawings strictly marked with 'synthetic_' prefix for dev/testing."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Synthetic Bearing Normal
    img_bn = Image.new("RGB", (224, 224), color=(35, 38, 42))
    d_bn = ImageDraw.Draw(img_bn)
    d_bn.ellipse([20, 20, 204, 204], fill=(70, 75, 80), outline=(120, 130, 140), width=6)
    d_bn.ellipse([60, 60, 164, 164], fill=(35, 38, 42), outline=(100, 110, 120), width=4)
    for angle_deg in range(0, 360, 45):
        rad = np.deg2rad(angle_deg)
        cx = 112 + int(56 * np.cos(rad))
        cy = 112 + int(56 * np.sin(rad))
        d_bn.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(180, 190, 200), outline=(220, 230, 240), width=2)
    img_bn.save(SAMPLES_DIR / "synthetic_bearing_normal.jpeg", "JPEG", quality=95)

    # Synthetic Bearing Defective
    img_bd = img_bn.copy()
    d_bd = ImageDraw.Draw(img_bd)
    d_bd.line([(112, 20), (125, 45), (105, 55)], fill=(15, 15, 15), width=4)
    d_bd.ellipse([150, 95, 175, 120], fill=(20, 20, 20), outline=(50, 50, 50), width=2)
    img_bd.save(SAMPLES_DIR / "synthetic_bearing_defective.jpeg", "JPEG", quality=95)

    # Synthetic NEU Pitted Steel
    img_pitted = Image.new("RGB", (200, 200), color=(140, 142, 145))
    d_p = ImageDraw.Draw(img_pitted)
    np.random.seed(42)
    for _ in range(120):
        px = int(np.random.randint(10, 190))
        py = int(np.random.randint(10, 190))
        pr = int(np.random.randint(2, 6))
        c = int(np.random.randint(30, 70))
        d_p.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(c, c, c))
    img_pitted = img_pitted.filter(ImageFilter.GaussianBlur(radius=0.7))
    img_pitted.save(SAMPLES_DIR / "synthetic_neu_pitted.jpeg", "JPEG", quality=95)

    # Synthetic NEU Scratched Steel
    img_scratch = Image.new("RGB", (200, 200), color=(140, 142, 145))
    d_s = ImageDraw.Draw(img_scratch)
    d_s.line([(20, 30), (180, 170)], fill=(40, 40, 40), width=3)
    d_s.line([(40, 20), (190, 150)], fill=(220, 220, 220), width=1)
    d_s.line([(10, 100), (170, 190)], fill=(35, 35, 35), width=2)
    img_scratch = img_scratch.filter(ImageFilter.GaussianBlur(radius=0.5))
    img_scratch.save(SAMPLES_DIR / "synthetic_neu_scratches.jpeg", "JPEG", quality=95)

    print("Generated synthetic sample images prefixed with 'synthetic_' in sample_images/")


if __name__ == "__main__":
    build_models()
    generate_synthetic_samples()
