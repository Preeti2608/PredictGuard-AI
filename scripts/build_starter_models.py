"""
Initialize starter models and sample images for Multi-Domain Inspection.
Constructs models using local pre-trained backbone from models/casting_defect_classifier.h5
with zero network dependencies.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "sample_images"


def build_models():
    print("Loading base casting classifier from local disk...")
    base_model_path = MODELS_DIR / "casting_defect_classifier.h5"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Missing {base_model_path}")

    full_base = tf.keras.models.load_model(str(base_model_path), compile=False)

    # Extract feature representation before classification head
    # Layer out_relu or global_average_pooling2d
    feature_layer = full_base.get_layer("global_average_pooling2d")
    feature_extractor = tf.keras.Model(inputs=full_base.input, outputs=feature_layer.output)

    # 1. Build Bearing Model (Binary: 1.0=Normal, 0.0=Defective)
    bearing_inputs = tf.keras.Input(shape=(224, 224, 3), name="bearing_image_input")
    x = feature_extractor(bearing_inputs, training=False)
    x = tf.keras.layers.Dense(64, activation="relu", name="bearing_dense")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    bearing_output = tf.keras.layers.Dense(1, activation="sigmoid", name="bearing_defect_prob")(x)
    bearing_model = tf.keras.Model(inputs=bearing_inputs, outputs=bearing_output, name="BearingDefectClassifier")
    bearing_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    bearing_model_path = MODELS_DIR / "bearing_defect_classifier.keras"
    bearing_model.save(str(bearing_model_path))
    print(f"Saved bearing defect classifier to: {bearing_model_path}")

    # 2. Build NEU Model (6-Class Softmax: crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches)
    neu_inputs = tf.keras.Input(shape=(224, 224, 3), name="neu_image_input")
    x = feature_extractor(neu_inputs, training=False)
    x = tf.keras.layers.Dense(128, activation="relu", name="neu_dense")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    neu_output = tf.keras.layers.Dense(6, activation="softmax", name="neu_classes_prob")(x)
    neu_model = tf.keras.Model(inputs=neu_inputs, outputs=neu_output, name="NEUSurfaceDefectClassifier")
    neu_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    neu_model_path = MODELS_DIR / "neu_defect_classifier.keras"
    neu_model.save(str(neu_model_path))
    print(f"Saved NEU surface defect classifier to: {neu_model_path}")


def create_sample_images():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Normal Bearing: Smooth concentric steel rings, circular balls, uniform metallic gray
    img_bearing_norm = Image.new("RGB", (300, 300), color=(140, 142, 145))
    draw = ImageDraw.Draw(img_bearing_norm)
    # Outer ring
    draw.ellipse([20, 20, 280, 280], outline=(70, 72, 75), width=18)
    draw.ellipse([38, 38, 262, 262], fill=(160, 163, 166))
    # Raceway groove
    draw.ellipse([70, 70, 230, 230], fill=(110, 112, 115))
    # Balls
    centers = [(150, 85), (200, 105), (215, 150), (200, 195), (150, 215), (100, 195), (85, 150), (100, 105)]
    for cx, cy in centers:
        draw.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=(210, 215, 220), outline=(50, 52, 55), width=2)
    # Inner ring & bore
    draw.ellipse([110, 110, 190, 190], fill=(160, 163, 166), outline=(70, 72, 75), width=8)
    draw.ellipse([125, 125, 175, 175], fill=(35, 36, 38))
    img_bearing_norm = img_bearing_norm.filter(ImageFilter.GaussianBlur(0.6))
    bearing_norm_path = SAMPLES_DIR / "sample_bearing_normal.jpeg"
    img_bearing_norm.save(str(bearing_norm_path), "JPEG", quality=95)
    print(f"Generated normal bearing sample: {bearing_norm_path}")

    # 2. Defective Bearing: Bearing with raceway spalling, ball indentation, fatigue cracks
    img_bearing_def = img_bearing_norm.copy()
    draw_def = ImageDraw.Draw(img_bearing_def)
    # Spalling pitting marks on raceway
    draw_def.polygon([(78, 142), (92, 138), (88, 158), (74, 154)], fill=(20, 20, 22))
    draw_def.polygon([(195, 100), (212, 95), (208, 115)], fill=(18, 18, 20))
    # Ball spall flaw
    draw_def.ellipse([145, 80, 158, 92], fill=(25, 25, 28))
    # Fatigue crack lines
    draw_def.line([(185, 185), (210, 210)], fill=(15, 15, 18), width=3)
    draw_def.line([(205, 200), (215, 205)], fill=(15, 15, 18), width=2)
    bearing_def_path = SAMPLES_DIR / "sample_bearing_defective.jpeg"
    img_bearing_def.save(str(bearing_def_path), "JPEG", quality=95)
    print(f"Generated defective bearing sample: {bearing_def_path}")

    # 3. NEU Pitted Surface Steel: Hot rolled steel with cavitation pitting craters
    img_pitted = Image.new("RGB", (300, 300), color=(128, 130, 134))
    draw_p = ImageDraw.Draw(img_pitted)
    np.random.seed(42)
    # Add rolling direction grain texture
    for y in range(0, 300, 3):
        shade = 128 + int(np.random.normal(0, 8))
        draw_p.line([(0, y), (300, y)], fill=(shade, shade, shade), width=2)
    # Pitted depressions
    pit_centers = [(60, 80), (75, 90), (140, 160), (150, 175), (210, 70), (230, 220), (180, 120), (90, 240)]
    for px, py in pit_centers:
        rad = np.random.randint(6, 14)
        draw_p.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(30, 32, 35))
        draw_p.ellipse([px - rad + 2, py - rad + 2, px + rad - 1, py + rad - 1], fill=(50, 52, 55))
    neu_pitted_path = SAMPLES_DIR / "sample_neu_pitted.jpeg"
    img_pitted.save(str(neu_pitted_path), "JPEG", quality=95)
    print(f"Generated NEU pitted surface sample: {neu_pitted_path}")

    # 4. NEU Scratches Steel: Hot rolled steel strip with longitudinal mechanical abrasions
    img_scratch = Image.new("RGB", (300, 300), color=(132, 134, 138))
    draw_s = ImageDraw.Draw(img_scratch)
    for y in range(0, 300, 4):
        shade = 132 + int(np.random.normal(0, 6))
        draw_s.line([(0, y), (300, y)], fill=(shade, shade, shade), width=2)
    # Longitudinal scratches across strip
    draw_s.line([(20, 40), (280, 52)], fill=(25, 25, 28), width=3)
    draw_s.line([(40, 130), (270, 145)], fill=(30, 30, 35), width=4)
    draw_s.line([(50, 148), (220, 155)], fill=(45, 45, 50), width=2)
    draw_s.line([(10, 220), (290, 235)], fill=(20, 20, 25), width=3)
    neu_scratch_path = SAMPLES_DIR / "sample_neu_scratches.jpeg"
    img_scratch.save(str(neu_scratch_path), "JPEG", quality=95)
    print(f"Generated NEU scratches sample: {neu_scratch_path}")


if __name__ == "__main__":
    build_models()
    create_sample_images()
