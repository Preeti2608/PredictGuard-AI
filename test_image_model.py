from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img

# Load the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "casting_defect_classifier.h5"

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

# Search for test images in casting_data first, falling back to sample_images
candidate_dirs = [BASE_DIR / "sample_images", BASE_DIR / "casting_data"]
image_files = []

for c_dir in candidate_dirs:
    if c_dir.exists():
        found = [
            p for p in c_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            and "__MACOSX" not in p.parts
        ]
        if found:
            image_files.extend(found)
            break

if not image_files:
    print("No images found!")
    print("Please check the sample_images or casting_data folder.")
else:
    image_path = image_files[0]

    # Load and preprocess image exactly as during inference
    image = load_img(str(image_path), target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = float(model.predict(image_array, verbose=0)[0][0])

    # Class mapping: def_front = 0, ok_front = 1
    predicted_class = "ok_front" if prediction >= 0.5 else "def_front"

    print("Model loaded successfully!")
    print("Image:", image_path.name)
    print("Raw prediction:", prediction)
    print("Predicted class:", predicted_class)
    print("Label:", "Normal casting" if predicted_class == "ok_front" else "Defective casting")
