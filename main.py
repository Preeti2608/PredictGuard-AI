import io
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("predictguard")

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DIR = BASE_DIR / "sample_images"

IMAGE_MODEL_PATH = MODELS_DIR / "casting_defect_classifier.h5"
SENSOR_MODEL_PATH = MODELS_DIR / "decision_tree.joblib"

# Load trained models
image_model = None
sensor_model = None

try:
    if IMAGE_MODEL_PATH.exists():
        image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)
        logger.info("Image classifier loaded from %s", IMAGE_MODEL_PATH)
    else:
        logger.error("Image model file missing at %s", IMAGE_MODEL_PATH)
except Exception as exc:
    logger.error("Failed to load image model: %s", exc)

try:
    if SENSOR_MODEL_PATH.exists():
        sensor_model = joblib.load(SENSOR_MODEL_PATH)
        logger.info("Sensor pipeline loaded from %s", SENSOR_MODEL_PATH)
    else:
        logger.error("Sensor model file missing at %s", SENSOR_MODEL_PATH)
except Exception as exc:
    logger.error("Failed to load sensor model: %s", exc)

# Create FastAPI application
app = FastAPI(
    title="PredictGuard AI",
    description="Industrial AI Predictive Maintenance Platform: Image Defect Classification & Sensor Failure Forecasting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local development and diverse environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sample images if directory exists
if SAMPLE_DIR.exists():
    app.mount("/sample_images", StaticFiles(directory=SAMPLE_DIR), name="sample_images")


class SensorInput(BaseModel):
    Type: str = Field(pattern="^[LMH]$", description="Product type variant: L (Low), M (Medium), H (High)")
    air_temperature: float = Field(alias="Air temperature [K]", description="Air temperature in Kelvin")
    process_temperature: float = Field(alias="Process temperature [K]", description="Process temperature in Kelvin")
    rotational_speed: float = Field(alias="Rotational speed [rpm]", description="Rotational speed in RPM")
    torque: float = Field(alias="Torque [Nm]", description="Torque in Newton-meters")
    tool_wear: float = Field(alias="Tool wear [min]", description="Tool wear time in minutes")

    model_config = {"populate_by_name": True}


@app.get("/", include_in_schema=False)
def home():
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        return {
            "message": "PredictGuard AI API is running",
            "docs": "/docs",
        }
    return FileResponse(index_file)


@app.get("/api/health", tags=["System"])
def health_check():
    """Verify system health, API responsiveness, and model availability."""
    models_ready = image_model is not None and sensor_model is not None
    return {
        "status": "online" if models_ready else "degraded",
        "image_model_loaded": image_model is not None,
        "sensor_model_loaded": sensor_model is not None,
        "version": "1.0.0",
    }


@app.post("/predict/image", tags=["Inference"])
async def predict_image(file: UploadFile = File(...)):
    """
    Classify a casting surface image into 'Normal casting' (ok_front)
    or 'Defective casting' (def_front) using the trained CNN.
    """
    if image_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image classification model is not loaded.",
        )

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a JPG or PNG image.",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        score = float(image_model.predict(image_array, verbose=0)[0][0])

        # Training mapping: def_front=0, ok_front=1
        predicted_class = "ok_front" if score >= 0.5 else "def_front"

        return {
            "prediction": predicted_class,
            "label": ("Normal casting" if predicted_class == "ok_front" else "Defective casting"),
            "score": score,
            "note": "Score is the model output; verify its calibration before treating it as a probability.",
        }

    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not parse image file. Ensure it is a valid JPG or PNG.",
        )
    except Exception as exc:
        logger.error("Error processing image prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not process image. Please verify file integrity.",
        )


@app.post("/predict/sensors", tags=["Inference"])
def predict_sensors(data: SensorInput):
    """
    Predict machine failure from operating telemetry using the trained
    scikit-learn Decision Tree pipeline.
    """
    if sensor_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sensor prediction pipeline is not loaded.",
        )

    try:
        sample = pd.DataFrame(
            [
                {
                    "Type": data.Type,
                    "Air temperature [K]": data.air_temperature,
                    "Process temperature [K]": data.process_temperature,
                    "Rotational speed [rpm]": data.rotational_speed,
                    "Torque [Nm]": data.torque,
                    "Tool wear [min]": data.tool_wear,
                }
            ]
        )

        prediction = int(sensor_model.predict(sample)[0])

        result = "Failure" if prediction == 1 else "No Failure"

        response = {
            "prediction": result,
            "prediction_code": prediction,
        }

        if hasattr(sensor_model, "predict_proba"):
            probabilities = sensor_model.predict_proba(sample)[0]
            response["class_probabilities"] = {
                str(label): float(probability) for label, probability in zip(sensor_model.classes_, probabilities)
            }

        return response

    except Exception as exc:
        logger.error("Error predicting sensor data: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not predict sensor data. Verify input values.",
        )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
