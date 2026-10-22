import json
from typing import Any, Dict, List, Optional

import joblib
import tensorflow as tf

from app.config import Settings
from app.core.logging import get_logger

logger = get_logger("predictguard.model_registry")


class ModelRegistry:
    """
    Centralized registry for ML models and runtime metadata.
    Models are loaded during FastAPI lifespan startup, not module import time.
    """

    def __init__(self) -> None:
        self.image_model: Optional[Any] = None
        self.sensor_model: Optional[Any] = None
        self.wear_regressor: Optional[Any] = None
        self.metadata: Dict[str, Any] = {"models": []}
        self.loaded_status: Dict[str, bool] = {
            "image_model": False,
            "sensor_model": False,
            "wear_model": False,
        }

    def load_all(self, settings: Settings) -> None:
        """Load all configured models and metadata synchronously during app startup."""
        logger.info("Initializing ModelRegistry from disk...")

        # 1. Load Image Classifier
        image_path = settings.image_model_path
        if not image_path.is_absolute():
            image_path = settings.base_dir / image_path

        if image_path.exists():
            try:
                self.image_model = tf.keras.models.load_model(str(image_path), compile=False)
                self.loaded_status["image_model"] = True
                logger.info("Loaded casting defect classifier from %s", image_path)
            except Exception as exc:
                logger.error("Failed to load image classifier from %s: %s", image_path, exc)
                self.image_model = None
                self.loaded_status["image_model"] = False
        else:
            logger.warning("Image model file does not exist at %s", image_path)
            self.image_model = None
            self.loaded_status["image_model"] = False

        # 2. Load Sensor Failure Classifier
        sensor_path = settings.sensor_model_path
        if not sensor_path.is_absolute():
            sensor_path = settings.base_dir / sensor_path

        if sensor_path.exists():
            try:
                self.sensor_model = joblib.load(sensor_path)
                self.loaded_status["sensor_model"] = True
                logger.info("Loaded sensor failure classifier from %s", sensor_path)
            except Exception as exc:
                logger.error("Failed to load sensor pipeline from %s: %s", sensor_path, exc)
                self.sensor_model = None
                self.loaded_status["sensor_model"] = False
        else:
            logger.warning("Sensor model file does not exist at %s", sensor_path)
            self.sensor_model = None
            self.loaded_status["sensor_model"] = False

        # 3. Load Tool Wear Regressor (optional)
        wear_path = settings.regression_model_path
        if not wear_path.is_absolute():
            wear_path = settings.base_dir / wear_path

        if wear_path.exists():
            try:
                self.wear_regressor = joblib.load(wear_path)
                self.loaded_status["wear_model"] = True
                logger.info("Loaded tool wear regressor from %s", wear_path)
            except Exception as exc:
                logger.warning("Optional wear regressor failed to load from %s: %s", wear_path, exc)
                self.wear_regressor = None
                self.loaded_status["wear_model"] = False
        else:
            self.wear_regressor = None
            self.loaded_status["wear_model"] = False

        # 4. Load metadata JSON
        meta_path = settings.model_metadata_path
        if not meta_path.is_absolute():
            meta_path = settings.base_dir / meta_path

        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info("Loaded model metadata from %s", meta_path)
            except Exception as exc:
                logger.error("Failed to parse metadata JSON at %s: %s", meta_path, exc)
                self.metadata = {"models": []}
        else:
            logger.warning("Model metadata JSON missing at %s", meta_path)
            self.metadata = {"models": []}

    def is_ready(self) -> bool:
        """System is ready if both primary production models are loaded and available."""
        return bool(self.image_model is not None and self.sensor_model is not None)

    def get_image_model(self) -> Any:
        return self.image_model

    def get_sensor_model(self) -> Any:
        return self.sensor_model

    def get_wear_regressor(self) -> Optional[Any]:
        return self.wear_regressor

    def get_metadata(self) -> List[Dict[str, Any]]:
        """Return metadata list enriched with live runtime load status."""
        models_list = self.metadata.get("models", [])
        enriched: List[Dict[str, Any]] = []
        for m in models_list:
            item = dict(m)
            name = item.get("name", "")
            if "Casting" in name:
                item["status"] = "loaded" if self.loaded_status["image_model"] else "missing_or_failed"
            elif "Machine Failure" in name:
                item["status"] = "loaded" if self.loaded_status["sensor_model"] else "missing_or_failed"
            elif "Tool Wear" in name:
                item["status"] = "loaded" if self.loaded_status["wear_model"] else "not_configured"
            else:
                item["status"] = "unknown"
            enriched.append(item)
        return enriched


# Global singleton registry instance
registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    return registry
