from functools import lru_cache
from pathlib import Path
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_name: str = "PredictGuard AI"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "production"
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    models_dir: Path = Path("models")
    image_model_path: Path = Path("models/casting_defect_classifier.h5")
    sensor_model_path: Path = Path("models/decision_tree.joblib")
    regression_model_path: Path = Path("models/tool_wear_regressor.joblib")
    model_metadata_path: Path = Path("models/model_metadata.json")
    sample_images_dir: Path = Path("sample_images")
    frontend_dir: Path = Path("frontend")
    docs_metrics_dir: Path = Path("docs/metrics")

    # Security & CORS
    # Default is empty list (same-origin only, removes wildcard + credentials)
    cors_origins: Union[str, List[str]] = []

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_predict: str = "60/minute"

    # Image upload hardening
    max_image_size_bytes: int = 5 * 1024 * 1024  # 5 MB
    max_image_pixels: int = 10_000_000  # 10 MP decompression bomb guard
    image_interpolation: str = "nearest"  # matches Keras default nearest

    # Risk thresholds
    image_defect_medium_threshold: float = 0.30
    image_defect_high_threshold: float = 0.70
    sensor_failure_medium_threshold: float = 0.30
    sensor_failure_high_threshold: float = 0.70

    # Physical ranges for sensor telemetry (based on AI4I predictive maintenance dataset)
    air_temp_min: float = 290.0
    air_temp_max: float = 315.0
    process_temp_min: float = 300.0
    process_temp_max: float = 325.0
    rotational_speed_min: float = 1000.0
    rotational_speed_max: float = 3000.0
    torque_min: float = 0.0
    torque_max: float = 100.0
    tool_wear_min: float = 0.0
    tool_wear_max: float = 300.0

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Union[str, List[str]]) -> List[str]:
        if isinstance(value, str):
            if not value.strip():
                return []
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache()
def get_settings() -> Settings:
    return Settings()
