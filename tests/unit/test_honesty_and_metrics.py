"""
Unit & Integration Tests for ML Honesty, Metric Agreement & Untrained Model Guards.

Enforces:
1. Every model marked trained must have real matching metrics in docs/metrics/.
2. Untrained/placeholder models must have null metrics and cannot serve predictions (503).
3. Metadata, metrics JSON, and config MUST agree on model file paths and status.
"""

import json
from pathlib import Path

from app.config import get_settings

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
METRICS_DIR = BASE_DIR / "docs" / "metrics"


def test_metadata_file_validity():
    assert METADATA_PATH.exists(), "models/model_metadata.json must exist"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "models" in meta
    assert len(meta["models"]) >= 4


def test_metadata_and_metrics_json_agreement():
    """Verify that metadata and metrics JSON files agree on status, file paths, and values."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 1. Bearing Model
    bearing_entry = next((m for m in metadata["models"] if "Bearing" in m["name"]), None)
    assert bearing_entry is not None
    bearing_metrics_file = METRICS_DIR / "bearing_metrics.json"
    assert bearing_metrics_file.exists()
    with open(bearing_metrics_file, "r", encoding="utf-8") as f:
        bearing_metrics = json.load(f)

    assert bearing_entry["trained"] is False
    assert bearing_entry["status"] == "UNTRAINED_PLACEHOLDER"
    assert bearing_metrics["status"] == "UNTRAINED_PLACEHOLDER"
    # Ensure metrics are null
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert bearing_entry["metrics"][k] is None
        assert bearing_metrics.get("benchmark", {}).get(k) is None

    # 2. NEU Model
    neu_entry = next((m for m in metadata["models"] if "NEU" in m["name"]), None)
    assert neu_entry is not None
    neu_metrics_file = METRICS_DIR / "neu_metrics.json"
    assert neu_metrics_file.exists()
    with open(neu_metrics_file, "r", encoding="utf-8") as f:
        neu_metrics = json.load(f)

    assert neu_entry["trained"] is False
    assert neu_entry["status"] == "UNTRAINED_PLACEHOLDER"
    assert neu_metrics["status"] == "UNTRAINED_PLACEHOLDER"
    for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        assert neu_entry["metrics"].get(k) is None
        assert neu_metrics.get("benchmark", {}).get(k) is None

    # 3. Sensor Model
    sensor_entry = next((m for m in metadata["models"] if "Machine Failure" in m["name"]), None)
    assert sensor_entry is not None
    sensor_metrics_file = METRICS_DIR / "sensor_metrics.json"
    assert sensor_metrics_file.exists()
    with open(sensor_metrics_file, "r", encoding="utf-8") as f:
        sensor_metrics = json.load(f)

    assert sensor_entry["trained"] is True
    assert sensor_metrics["status"] == "COMPLETED"
    # Check calibrated metrics match
    assert sensor_entry["metrics"]["accuracy"] == sensor_metrics["calibrated_test_metrics"]["accuracy"]
    assert sensor_entry["metrics"]["recall"] == sensor_metrics["calibrated_test_metrics"]["failure_recall"]
    assert sensor_entry["metrics"]["precision"] == sensor_metrics["calibrated_test_metrics"]["failure_precision"]


def test_config_and_metadata_agree_on_served_models():
    """Verify that app/config.py and models/model_metadata.json name the same files."""
    settings = get_settings()
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    sensor_meta = next(m for m in metadata["models"] if "Machine Failure" in m["name"])
    # Both must point to the calibrated model file
    assert settings.sensor_model_path.name == Path(sensor_meta["file_path"]).name


def test_untrained_domain_returns_503(client):
    """Calling an untrained domain must return HTTP 503 with error_code 'MODEL_NOT_TRAINED'."""
    sample_file = BASE_DIR / "sample_images" / "sample_normal.jpeg"
    with open(sample_file, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        # Bearing domain
        res_bearing = client.post("/v1/predict/image", files=files, data={"domain": "bearing"})
        assert res_bearing.status_code == 503
        data_b = res_bearing.json()
        assert data_b["error_code"] == "MODEL_NOT_TRAINED"

    with open(sample_file, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        # NEU domain
        res_neu = client.post("/v1/predict/image", files=files, data={"domain": "neu_surface"})
        assert res_neu.status_code == 503
        data_n = res_neu.json()
        assert data_n["error_code"] == "MODEL_NOT_TRAINED"
