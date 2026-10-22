"""
PredictGuard AI — Automated System Integration Test Suite
Verifies server startup, static asset delivery, health checks,
and both machine learning prediction endpoints (image and sensor).
Uses FastAPI TestClient so no live server background process is required.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

PROJECT_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_1_health_check(client):
    r = client.get("/api/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data.get("status") == "online", f"Status degraded: {data}"
    assert data.get("image_model_loaded") is True, "Image model not loaded"
    assert data.get("sensor_model_loaded") is True, "Sensor model not loaded"


def test_2_homepage_delivery(client):
    r = client.get("/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert "PredictGuard AI" in r.text, "Title not found in UI"
    assert "Casting Surface Inspection" in r.text, "Casting card missing"
    assert "Machine Failure Forecasting" in r.text, "Sensor card missing"


def test_3_sensor_prediction_normal(client):
    normal_payload = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }
    r = client.post("/predict/sensors", json=normal_payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    res = r.json()
    assert res.get("prediction") == "No Failure", f"Expected 'No Failure', got {res.get('prediction')}"
    assert res.get("prediction_code") == 0, f"Expected code 0, got {res.get('prediction_code')}"
    assert "class_probabilities" in res, "Missing class probabilities"


def test_4_sensor_prediction_failure(client):
    failure_payload = {
        "Type": "L",
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 311.0,
        "Rotational speed [rpm]": 1350.0,
        "Torque [Nm]": 68.0,
        "Tool wear [min]": 215.0,
    }
    r = client.post("/predict/sensors", json=failure_payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    res = r.json()
    assert res.get("prediction") == "Failure", f"Expected 'Failure', got {res.get('prediction')}"
    assert res.get("prediction_code") == 1, f"Expected code 1, got {res.get('prediction_code')}"


def test_5_sensor_prediction_invalid(client):
    invalid_payload = {
        "Type": "INVALID_TYPE",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }
    r = client.post("/predict/sensors", json=invalid_payload)
    assert r.status_code == 422, f"Expected 422 validation error, got {r.status_code}"


def test_6_image_prediction_defective(client):
    def_img_path = PROJECT_DIR / "sample_images" / "sample_defective.jpeg"
    assert def_img_path.exists(), f"Sample image missing at {def_img_path}"
    with open(def_img_path, "rb") as f:
        files = {"file": ("sample_defective.jpeg", f, "image/jpeg")}
        r = client.post("/predict/image", files=files)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    res = r.json()
    assert res.get("prediction") == "def_front", f"Expected 'def_front', got {res.get('prediction')}"
    assert res.get("label") == "Defective casting", f"Expected 'Defective casting', got {res.get('label')}"
    assert res.get("score") < 0.5, f"Score should be < 0.5 for defect, got {res.get('score')}"


def test_7_image_prediction_normal(client):
    norm_img_path = PROJECT_DIR / "sample_images" / "sample_normal.jpeg"
    assert norm_img_path.exists(), f"Sample image missing at {norm_img_path}"
    with open(norm_img_path, "rb") as f:
        files = {"file": ("sample_normal.jpeg", f, "image/jpeg")}
        r = client.post("/predict/image", files=files)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    res = r.json()
    assert res.get("prediction") == "ok_front", f"Expected 'ok_front', got {res.get('prediction')}"
    assert res.get("label") == "Normal casting", f"Expected 'Normal casting', got {res.get('label')}"
    assert res.get("score") >= 0.5, f"Score should be >= 0.5 for normal, got {res.get('score')}"


def test_8_image_prediction_invalid_file(client):
    files = {"file": ("test.txt", b"invalid image bytes", "text/plain")}
    r = client.post("/predict/image", files=files)
    assert r.status_code == 400, f"Expected 400 Bad Request, got {r.status_code}"


def run_tests():
    print("=" * 60)
    print("PREDICTGUARD AI — INTEGRATION & REGRESSION TEST SUITE")
    print("=" * 60)
    passed = 0
    total = 8

    with TestClient(app) as c:
        tests = [
            ("Checking GET /api/health", test_1_health_check),
            ("Checking GET / (Dashboard UI Delivery)", test_2_homepage_delivery),
            ("Checking POST /predict/sensors [Nominal Telemetry]", test_3_sensor_prediction_normal),
            ("Checking POST /predict/sensors [High-Stress Telemetry]", test_4_sensor_prediction_failure),
            ("Checking POST /predict/sensors [Invalid Machine Type]", test_5_sensor_prediction_invalid),
            ("Checking POST /predict/image [Defective Sample Image]", test_6_image_prediction_defective),
            ("Checking POST /predict/image [Normal Sample Image]", test_7_image_prediction_normal),
            ("Checking POST /predict/image [Invalid File Type Reject]", test_8_image_prediction_invalid_file),
        ]

        for i, (name, test_fn) in enumerate(tests, 1):
            print(f"\n[Test {i}/8] {name} ...")
            try:
                test_fn(c)
                print(" PASS")
                passed += 1
            except Exception as e:
                print(f" FAIL: {e}")

    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} TESTS PASSED")
    print("=" * 60)
    if passed == total:
        print("ALL TESTS PASSED WITH 100% SUCCESS.")
        sys.exit(0)
    else:
        print(f"WARNING: {total - passed} tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
