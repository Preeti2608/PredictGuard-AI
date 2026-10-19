"""
PredictGuard AI — Automated System Integration Test Suite
Verifies server startup, static asset delivery, health checks,
and both machine learning prediction endpoints (image and sensor).
"""

import sys
import time
from pathlib import Path
import requests

BASE_URL = "http://127.0.0.1:8000"
PROJECT_DIR = Path(__file__).resolve().parent

def run_tests():
    print("=" * 60)
    print("PREDICTGUARD AI — INTEGRATION & REGRESSION TEST SUITE")
    print("=" * 60)
    passed = 0
    total = 8

    # 1. Health check endpoint
    print("\n[Test 1/8] Checking GET /api/health ...")
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert data.get("status") == "online", f"Status degraded: {data}"
        assert data.get("image_model_loaded") is True, "Image model not loaded"
        assert data.get("sensor_model_loaded") is True, "Sensor model not loaded"
        print(f" PASS: API Online. Models loaded: {data}")
        passed += 1
    except Exception as e:
        print(f" FAIL: {e}")

    # 2. Homepage delivery
    print("\n[Test 2/8] Checking GET / (Dashboard UI Delivery) ...")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert "PredictGuard AI" in r.text, "Title not found in UI"
        assert "Casting Surface Inspection" in r.text, "Casting card missing"
        assert "Machine Failure Forecasting" in r.text, "Sensor card missing"
        print(f" PASS: Homepage HTML served successfully ({len(r.text)} bytes).")
        passed += 1
    except Exception as e:
        print(f" FAIL: {e}")

    # 3. Sensor prediction - Normal operation (No Failure)
    print("\n[Test 3/8] Checking POST /predict/sensors [Nominal Telemetry] ...")
    try:
        normal_payload = {
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100.0
        }
        r = requests.post(f"{BASE_URL}/predict/sensors", json=normal_payload, timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        res = r.json()
        assert res.get("prediction") == "No Failure", f"Expected 'No Failure', got {res.get('prediction')}"
        assert res.get("prediction_code") == 0, f"Expected code 0, got {res.get('prediction_code')}"
        assert "class_probabilities" in res, "Missing class probabilities"
        print(f" PASS: Normal sensor prediction verified -> {res['prediction']} (Code: {res['prediction_code']})")
        print(f"       Class probabilities: {res['class_probabilities']}")
        passed += 1
    except Exception as e:
        print(f" FAIL: {e}")

    # 4. Sensor prediction - High stress operation (Failure)
    print("\n[Test 4/8] Checking POST /predict/sensors [High-Stress Telemetry] ...")
    try:
        failure_payload = {
            "Type": "L",
            "Air temperature [K]": 302.0,
            "Process temperature [K]": 311.0,
            "Rotational speed [rpm]": 1350.0,
            "Torque [Nm]": 68.0,
            "Tool wear [min]": 215.0
        }
        r = requests.post(f"{BASE_URL}/predict/sensors", json=failure_payload, timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        res = r.json()
        assert res.get("prediction") == "Failure", f"Expected 'Failure', got {res.get('prediction')}"
        assert res.get("prediction_code") == 1, f"Expected code 1, got {res.get('prediction_code')}"
        print(f" PASS: Failure sensor prediction verified -> {res['prediction']} (Code: {res['prediction_code']})")
        print(f"       Class probabilities: {res['class_probabilities']}")
        passed += 1
    except Exception as e:
        print(f" FAIL: {e}")

    # 5. Sensor prediction - Invalid input validation
    print("\n[Test 5/8] Checking POST /predict/sensors [Invalid Machine Type] ...")
    try:
        invalid_payload = {
            "Type": "INVALID_TYPE",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100.0
        }
        r = requests.post(f"{BASE_URL}/predict/sensors", json=invalid_payload, timeout=10)
        assert r.status_code == 422, f"Expected 422 validation error, got {r.status_code}"
        print(f" PASS: Request correctly rejected by Pydantic validation (HTTP 422).")
        passed += 1
    except Exception as e:
        print(f" FAIL: {e}")

    # 6. Image prediction - Defective Casting
    print("\n[Test 6/8] Checking POST /predict/image [Defective Sample Image] ...")
    def_img_path = PROJECT_DIR / "sample_images" / "sample_defective.jpeg"
    try:
        assert def_img_path.exists(), f"Sample image missing at {def_img_path}"
        with open(def_img_path, "rb") as f:
            files = {"file": ("sample_defective.jpeg", f, "image/jpeg")}
            r = requests.post(f"{BASE_URL}/predict/image", files=files, timeout=20)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        res = r.json()
        assert res.get("prediction") == "def_front", f"Expected 'def_front', got {res.get('prediction')}"
        assert res.get("label") == "Defective casting", f"Expected 'Defective casting', got {res.get('label')}"
        assert res.get("score") < 0.5, f"Score should be < 0.5 for defect, got {res.get('score')}"
        print(f" PASS: Defective casting classified correctly -> Label: '{res['label']}' (Score: {res['score']:.6f})")
        passed += 1
    except Exception as e:
        print(f" FAIL: {e}")

    # 7. Image prediction - Normal Casting
    print("\n[Test 7/8] Checking POST /predict/image [Normal Sample Image] ...")
    norm_img_path = PROJECT_DIR / "sample_images" / "sample_normal.jpeg"
    try:
        assert norm_img_path.exists(), f"Sample image missing at {norm_img_path}"
        with open(norm_img_path, "rb") as f:
            files = {"file": ("sample_normal.jpeg", f, "image/jpeg")}
            r = requests.post(f"{BASE_URL}/predict/image", files=files, timeout=20)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        res = r.json()
        assert res.get("prediction") == "ok_front", f"Expected 'ok_front', got {res.get('prediction')}"
        assert res.get("label") == "Normal casting", f"Expected 'Normal casting', got {res.get('label')}"
        assert res.get("score") >= 0.5, f"Score should be >= 0.5 for normal, got {res.get('score')}"
        print(f" PASS: Normal casting classified correctly -> Label: '{res['label']}' (Score: {res['score']:.6f})")
        passed += 1
    except Exception as e:
        print(f" FAIL: {e}")

    # 8. Image prediction - Invalid file type rejection
    print("\n[Test 8/8] Checking POST /predict/image [Invalid File Type Reject] ...")
    try:
        files = {"file": ("test.txt", b"invalid image bytes", "text/plain")}
        r = requests.post(f"{BASE_URL}/predict/image", files=files, timeout=10)
        assert r.status_code == 400, f"Expected 400 Bad Request, got {r.status_code}"
        print(f" PASS: Invalid file type rejected with HTTP 400: '{r.json().get('detail')}'")
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
