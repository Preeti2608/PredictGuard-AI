def test_predict_sensors_nominal(client):
    payload = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }
    res = client.post("/v1/predict/sensors", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "No Failure"
    assert data["prediction_code"] == 0
    assert data["failure_probability"] <= 0.5
    assert data["risk_level"] == "LOW"
    assert "top_contributing_factors" in data
    assert len(data["top_contributing_factors"]) > 0
    assert "class_probabilities" in data
    assert data["latency_ms"] > 0
    assert "request_id" in data
    assert res.headers.get("X-Request-ID") == data["request_id"]


def test_predict_sensors_high_stress(client):
    payload = {
        "Type": "L",
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 311.0,
        "Rotational speed [rpm]": 1350.0,
        "Torque [Nm]": 68.0,
        "Tool wear [min]": 215.0,
    }
    res = client.post("/v1/predict/sensors", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "Failure"
    assert data["prediction_code"] == 1
    assert data["failure_probability"] > 0.5
    assert data["risk_level"] in ["MEDIUM", "HIGH"]
    assert "recommended_action" in data


def test_predict_sensors_snake_case_keys(client):
    payload = {
        "type": "H",
        "air_temperature": 298.0,
        "process_temperature": 308.0,
        "rotational_speed": 1600.0,
        "torque": 35.0,
        "tool_wear": 50.0,
    }
    res = client.post("/v1/predict/sensors", json=payload)
    assert res.status_code == 200
    assert res.json()["prediction"] == "No Failure"


def test_predict_sensors_legacy_path_alias(client):
    payload = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }
    res = client.post("/predict/sensors", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "No Failure"
    assert data["prediction_code"] == 0


def test_predict_sensors_out_of_range(client):
    # Air temperature out of physical range (500 K)
    payload = {
        "Type": "M",
        "Air temperature [K]": 500.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }
    res = client.post("/v1/predict/sensors", json=payload)
    assert res.status_code == 422
    assert res.json()["error_code"] == "VALIDATION_ERROR"


def test_predict_sensors_missing_field(client):
    payload = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        # missing process_temperature, rotational_speed, torque, tool_wear
    }
    res = client.post("/v1/predict/sensors", json=payload)
    assert res.status_code == 422


def test_predict_sensors_nan_value(client):
    payload = {
        "Type": "M",
        "Air temperature [K]": "NaN",
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }
    res = client.post("/v1/predict/sensors", json=payload)
    assert res.status_code == 422


def test_predict_sensors_invalid_type(client):
    payload = {
        "Type": "INVALID_VARIANT",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }
    res = client.post("/v1/predict/sensors", json=payload)
    assert res.status_code == 422
