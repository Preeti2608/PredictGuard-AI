from app.services.model_registry import get_model_registry


def test_health_live(client):
    res = client.get("/health/live")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "live"
    assert "timestamp" in data


def test_health_ready_nominal(client):
    res = client.get("/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["image_model_loaded"] is True
    assert data["sensor_model_loaded"] is True


def test_health_ready_failure_when_model_missing(client):
    registry = get_model_registry()
    original_model = registry.image_model

    try:
        # Simulate missing image model
        registry.image_model = None
        registry.loaded_status["image_model"] = False

        res = client.get("/health/ready")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "not_ready"
        assert data["image_model_loaded"] is False
    finally:
        # Restore model
        registry.image_model = original_model
        registry.loaded_status["image_model"] = True

    # Verify restored
    res_restored = client.get("/health/ready")
    assert res_restored.status_code == 200


def test_legacy_health_contract(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["image_model_loaded"] is True
    assert data["sensor_model_loaded"] is True
    assert "version" in data
