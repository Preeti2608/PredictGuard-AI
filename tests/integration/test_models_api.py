def test_get_models_metadata(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert len(data["models"]) >= 2

    model_names = [m["name"] for m in data["models"]]
    assert any("Casting" in name for name in model_names)
    assert any("Machine Failure" in name for name in model_names)

    for m in data["models"]:
        assert "framework" in m
        assert "dataset" in m
        assert "status" in m


def test_home_page_delivery(client):
    res = client.get("/")
    assert res.status_code == 200
    text = res.text
    # Hard constraints from test_api_endpoints.py
    assert "PredictGuard AI" in text
    assert "Casting Surface Inspection" in text
    assert "Machine Failure Forecasting" in text
