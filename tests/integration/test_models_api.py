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


def test_get_domains_metadata(client):
    res = client.get("/v1/domains")
    assert res.status_code == 200
    data = res.json()
    assert "domains" in data
    assert len(data["domains"]) >= 3
    domain_ids = [d["id"] for d in data["domains"]]
    assert "casting" in domain_ids
    assert "neu_surface" in domain_ids
    assert "mvtec" in domain_ids

    # Verify licenses and trained status
    mvtec_domain = next(d for d in data["domains"] if d["id"] == "mvtec")
    assert "CC BY-NC-SA 4.0" in mvtec_domain["license"]
    assert mvtec_domain["trained"] is False


def test_home_page_delivery(client):
    res = client.get("/")
    assert res.status_code == 200
    text = res.text
    # Hard constraints from test_api_endpoints.py
    assert "PredictGuard AI" in text
    assert "Casting Surface Inspection" in text
    assert "Machine Failure Forecasting" in text
