from pathlib import Path

SAMPLE_DIR = Path("sample_images")


def test_predict_image_defective_class(client):
    img_path = SAMPLE_DIR / "sample_defective.jpeg"
    assert img_path.exists(), "Sample defective image missing"

    with open(img_path, "rb") as f:
        files = {"file": ("sample_defective.jpeg", f, "image/jpeg")}
        res = client.post("/v1/predict/image", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "def_front"
    assert data["label"] == "Defective casting"
    assert data["defect_probability"] > 0.5
    assert data["score"] < 0.5
    assert data["risk_level"] in ["MEDIUM", "HIGH"]
    assert "recommended_action" in data
    assert "model_version" in data
    assert data["latency_ms"] > 0
    assert "request_id" in data
    # Header check
    assert res.headers.get("X-Request-ID") == data["request_id"]


def test_predict_image_normal_class(client):
    img_path = SAMPLE_DIR / "sample_normal.jpeg"
    assert img_path.exists(), "Sample normal image missing"

    with open(img_path, "rb") as f:
        files = {"file": ("sample_normal.jpeg", f, "image/jpeg")}
        res = client.post("/v1/predict/image", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "ok_front"
    assert data["label"] == "Normal casting"
    assert data["defect_probability"] < 0.5
    assert data["score"] >= 0.5
    assert data["risk_level"] == "LOW"


def test_predict_image_legacy_path_alias(client):
    img_path = SAMPLE_DIR / "sample_normal.jpeg"
    with open(img_path, "rb") as f:
        files = {"file": ("sample_normal.jpeg", f, "image/jpeg")}
        res = client.post("/predict/image", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "ok_front"
    assert "score" in data
    assert "note" in data


def test_predict_image_empty_file(client):
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = client.post("/v1/predict/image", files=files)
    assert res.status_code in [400, 422]
    data = res.json()
    assert "error_code" in data
    assert data["request_id"]


def test_predict_image_corrupt_file(client):
    # Valid JPEG header followed by junk bytes
    files = {"file": ("corrupt.jpg", b"\xff\xd8\xff\xe0junkdata12345", "image/jpeg")}
    res = client.post("/v1/predict/image", files=files)
    assert res.status_code == 400
    assert res.json()["error_code"] == "INVALID_IMAGE"


def test_predict_image_wrong_type(client):
    files = {"file": ("document.pdf", b"%PDF-1.4 header", "application/pdf")}
    res = client.post("/v1/predict/image", files=files)
    assert res.status_code in [400, 415]


def test_predict_image_spoofed_content_type(client):
    # Client sends text file claiming Content-Type: image/jpeg
    files = {"file": ("fake.jpg", b"This is plainly a text file and not a JPEG", "image/jpeg")}
    res = client.post("/v1/predict/image", files=files)
    assert res.status_code == 400
    assert "JPG or PNG" in res.json()["message"]


def test_predict_image_oversize(client):
    # Send > 5MB file
    oversize_bytes = b"\xff\xd8\xff" + b"0" * (6 * 1024 * 1024)
    files = {"file": ("huge.jpg", oversize_bytes, "image/jpeg")}
    res = client.post("/v1/predict/image", files=files)
    assert res.status_code in [400, 413]
