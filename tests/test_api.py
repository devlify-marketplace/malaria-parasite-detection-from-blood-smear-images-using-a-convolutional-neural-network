from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from malaria_cnn.api import app


def test_health_and_model_info():
    with TestClient(app) as client:
        health = client.get("/api/health")
        info = client.get("/api/model-info")
        assert health.status_code == 200
        assert "model_ready" in health.json()
        assert info.status_code == 200
        assert info.json()["classes"] == ["Uninfected", "Parasitized"]
        assert "not validated" in info.json()["scope"]


def test_demo_page_and_local_illustration_are_served():
    with TestClient(app) as client:
        page = client.get("/")
        image = client.get("/assets/cdc-malaria-smear.png")
        assert page.status_code == 200
        assert "CellSight" in page.text
        assert "not a medical diagnosis" in page.text
        assert image.status_code == 200
        assert image.headers["content-type"].startswith("image/png")


def test_predict_accepts_png_and_does_not_retain_upload():
    with TestClient(app) as client:
        if not client.get("/api/health").json()["model_ready"]:
            return
        buffer = BytesIO()
        Image.new("RGB", (72, 88), color=(130, 90, 120)).save(buffer, format="PNG")
        response = client.post(
            "/api/predict",
            files={"file": ("test.png", buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["prediction"] in {"Uninfected", "Parasitized"}
        assert 0 <= result["parasitized_score"] <= 1
        assert result["upload_retained"] is False
        assert "not a patient diagnosis" in result["disclaimer"]


def test_predict_rejects_invalid_file():
    with TestClient(app) as client:
        if not client.get("/api/health").json()["model_ready"]:
            return
        response = client.post("/api/predict", files={"file": ("bad.png", b"not an image", "image/png")})
        assert response.status_code == 415
