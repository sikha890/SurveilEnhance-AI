from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is True


def test_model_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/model")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "SRCNN"
        assert data["scale_factor"] == 4
        assert data["checkpoint"]["epoch"] == 5


def test_enhancement_endpoint():
    from PIL import Image
    from io import BytesIO

    image = Image.new("RGB", (32, 32), color=(120, 100, 80))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    with TestClient(app) as client:
        response = client.post(
            "/api/enhance",
            files={"file": ("test.png", buffer, "image/png")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["x-model-name"] == "SRCNN"
    assert response.headers["x-output-width"] == "128"
    assert response.headers["x-output-height"] == "128"


def test_reject_unsupported_file():
    with TestClient(app) as client:
        response = client.post(
            "/api/enhance",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415
