from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_api_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_analyze_invalid_format():
    response = client.post(
        "/analyze",
        files={
            "file": ("test.txt", b"hello", "text/plain")
        }
    )

    assert response.status_code == 400


def test_api_analyze_success():
    from PIL import Image
    import io

    image = Image.new("RGB", (100, 100), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    with (
        patch("api.generate_caption", return_value="a white image"),
        patch("api.translate_text", return_value="Белое изображение")
    ):
        response = client.post(
            "/analyze",
            files={
                "file": (
                    "test.png",
                    buffer.getvalue(),
                    "image/png"
                )
            }
        )

    assert response.status_code == 200
    assert response.json()["english_description"] == "a white image"
    assert response.json()["russian_description"] == "Белое изображение"