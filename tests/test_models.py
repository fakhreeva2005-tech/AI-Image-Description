
from unittest.mock import MagicMock, patch

from PIL import Image

from model import generate_caption
from florence_model import generate_detailed_caption


def test_blip_fast_mode():
    """Проверяет генерацию краткого описания."""

    image = Image.new("RGB", (100, 100), "white")

    processor = MagicMock()
    model = MagicMock()

    processor.return_value = {
        "pixel_values": MagicMock()
    }
    processor.decode.return_value = "a white image"

    with patch(
        "model.load_caption_model",
        return_value=(processor, model)
    ):
        result = generate_caption(image, mode="fast")

    assert result == "a white image"
    model.generate.assert_called_once()


def test_florence_detailed_mode():
    """Проверяет генерацию подробного описания."""

    image = Image.new("RGB", (100, 100), "white")

    processor = MagicMock()
    model = MagicMock()

    processor.return_value = {
        "pixel_values": MagicMock()
    }

    processor.batch_decode.return_value = [
        "generated description"
    ]

    processor.post_process_generation.return_value = {
        "<MORE_DETAILED_CAPTION>":
            "A white image with a plain background."
    }

    with patch(
        "florence_model.load_florence_model",
        return_value=(processor, model)
    ):
        result = generate_detailed_caption(image)

    assert result == (
        "A white image with a plain background."
    )

    model.generate.assert_called_once()

    processor.post_process_generation.assert_called_once()
