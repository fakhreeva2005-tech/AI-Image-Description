
import torch
import streamlit as st

from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"


@st.cache_resource(show_spinner="Загрузка модели BLIP...")
def load_caption_model():
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def generate_caption(image: Image.Image, mode="fast") -> str:
    """Генерирует описание изображения."""

    processor, model = load_caption_model()
    image = ImageOps.exif_transpose(image).convert("RGB")

    def predict(prompt=None):
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=5,
                repetition_penalty=1.2
            )

        return processor.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

    # Быстрый режим
    if mode == "fast":
        return predict()

    # Подробный режим
    if mode == "detailed":

        main_caption = predict()

        prompts = [
            "In the image,",
            "In the background,"
        ]

        descriptions = [main_caption]

        for prompt in prompts:
            caption = predict(prompt)

            if caption.lower() not in [
                item.lower() for item in descriptions
            ]:
                descriptions.append(caption)

        return "\n".join(descriptions)

    raise ValueError("Неизвестный режим анализа")
