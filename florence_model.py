
import torch
import streamlit as st

from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    Florence2ForConditionalGeneration
)

MODEL_NAME = "florence-community/Florence-2-base-ft"


@st.cache_resource(show_spinner="Загрузка Florence-2...")
def load_florence_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = Florence2ForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return processor, model


def generate_detailed_caption(image: Image.Image) -> str:
    """Создаёт подробное описание изображения."""

    processor, model = load_florence_model()

    image = ImageOps.exif_transpose(image).convert("RGB")

    prompt = "<MORE_DETAILED_CAPTION>"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    result = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=image.size
    )

    return result[prompt].strip()
