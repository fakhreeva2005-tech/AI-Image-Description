
import torch
import streamlit as st

from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"


@st.cache_resource(show_spinner="Загрузка модели перевода...")
def load_translation_model():
    """Загружает модель перевода один раз."""

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()

    return tokenizer, model



def translate_text(text: str) -> str:
    """Переводит описание с английского на русский."""

    if not text or not text.strip():
        return ""

    tokenizer, model = load_translation_model()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    translations = []

    for line in lines:
        inputs = tokenizer(
            line,
            return_tensors="pt",
            truncation=True
        )

        with torch.inference_mode():
            translated = model.generate(
                **inputs,
                max_new_tokens=100
            )

        result = tokenizer.decode(
            translated[0],
            skip_special_tokens=True
        )

        translations.append(result)

    return "\n".join(translations)


if __name__ == "__main__":
    example = "A cat is sitting on a wooden table."
    print("Оригинал:", example)
    print("Перевод:", translate_text(example))
