import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from model import generate_caption
from translator import translate_text


app = FastAPI(
    title="AI Image Description API",
    description="API для анализа изображений с помощью BLIP и OPUS-MT",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "AI Image Description API работает!",
        "status": "ok"
    }


@app.post("/analyze")
def analyze_image(file: UploadFile = File(...)):
    # Проверяем формат файла
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только JPG и PNG"
        )

    try:
        # Читаем загруженное изображение
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Не удалось открыть изображение"
        )

    # Получаем описание через BLIP
    english_caption = generate_caption(image, mode="fast")

    # Переводим описание через OPUS-MT
    russian_caption = translate_text(english_caption)

    return {
        "filename": file.filename,
        "model": "Salesforce/blip-image-captioning-base",
        "english_description": english_caption,
        "russian_description": russian_caption
    }