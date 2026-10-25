
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

MODEL_NAME = "florence-community/Florence-2-base-ft"

print("Загружаем Florence-2...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Florence2ForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Модель успешно загружена!")

# Создаём небольшое тестовое изображение
image = Image.open("cat_test.png").convert("RGB")

# Проверяем генерацию подробного описания
prompt = "<MORE_DETAILED_CAPTION>"

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)

print("Генерируем описание...")

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

print("\nРЕЗУЛЬТАТ:")
print(result)
