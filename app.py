
import io

import streamlit as st
from PIL import Image, UnidentifiedImageError

from model import generate_caption
from florence_model import generate_detailed_caption
from translator import translate_text


st.set_page_config(
    page_title="AI Image Description",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ===== ДИЗАЙН =====

st.markdown("""
<style>
    .stApp {
        background: #FAF9FF;
        color: #25233A;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #302A4A !important;
    }

    .hero {
        text-align: center;
        padding: 30px 15px 25px;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        color: #5E43B5;
        letter-spacing: -1px;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #77718C;
        line-height: 1.6;
    }

    .info-card {
        background: white;
        border: 1px solid #EAE5FA;
        border-radius: 18px;
        padding: 22px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(80, 55, 150, 0.04);
    }

    .result-label {
        color: #6C51BD;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .result-text {
        color: #302A4A;
        font-size: 18px;
        line-height: 1.7;
        overflow-wrap: anywhere;
    }

    div.stButton > button[kind="primary"] {
        background: #7958D6;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #6543C3;
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 12px;
    }

    .footer {
        text-align: center;
        color: #9891AA;
        font-size: 13px;
        padding-top: 35px;
    }
</style>
""", unsafe_allow_html=True)


# ===== ЗАГОЛОВОК =====

st.markdown("""
<div class="hero">
    <div class="hero-title">✨ AI Image Description</div>
    <div class="hero-subtitle">
        Превращаем изображения в слова с помощью
        искусственного интеллекта.<br>
        Две нейросети — описание и перевод на русский язык.
    </div>
</div>
""", unsafe_allow_html=True)


# ===== НАСТРОЙКИ =====

st.subheader("⚙️ Настройки анализа")

mode_name = st.radio(
    "Выберите режим распознавания",
    ["⚡ Быстрый", "🔍 Подробный"],
    horizontal=True
)


mode = "fast" if mode_name == "⚡ Быстрый" else "detailed"


if mode == "fast":
    model_title = "⚡ BLIP"
    model_description = (
        "Быстрый анализ изображения. Нейросеть определяет "
        "основные объекты и создаёт краткое описание сцены."
    )
    model_name = "Salesforce/blip-image-captioning-base"

else:
    model_title = "🔍 Florence-2"
    model_description = (
        "Подробный анализ изображения. Нейросеть создаёт "
        "развёрнутое описание объектов, деталей "
        "и окружающей обстановки."
    )
    model_name = "florence-community/Florence-2-base-ft"


st.markdown(
    f"""
    <div class="info-card">
        <div class="result-label">{model_title}</div>
        <div class="result-text" style="font-size: 15px;">
            {model_description}
        </div>
        <div style="
            margin-top: 14px;
            color: #9186AA;
            font-size: 12px;
            overflow-wrap: anywhere;
        ">
            Модель: {model_name}
        </div>
        <div style="
            margin-top: 10px;
            color: #6C51BD;
            font-size: 13px;
            font-weight: 600;
        ">
            🌐 Перевод: OPUS-MT (английский → русский)
        </div>
    </div>
    """,
    unsafe_allow_html=True
)




# ===== ЗАГРУЗКА =====

st.subheader("📤 Загрузка изображения")

uploaded_file = st.file_uploader(
    "Выберите фотографию",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image_bytes = uploaded_file.getvalue()

        image = Image.open(io.BytesIO(image_bytes))
        image.load()

        st.image(
            image,
            caption="Предварительный просмотр",
            use_container_width=True
        )

        if st.button(
            "✨ Анализировать изображение",
            type="primary",
            use_container_width=True
        ):

            with st.status(
                "Выполняется анализ изображения...",
                expanded=True
            ) as status:

                st.write("🔍 BLIP создаёт описание...")

                if mode == "detailed":
                    english_caption = generate_detailed_caption(image)
                else:
                    english_caption = generate_caption(image, mode="fast")

                st.write("🌐 OPUS-MT переводит описание...")

                russian_caption = translate_text(
                    english_caption
                )

                status.update(
                    label="Анализ завершён!",
                    state="complete",
                    expanded=False
                )

            st.session_state["result"] = {
                "english": english_caption,
                "russian": russian_caption,
                "mode": mode,
                "image_bytes": image_bytes
            }

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "Не удалось открыть изображение. "
            "Попробуйте другой JPG или PNG."
        )

    except Exception as error:
        st.error("Произошла ошибка при анализе изображения.")
        st.exception(error)


# ===== РЕЗУЛЬТАТЫ =====


result = st.session_state.get("result")

if result and uploaded_file is not None:

    if (
        result["image_bytes"] == uploaded_file.getvalue()
        and result["mode"] == mode
    ):

        st.markdown("---")
        st.subheader("✨ Результаты анализа")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown(
                '<div class="info-card">'
                '<div class="result-label">🇬🇧 English description</div>'
                '<div class="result-text">'
                + __import__("html").escape(result["english"])
                + '</div></div>',
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                '<div class="info-card">'
                '<div class="result-label">🇷🇺 Описание на русском</div>'
                '<div class="result-text">'
                + __import__("html").escape(result["russian"])
                + '</div></div>',
                unsafe_allow_html=True
            )

        report = (
            "AI Image Description\n"
            "====================\n\n"
            f"Mode: {result['mode']}\n\n"
            f"English:\n{result['english']}\n\n"
            f"Русский:\n{result['russian']}\n"
        )

        st.download_button(
            "📥 Скачать описание (.txt)",
            data=report.encode("utf-8"),
            file_name="image_description.txt",
            mime="text/plain",
            use_container_width=True
        )


# ===== ПОДВАЛ =====

st.markdown("""
<div class="footer">
    AI Image Description · BLIP + OPUS-MT · Hugging Face
</div>
""", unsafe_allow_html=True)
