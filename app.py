import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
import io

MODELS = {
    "Маскировка лиц": {
        "path": "models/face_/face_yolo11m_best.pt",
        "description": "Обнаружение лиц на изображении",
        "model": "Yolo11 medium,",
        "type": "detection",
        "info": {"эпохи": 30, "обучающая выборка": "14k", "метрики": "mAP50=0.92"}
    },
    "Диагностика опухолей мозга": {
        "path": "models/brain_/brain_model.pt",
        "description": "Brain Tumor Segmentation",
        "type": "segmentation",
        "info": {"эпохи": 50, "обучающая выборка": "5k", "метрики": "Dice=0.87"}
    },
    "Аэрокосмические изображения": {
        "path": "models/aerospace/best.pt",
        "description": "Aerospace Image Analysis",
        "type": "classification",
        "info": {"эпохи": 40, "обучающая выборка": "8k", "метрики": "acc=0.95"}
    },
}

# --- Sidebar для выбора страницы ---
st.sidebar.title("Выбор темы")
page = st.sidebar.radio("", list(MODELS.keys()))
model_info = MODELS[page]

st.title(page)
st.write(model_info["description"])

# --- Показ метрик ---
with st.expander("Информация о модели"):
    st.write(f"Количество эпох: {model_info['info']['эпохи']}")
    st.write(f"Размер обучающей выборки: {model_info['info']['обучающая выборка']}")
    st.write(f"Метрики: {model_info['info']['метрики']}")

# --- Загрузка изображений ---
uploaded_files = st.file_uploader("Загрузите изображения", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

url_input = st.text_area("Или укажите URL изображений (по одному на строку)")

# --- Функция для загрузки изображений по URL ---
def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        st.error(f"Не удалось загрузить {url}: {e}")
        return None

images = []

# --- Обработка файлов ---
for file in uploaded_files:
    img = Image.open(file).convert("RGB")
    images.append(img)

# --- Обработка URL ---
if url_input:
    urls = url_input.split("\n")
    for url in urls:
        img = load_image_from_url(url.strip())
        if img:
            images.append(img)

if images:
    model = YOLO(model_info["path"])


# --- Инференс ---
for idx, img in enumerate(images):
    st.write(f'Изображение {idx+1}')
    results = model.predict(img)

    for result in results:
        img_np = np.array(img)
        img_out = img_np.copy()  # для маскировки

        # --- Маскирование лиц ---
        if page == "Маскировка лиц" and result.boxes is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                if w > 30 and h > 30:
                    roi = img_out[y1:y2, x1:x2]
                    k = max(15, (min(w, h)//2) | 1)
                    roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
                    img_out[y1:y2, x1:x2] = roi_blur

            img_to_show = img_out  # изображение для отображения и скачивания
            caption_text = "Результат обработки"

        else:
            img_plot = result.plot()  # numpy BGR
            img_to_show = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
            caption_text = "Результат исследования"

        # --- Две колонки: оригинал и инференс ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_np, caption="Оригинал", use_container_width=True)
        with col2:
            st.image(img_to_show, caption=caption_text, use_container_width=True)

            # --- Кнопка скачать ---
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(img_to_show).save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Скачать",  
                data=byte_im,
                file_name=f"inference_{idx+1}.png",
                mime="image/png",
                key=f"download_{idx}",
                use_container_width=False
            )