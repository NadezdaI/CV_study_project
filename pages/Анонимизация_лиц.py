import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2
import io
from pathlib import Path
import base64


img = Image.open("images/face_img/face_mask.png").convert("RGBA")

buffered = BytesIO()
img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{img_str}" width="80" style="margin-right: 10px;">
        <h1 style="margin:0">Анонимизация лиц</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
Сервис обнаруживает лица на загруженных изображениях и скрывает их с помощью GaussianBlur.  Метод cv2.GaussianBlur — функция OpenCV (Open Source Computer Vision Library), которая применяется для размытия изображения с использованием гауссового фильтра. 

"""
)

# --- Загрузка изображений ---
uploaded_files = st.file_uploader("Загрузите изображения", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
url_input = st.text_area("Или укажите URL изображений (по одному на строку)")

def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        st.error(f"Не удалось загрузить {url}: {e}")
        return None

images = []

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        images.append(img)

if url_input:
    urls = url_input.split("\n")
    for url in urls:
        img = load_image_from_url(url.strip())
        if img:
            images.append(img)

# --- Инференс ---
if images:
    model = YOLO("models/face_/face_yolo11m_best.pt")

    for idx, img in enumerate(images):
        st.write(f"Изображение {idx+1}")
        results = model.predict(img)

        for result in results:
            img_np = np.array(img)
            img_out = img_np.copy()

            if result.boxes is not None:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    roi = img_out[y1:y2, x1:x2]
                    if roi.size > 0:
                        k = max(15, (min(x2 - x1, y2 - y1) // 2) | 1)
                        roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
                        img_out[y1:y2, x1:x2] = roi_blur

                img_to_show = img_out
                caption_text = "Результат обработки"

            else:
                img_plot = result.plot()
                img_to_show = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
                caption_text = "Результат обработки"

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_np, caption="Оригинал", use_container_width=True)
            with col2:
                st.image(img_to_show, caption=caption_text, use_container_width=True)

                buf = io.BytesIO()
                Image.fromarray(img_to_show).save(buf, format="PNG")
                st.download_button(
                    "Скачать",
                    buf.getvalue(),
                    file_name=f"inference_{idx+1}.png",
                    mime="image/png"
                )

st.markdown("---")

st.markdown(
    """
    Модель: YOLOv11-medium   
    Время обучения: 4ч  
    Количество эпох: 18  
    Датасет: [Face-Detection-Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)  
    Объем: учебный - 13386  /  валидационный - 3347 изображений  

    """
)

st.markdown("""**Метрики**""")

col1, col2, col3 = st.columns(3)
col1.metric("mAP50", "0.897", "0.72")
col2.metric("Precision", "0.89", "0.814")
col3.metric("Recall", "0.828", "0.654")

with st.expander("Показать детали"):
    image1 = Image.open("images/face_img/metrics_mAP50.png")
    image2 = Image.open("images/face_img/BoxPR_curve.png")
    st.image(image1, use_container_width=True)
    st.image(image2, use_container_width=True)