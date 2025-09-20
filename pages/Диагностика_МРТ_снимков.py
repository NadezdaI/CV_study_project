import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2
import io
import base64


img = Image.open("images/brain_img/mri.webp").convert("RGBA")

buffered = BytesIO()
img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{img_str}" width="80" style="margin-right: 10px;">
        <h1 style="margin:0">Диагностика МРТ снимков</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    Сервис предназначен для анализа МРТ-снимков головного мозга с целью выявления и точной локализации опухоли, позволяя отделять патологические ткани от здоровых структур. 
    Модель может обрабатывать снимки в трех плоскостях - аксиальной, корональной и сагиттальной.
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
    model = YOLO("models/brain_/best.pt")

    for idx, img in enumerate(images):
        st.write(f"Изображение {idx+1}")
        results = model.predict(img)

        for result in results:
            img_np = np.array(img)
            img_out = img_np.copy()

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
    Время обучения: 1,5ч  
    Количество эпох: 30  
    Датасет: [Brain tumor object detection datasets](https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets)  
    Объем: 2220 изображений  

    """
)

st.markdown("""Метрики""")

with st.expander("Показать детали"):
    image1 = Image.open("images/brain_img/image.png")
    st.image(image1, use_container_width=True)
