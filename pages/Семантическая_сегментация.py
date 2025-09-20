import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import json
import io
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import base64
import cv2


img_header = Image.open("images/unet_img/segments.png").convert("RGBA")

buffered = io.BytesIO()
img_header.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{img_str}" width="80" style="margin-right: 10px;">
        <h1 style="margin:0">Семантическая сегментация</h1>
    </div>
    """,
    unsafe_allow_html=True
)

with open("models/unet/unet_config.json", "r") as f:
    config = json.load(f)
encoder_name = config.get("encoder_name", "resnet18")
in_channels = config.get("in_channels", 3)
classes = config.get("classes", 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name=encoder_name,
    encoder_weights=None,
    in_channels=in_channels,
    classes=classes
).to(device)

weights_path = "models/unet/unet_resnet18_best.pt"
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()


# --- Загрузка изображений ---
uploaded_files = st.file_uploader(
    "Загрузите изображения", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True
)
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
        images.append(Image.open(file).convert("RGB"))

if url_input:
    for url in url_input.split("\n"):
        img = load_image_from_url(url.strip())
        if img:
            images.append(img)

# --- Функция инференса и отображения цветной маски ---
def show_preds_streamlit(img_list, model, thr=0.5, device='cpu'):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for idx, img in enumerate(img_list):
        st.write(f"### Изображение {idx+1}")
        
        # Сохраняем оригинальные размеры
        original_size = img.size
        original_np = np.array(img)
        
        # Преобразуем и делаем предсказание
        x = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_mask = torch.sigmoid(model(x))
        
        # Получаем маску и преобразуем к оригинальному размеру
        mask_prob = pred_mask.squeeze().cpu().numpy()
        mask_bin = (mask_prob > thr).astype(np.uint8)
        
        # Масштабируем маску к оригинальному размеру
        mask_resized = cv2.resize(mask_bin, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Создаем цветную маску (зелёный)
        color_mask = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)
        color_mask[mask_resized == 1] = [0, 255, 0]
        
        # Накладываем маску на оригинальное изображение
        overlay = cv2.addWeighted(original_np, 0.7, color_mask, 0.3, 0)
        overlay_img = Image.fromarray(overlay)
        
        # Показываем результаты
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Оригинал", use_container_width=True)
        with col2:
            st.image(overlay_img, caption="Результат обработки", use_container_width=True)
            
            # Кнопка скачивания
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            st.download_button(
                f"Скачать",
                buf.getvalue(),
                file_name=f"segmented_{idx+1}.png",
                mime="image/png",
                key=f"download_{idx}"
            )

# --- Запуск инференса ---
if images:
    st.write("## Результаты сегментации")
    # Добавляем слайдер для порога
    threshold = st.slider("Порог уверенности", 0.1, 0.9, 0.5, 0.05)
    show_preds_streamlit(images, model, thr=threshold)

st.markdown("---")

st.markdown(
    """
    Модель: U-Net  
    Время обучения: 1,5ч  
    Количество эпох: 30  
    Датасет: [Forest Aerial Images for Segmentation](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation)  
    Объем: 5108 изображений  
    """
)

st.markdown("""Метрики""")

with st.expander("Показать детали"):
    image1 = Image.open("images/unet_img/unet_metrics.png")
    st.image(image1, use_container_width=True)