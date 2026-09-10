import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

st.set_page_config(page_title="Detección de Gestos", page_icon="✋")

# 1. Carga eficiente del modelo y etiquetas
@st.cache_resource
def load_vision_model():
    # compile=False evita warnings de Keras en inferencia
    model = load_model('keras_model.h5', compile=False)
    
    # Intenta cargar labels.txt de Teachable Machine, o usa un fallback
    try:
        with open('labels.txt', 'r') as f:
            labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    except FileNotFoundError:
        labels = ["Izquierda", "Arriba", "Derecha", "Clase 4"] 
        
    return model, labels

st.title("✋ Interfaz de Control por Gestos")
st.markdown("Usa la cámara para clasificar gestos en tiempo real mediante el modelo Keras.")

with st.spinner("Inicializando red neuronal..."):
    model, labels = load_vision_model()

img_file_buffer = st.camera_input("Capturar gesto")

if img_file_buffer is not None:
    # 2. Preprocesamiento limpio
    img = Image.open(img_file_buffer).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 3. Inferencia
    prediction = model.predict(data)[0]
    
    # 4. Extracción de la clase ganadora
    max_index = np.argmax(prediction)
    confidence = prediction[max_index]
    predicted_class = labels[max_index]

    if confidence > 0.6: # Umbral de seguridad
        st.success(f"**Gesto detectado:** {predicted_class} (Confianza: {confidence:.2f})")
        st.progress(float(confidence))
    else:
        st.warning("No se detecta un gesto claro. Intenta acercar la mano.")
