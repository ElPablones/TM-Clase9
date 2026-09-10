import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

st.set_page_config(page_title="Validador de Gestos", page_icon="✋", layout="centered")

# 1. Cargar el modelo en caché para que la app sea rápida
@st.cache_resource
def load_vision_model():
    model = load_model('keras_model.h5', compile=False)
    try:
        with open('labels.txt', 'r') as f:
            # Limpia los números al inicio de las etiquetas
            labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    except FileNotFoundError:
        labels = ["Arriba", "Abajo", "Izquierda", "Derecha"]
    return model, labels

st.title("✋ Validador de Gestos Keras")
st.markdown("Usa esta interfaz para probar la precisión de tu modelo entrenado en Teachable Machine.")

with st.spinner("Cargando modelo..."):
    model, labels = load_vision_model()

# 2. Componente nativo de Streamlit (A prueba de fallos en la nube)
img_file_buffer = st.camera_input("Toma una foto de tu gesto")

if img_file_buffer is not None:
    # 3. Procesamiento estricto de la imagen
    image = Image.open(img_file_buffer).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.array(image)
    
    # Normalizar la matriz de píxeles
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 4. Inferencia
    prediction = model.predict(data)[0]
    max_index = np.argmax(prediction)
    confidence = prediction[max_index]
    predicted_class = labels[max_index].upper()

    # 5. Interfaz de Resultados
    st.divider()
    
    if confidence > 0.70:
        st.success(f"### Gesto detectado: **{predicted_class}**")
        st.progress(float(confidence), text=f"Nivel de confianza: {confidence*100:.1f}%")
    else:
        st.warning("⚠️ No estoy seguro del gesto. Intenta mejorar la iluminación o la posición de la mano.")
    
    with st.expander("Ver todas las probabilidades"):
        for i, label in enumerate(labels):
            st.metric(label=label, value=f"{prediction[i]*100:.1f}%")
