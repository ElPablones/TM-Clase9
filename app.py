import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

st.set_page_config(page_title="Validador de Gestos", page_icon="✋", layout="centered")

@st.cache_resource
def load_vision_model():
    model = load_model('keras_model.h5', compile=False)
    try:
        with open('labels.txt', 'r') as f:
            labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    except FileNotFoundError:
        labels = ["Arriba", "Abajo", "Izquierda", "Derecha"]
    return model, labels

st.title("✋ Validador de Gestos Keras")
st.markdown("Evalúa y calibra la precisión de tu modelo en tiempo real.")

# --- NUEVO: Panel de Control de Umbrales ---
with st.sidebar:
    st.header("⚙️ Calibración")
    st.markdown("Ajusta la sensibilidad del modelo.")
    
    # Control dinámico del umbral (por defecto 70%)
    umbral_confianza = st.slider(
        "Umbral de Confianza Mínimo", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.70, 
        step=0.05,
        help="Porcentaje mínimo requerido para aceptar un gesto como válido."
    )

with st.spinner("Cargando modelo..."):
    model, labels = load_vision_model()

img_file_buffer = st.camera_input("Toma una foto de tu gesto")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.array(image)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)[0]
    max_index = np.argmax(prediction)
    confidence = prediction[max_index]
    predicted_class = labels[max_index].upper()

    st.divider()
    
    # --- NUEVO: Lógica dinámica de niveles ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if confidence >= umbral_confianza:
            st.success(f"### Gesto detectado: **{predicted_class}**")
            st.progress(float(confidence), text=f"Nivel de confianza: {confidence*100:.1f}%")
        elif confidence >= (umbral_confianza - 0.15):
            # Nivel intermedio: Está cerca del umbral pero no lo supera
            st.warning(f"### Gesto dudoso: **{predicted_class}**")
            st.progress(float(confidence), text=f"Confianza insuficiente ({confidence*100:.1f}%). Requiere {umbral_confianza*100}%")
            st.caption("El modelo tiene una idea de qué gesto es, pero no está completamente seguro.")
        else:
            # Nivel bajo: Predicción muy pobre
            st.error("### Movimiento no reconocido")
            st.progress(float(confidence), text=f"Confianza muy baja: {confidence*100:.1f}%")
            st.caption("Marca el gesto con mayor claridad frente al lente.")
    
    with col2:
        with st.expander("Ver distribución total", expanded=True):
            for i, label in enumerate(labels):
                # Resaltar la clase ganadora en la lista
                if i == max_index:
                    st.markdown(f"**🟢 {label}: {prediction[i]*100:.1f}%**")
                else:
                    st.markdown(f"⚪ {label}: {prediction[i]*100:.1f}%")
