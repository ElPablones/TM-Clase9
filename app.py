import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import streamlit.components.v1 as components

st.set_page_config(page_title="Controlador MIDI por Gestos", page_icon="🎛️", layout="wide")

# 1. Carga eficiente del modelo y etiquetas
@st.cache_resource
def load_vision_model():
    model = load_model('keras_model.h5', compile=False)
    try:
        with open('labels.txt', 'r') as f:
            # Limpia el texto para obtener solo el nombre (ej. "0 Arriba" -> "Arriba")
            labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    except FileNotFoundError:
        labels = ["Arriba", "Abajo", "Izquierda", "Derecha"]
    return model, labels

# 2. Motor de Síntesis de Audio (Tone.js)
def play_synth_sound(gesture):
    # Mapeo espacial a notas musicales
    notas = {
        "arriba": "C5",      # Nota aguda
        "abajo": "C3",       # Nota grave
        "izquierda": "F3",   # Tono medio-bajo
        "derecha": "G4"      # Tono medio-alto
    }
    
    nota = notas.get(gesture.lower(), "A4") # A4 por defecto si hay otra clase
    
    # Inyectamos Tone.js en el frontend para generar el sonido de forma nativa
    js_code = f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js"></script>
    <div id="audio-ui" style="font-family: sans-serif; text-align: center; padding: 10px; background: #e0f7fa; border-radius: 8px;">
        <p style="margin: 0; color: #006064;">🔊 Generando frecuencia para: <b>{nota}</b></p>
    </div>
    <script>
        // La interacción del usuario con la cámara autoriza el AudioContext
        async function triggerAudio() {{
            await Tone.start();
            // Sintetizador básico, ideal para prototipado rápido
            const synth = new Tone.Synth({{
                oscillator: {{ type: "sawtooth" }},
                envelope: {{ attack: 0.05, decay: 0.2, sustain: 0.2, release: 1 }}
            }}).toDestination();
            
            synth.triggerAttackRelease("{nota}", "4n");
        }}
        triggerAudio();
    </script>
    """
    components.html(js_code, height=60)

st.title("🎛️ Controlador Espacial por Gestos")
st.markdown("Captura un gesto. La red neuronal lo clasificará y activará un oscilador web en tiempo real.")

with st.spinner("Cargando modelo neuronal..."):
    model, labels = load_vision_model()

# Captura de imagen
img_file_buffer = st.camera_input("Capturar Gesto")

if img_file_buffer is not None:
    # 3. Preprocesamiento estricto para Keras (224x224, normalizado -1 a 1)
    image = Image.open(img_file_buffer).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.array(image)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 4. Inferencia
    prediction = model.predict(data)[0]
    max_index = np.argmax(prediction)
    confidence = prediction[max_index]
    predicted_class = labels[max_index]

    col1, col2 = st.columns([2, 1])
    
    with col1:
        if confidence > 0.7: # Umbral de seguridad del 70%
            st.success(f"### Gesto: {predicted_class}")
            st.progress(float(confidence), text=f"Confianza: {confidence*100:.1f}%")
            
            # Disparamos la síntesis web enviando el gesto detectado
            play_synth_sound(predicted_class)
        else:
            st.warning("Movimiento no reconocido. Marca el gesto con más claridad.")
            
    with col2:
        with st.expander("Ver telemetría cruda"):
            for i, label in enumerate(labels):
                st.write(f"{label}: {prediction[i]:.3f}")
