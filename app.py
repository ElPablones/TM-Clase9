import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.set_page_config(page_title="Theremin de Gestos", page_icon="🎹", layout="wide")

@st.cache_resource
def load_vision_model():
    model = load_model('keras_model.h5', compile=False)
    try:
        with open('labels.txt', 'r') as f:
            labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    except FileNotFoundError:
        labels = ["Arriba", "Abajo", "Izquierda", "Derecha"]
    return model, labels

NOTAS_MIDI = {
    "abajo": "C4",
    "izquierda": "D4",
    "derecha": "E4",
    "arriba": "G4"
}

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class GestureAudioProcessor:
    def __init__(self):
        self.model, self.labels = load_vision_model()
        
        self.current_gesture = None
        self.frames_held = 0
        self.cooldown = 0
        self.active_note = ""
        
        # Variables para optimización (Frame-Skipping)
        self.frame_counter = 0
        self.last_prediction = None
        self.last_confidence = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        # LÓGICA DE RENDIMIENTO: Solo predecir 1 de cada 3 frames
        if self.frame_counter % 3 == 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            normalized_img = (np.array(img_resized, dtype=np.float32) / 127.0) - 1
            data = np.expand_dims(normalized_img, axis=0)

            prediction = self.model.predict(data, verbose=0)[0]
            max_index = np.argmax(prediction)
            self.last_confidence = prediction[max_index]
            self.last_prediction = self.labels[max_index].strip().lower()

        # Usar la última predicción calculada para la lógica interactiva
        if self.cooldown > 0:
            self.cooldown -= 1
        else:
            if self.last_confidence > 0.85 and self.last_prediction:
                if self.last_prediction == self.current_gesture:
                    self.frames_held += 1
                else:
                    self.current_gesture = self.last_prediction
                    self.frames_held = 1

                # Disparo más rápido (3 frames en lugar de 5)
                if self.frames_held >= 3:
                    self.active_note = NOTAS_MIDI.get(self.current_gesture, "")
                    self.cooldown = 15  # Recuperación más rápida para agilizar el ritmo
                    print(f"Trigger Nota: {self.active_note} - Gesto: {self.current_gesture}")
            else:
                self.current_gesture = None
                self.frames_held = 0

        # Interfaz de Usuario (HUD en el video)
        if self.cooldown > 0:
            # Sombra para texto (mejor legibilidad)
            cv2.putText(img, f"NOTA: {self.active_note}", (32, 82), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)
            cv2.putText(img, f"NOTA: {self.active_note}", (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 100), 3)
            
            cv2.putText(img, f"Gesto: {self.current_gesture.capitalize()}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(img, (30, 150), (30 + (self.cooldown * 12), 170), (0, 200, 255), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🎹 Theremin Pentatónico Visual")
st.markdown("Interactúa con la cámara. Mantén el gesto para disparar notas en progresión.")

# CONTROL DE LAYOUT: Restringir el tamaño de la cámara usando columnas
col_cam, col_info = st.columns([2, 1])

with col_cam:
    st.markdown("### Cámara Interactiva")
    webrtc_streamer(
        key="gesture-synth",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=GestureAudioProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480}, # Forzar resolución menor
            "audio": False
        },
        async_processing=True,
    )

with col_info:
    st.markdown("### Panel de Control")
    st.info("**Instrucciones:** Haz un gesto frente a la cámara y mantenlo un instante. La barra amarilla indica el tiempo de recuperación antes de la siguiente nota.")
    st.markdown("""
    **Mapeo de Escala:**
    * ⬇️ Abajo: **C4** (Tónica)
    * ⬅️ Izquierda: **D4**
    * ➡️ Derecha: **E4**
    * ⬆️ Arriba: **G4**
    """)
