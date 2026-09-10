import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.set_page_config(page_title="Sintetizador por Gestos", page_icon="🎹", layout="wide")

@st.cache_resource
def load_vision_model():
    model = load_model('keras_model.h5', compile=False)
    try:
        with open('labels.txt', 'r') as f:
            labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    except FileNotFoundError:
        labels = ["Arriba", "Abajo", "Izquierda", "Derecha"]
    return model, labels

# Mapeo armónico Pentatónico
NOTAS_MIDI = {
    "abajo": "C4",
    "izquierda": "D4",
    "derecha": "E4",
    "arriba": "G4"
}

# Configuración WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class GestureAudioProcessor:
    def __init__(self):
        self.model, self.labels = load_vision_model()
        
        # Máquina de estados para Debouncing
        self.current_gesture = None
        self.frames_held = 0
        self.cooldown = 0
        self.active_note = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Extraer frame y preparar para Keras
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocesamiento estricto 224x224
        img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        normalized_img = (np.array(img_resized, dtype=np.float32) / 127.0) - 1
        data = np.expand_dims(normalized_img, axis=0)

        # Inferencia
        prediction = self.model.predict(data, verbose=0)[0]
        max_index = np.argmax(prediction)
        confidence = prediction[max_index]
        predicted_class = self.labels[max_index].strip().lower()

        # Lógica de Debouncing y Trigger
        if self.cooldown > 0:
            self.cooldown -= 1
        else:
            if confidence > 0.85:  # Umbral alto para evitar falsos positivos
                if predicted_class == self.current_gesture:
                    self.frames_held += 1
                else:
                    self.current_gesture = predicted_class
                    self.frames_held = 1

                # Disparar nota si el gesto se mantiene estable por 5 frames
                if self.frames_held == 5:
                    self.active_note = NOTAS_MIDI.get(self.current_gesture, "")
                    self.cooldown = 20  # Bloquear nuevas notas por ~20 frames
                    
                    # NOTA ARQUITECTÓNICA: Aquí iría el trigger de audio.
                    # En Python nativo usaríamos librerías como 'mido' o 'pygame'.
                    print(f"MIDI TRIGGER -> Note ON: {self.active_note}")
            else:
                self.current_gesture = None
                self.frames_held = 0

        # Feedback Visual (HUD)
        if self.cooldown > 0:
            # Mostrar la nota que está sonando
            cv2.putText(img, f"NOTA: {self.active_note}", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(img, f"Gesto: {self.current_gesture.capitalize()}", (30, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Dibujar barra de progreso del cooldown
            cv2.rectangle(img, (30, 160), (30 + (self.cooldown * 10), 180), (0, 255, 255), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🎹 Theremin Pentatónico Visual")
st.markdown("Usa gestos estables frente a la cámara para disparar notas en una progresión de Do Mayor Pentatónico.")

webrtc_streamer(
    key="gesture-synth",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=GestureAudioProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.divider()
st.caption("**Limitación Arquitectónica de Streamlit:** WebRTC ejecuta el procesamiento de Python en un hilo en segundo plano (background thread). Esto impide inyectar JavaScript (Tone.js) directamente a la vista principal de Streamlit en tiempo real sin una alta latencia. Para un instrumento final de producción, esta lógica de TensorFlow/Keras debería migrar directamente al navegador usando **TensorFlow.js** junto a la Web Audio API.")
