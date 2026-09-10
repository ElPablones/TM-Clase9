import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import mido

st.set_page_config(page_title="Controlador MIDI Gestual", layout="wide")

# 1. Configuración de Salida MIDI
# Crea un puerto virtual o conéctalo al puerto que use tu loopMIDI/IAC Driver
try:
    # mido.get_output_names() te permite ver los puertos disponibles
    midi_out = mido.open_output(mido.get_output_names()[0]) 
except:
    midi_out = None
    st.warning("⚠️ No se detectó un puerto MIDI. La interfaz visual funcionará, pero no enviará notas a tu DAW.")

@st.cache_resource
def load_vision_model():
    model = load_model('keras_model.h5', compile=False)
    try:
        with open('labels.txt', 'r') as f:
            labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    except FileNotFoundError:
        labels = ["Arriba", "Abajo", "Izquierda", "Derecha"]
    return model, labels

# Mapeo de Gestos a Notas MIDI (Escala Pentatónica C)
MIDI_MAP = {
    "abajo": 60,      # C4
    "izquierda": 62,  # D4
    "derecha": 64,    # E4
    "arriba": 67      # G4
}

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class MidiGestureProcessor:
    def __init__(self):
        self.model, self.labels = load_vision_model()
        self.current_gesture = None
        self.frames_held = 0
        self.cooldown = 0
        self.frame_counter = 0
        self.last_prediction = None
        self.last_confidence = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        self.frame_counter += 1

        # Inferencia 1 de cada 3 frames para mantener el video fluido
        if self.frame_counter % 3 == 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            normalized_img = (np.array(img_resized, dtype=np.float32) / 127.0) - 1
            data = np.expand_dims(normalized_img, axis=0)

            prediction = self.model.predict(data, verbose=0)[0]
            max_index = np.argmax(prediction)
            self.last_confidence = prediction[max_index]
            self.last_prediction = self.labels[max_index].strip().lower()

        # Lógica de Interacción y MIDI
        trigger_active = False
        
        if self.cooldown > 0:
            self.cooldown -= 1
            trigger_active = True # Mantener iluminado el HUD
        else:
            if self.last_confidence > 0.85 and self.last_prediction:
                if self.last_prediction == self.current_gesture:
                    self.frames_held += 1
                else:
                    self.current_gesture = self.last_prediction
                    self.frames_held = 1

                if self.frames_held >= 3:
                    self.cooldown = 15
                    trigger_active = True
                    
                    # Enviar Nota MIDI al DAW
                    nota = MIDI_MAP.get(self.current_gesture)
                    if nota and midi_out:
                        msg = mido.Message('note_on', note=nota, velocity=100, time=0)
                        midi_out.send(msg)
            else:
                self.current_gesture = None
                self.frames_held = 0

        # --- DIBUJO DE INTERFAZ (HUD) SOBRE EL VIDEO ---
        overlay = img.copy()
        alpha = 0.4
        
        # Coordenadas de los 4 "pads" visuales
        zones = {
            "arriba": (0, 0, w, int(h*0.2)),
            "abajo": (0, int(h*0.8), w, h),
            "izquierda": (0, 0, int(w*0.2), h),
            "derecha": (int(w*0.8), 0, w, h)
        }

        # Dibujar zonas
        for zone_name, (x1, y1, x2, y2) in zones.items():
            color = (0, 255, 0) if (trigger_active and self.current_gesture == zone_name) else (50, 50, 50)
            thickness = -1 if (trigger_active and self.current_gesture == zone_name) else 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Panel de Estado
        cv2.putText(img, "MIDI Out Activo" if midi_out else "Sin Conexion MIDI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if trigger_active and self.current_gesture:
            cv2.putText(img, f"TOCANDO: {self.current_gesture.upper()}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🎛️ Controlador MIDI Visual")
st.markdown("Convierte tus gestos en señales MIDI para controlar sintetizadores o efectos en tu DAW.")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="midi-controller",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=MidiGestureProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Ruteo MIDI")
    st.info("Para recibir estas señales en tu DAW, necesitas un puerto MIDI virtual (ej. loopMIDI en Windows o IAC Driver en Mac).")
    if st.button("Listar Puertos MIDI Disponibles"):
        puertos = mido.get_output_names()
        if puertos:
            for p in puertos:
                st.write(f"- {p}")
        else:
            st.error("No se detectaron puertos.")
