import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import pyautogui
import time

# Configuración de seguridad de PyAutoGUI (pausa de seguridad)
pyautogui.PAUSE = 0.1

st.set_page_config(page_title="Controlador Gestual", page_icon="🕹️", layout="wide")

@st.cache_resource
def load_vision_model():
    model = load_model('keras_model.h5', compile=False)
    try:
        with open('labels.txt', 'r') as f:
            labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    except FileNotFoundError:
        labels = ["Arriba", "Abajo", "Izquierda", "Derecha"]
    return model, labels

st.title("🕹️ Interfaz de Control Cinético")
st.markdown("Usa la cámara nativa para convertir tus gestos en comandos de teclado físicos en tiempo real.")

model, labels = load_vision_model()

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Panel de Control")
    iniciar_camara = st.checkbox("Encender Sistema de Visión")
    st.markdown("""
    **Mapeo de Teclas Activo:**
    * ⬆️ Arriba: `Flecha Arriba` (Scroll / Subir)
    * ⬇️ Abajo: `Flecha Abajo` (Scroll / Bajar)
    * ⬅️ Izquierda: `Flecha Izquierda` (Anterior)
    * ➡️ Derecha: `Flecha Derecha` (Siguiente)
    """)
    estado_ui = st.empty()

with col1:
    # Contenedor vacío donde inyectaremos el video frame a frame
    marco_video = st.empty()

if iniciar_camara:
    # 0 es el índice de la cámara web principal
    cap = cv2.VideoCapture(0)
    
    # Reducir la resolución de captura para maximizar los FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Máquina de estados para evitar múltiples pulsaciones (Debouncing)
    current_gesture = None
    frames_held = 0
    cooldown = 0
    frame_counter = 0
    last_prediction = None
    last_confidence = 0.0

    while iniciar_camara:
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo acceder a la cámara. Verifica los permisos de tu sistema operativo.")
            break

        # Efecto espejo para que la interacción sea natural
        frame = cv2.flip(frame, 1)
        frame_counter += 1

        # Optimización: Inferencia 1 de cada 3 frames
        if frame_counter % 3 == 0:
            # Preprocesamiento para Keras
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            normalized_img = (np.array(img_resized, dtype=np.float32) / 127.0) - 1
            data = np.expand_dims(normalized_img, axis=0)

            prediction = model.predict(data, verbose=0)[0]
            max_index = np.argmax(prediction)
            last_confidence = prediction[max_index]
            last_prediction = labels[max_index].strip().lower()

        trigger_active = False

        # Lógica de Disparo y Cooldown
        if cooldown > 0:
            cooldown -= 1
            trigger_active = True
        else:
            if last_confidence > 0.85 and last_prediction:
                if last_prediction == current_gesture:
                    frames_held += 1
                else:
                    current_gesture = last_prediction
                    frames_held = 1

                # Si mantiene el gesto por 4 frames, dispara la tecla
                if frames_held >= 4:
                    trigger_active = True
                    cooldown = 15  # Tiempo de espera antes de la siguiente pulsación
                    
                    # Interacción con el sistema operativo
                    if current_gesture == "arriba":
                        pyautogui.press('up')
                    elif current_gesture == "abajo":
                        pyautogui.press('down')
                    elif current_gesture == "izquierda":
                        pyautogui.press('left')
                    elif current_gesture == "derecha":
                        pyautogui.press('right')
                    
                    estado_ui.success(f"**Comando enviado:** Flecha {current_gesture.capitalize()}")
            else:
                current_gesture = None
                frames_held = 0

        # Dibujar HUD Visual directamente sobre el frame
        if cooldown > 0:
            cv2.rectangle(frame, (20, 20), (20 + (cooldown * 15), 40), (0, 255, 0), -1)
            cv2.putText(frame, f"ACCION: {current_gesture.upper()}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "LISTO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Convertir BGR a RGB para que Streamlit lo renderice correctamente
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        marco_video.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
