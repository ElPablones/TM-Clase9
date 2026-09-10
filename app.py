import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Arpegiador Multimodal", layout="wide")

st.title("🎹 Arpegiador por Gestos")
st.markdown("La detección y el audio se ejecutan nativamente en el navegador vía TensorFlow.js y Tone.js.")

# Abrir y leer el archivo HTML local
try:
    with open("index.html", "r", encoding="utf-8") as f:
        html_source = f.read()
        
    # Incrustar el código en la interfaz de Streamlit
    components.html(
        html_source, 
        height=650, 
        scrolling=True
    )
except FileNotFoundError:
    st.error("No se encontró el archivo index.html en el repositorio.")
