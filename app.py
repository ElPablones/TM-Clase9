import streamlit as st
import base64

st.set_page_config(page_title="Arpegiador Multimodal", layout="wide")

st.title("🎹 Arpegiador por Gestos")
st.markdown("La detección y el audio se ejecutan nativamente en el navegador vía TensorFlow.js y Tone.js.")

try:
    with open("index.html", "r", encoding="utf-8") as f:
        html_source = f.read()
        
    # Codificamos el HTML en Base64 para inyectarlo sin romper las comillas
    b64_html = base64.b64encode(html_source.encode('utf-8')).decode('utf-8')
    
    # Construimos el iframe manualmente otorgando permisos explícitos (allow="camera")
    iframe_code = f'''
        <iframe 
            src="data:text/html;base64,{b64_html}" 
            allow="camera; microphone; autoplay" 
            width="100%" 
            height="700px" 
            style="border:none; border-radius: 8px;">
        </iframe>
    '''
    
    st.markdown(iframe_code, unsafe_allow_html=True)
    
except FileNotFoundError:
    st.error("No se encontró el archivo index.html en el repositorio.")
