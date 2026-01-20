import streamlit as st
import uuid
# ... (imports)

# UI Text in Spanish / Logic in English
st.set_page_config(page_title="Video RAG Pro", layout="wide")

if 'video_start_time' not in st.session_state:
    print("UI: Initializing video_start_time state") # Internal log
    st.session_state.video_start_time = 0

if 'video_key' not in st.session_state:
    st.session_state.video_key = str(uuid.uuid4())

def seek_video(seconds):
    """Updates the video pointer and forces a widget refresh."""
    print(f"UI: Seeking video to {seconds} seconds") # Internal log
    st.session_state.video_start_time = seconds
    st.session_state.video_key = str(uuid.uuid4())

st.title("📹 Base de Conocimientos de Vídeo")

with st.sidebar:
    st.header("1. Cargar Contenido")
    url = st.text_input("Enlace de YouTube")
    if st.button("Procesar"):
        with st.status("Processing pipeline...", expanded=False) as status:
            print(f"Pipeline: Starting ingestion for {url}") # Internal log
            # ... (Lógica de transcripción y chunking)
            status.update(label="✅ Vídeo Indexado Correctamente", state="complete")

col_vid, col_chat = st.columns([1.2, 0.8])

with col_vid:
    st.subheader("Reproductor")
    if url:
        # Key Hack application
        st.video(url, start_time=st.session_state.video_start_time, key=st.session_state.video_key)

with col_chat:
    st.subheader("Consulta a la IA")
    query = st.text_input("Escribe tu pregunta:")
    
    if query and url:
        print(f"RAG: Processing query '{query}'") # Internal log
        engine = RAGEngine()
        # ... (Lógica de respuesta)
        st.markdown("---")
        st.caption("Saltar al momento:")
        # Botones de salto
        # ...