import streamlit as st
import uuid
import os 
from src.core.transcriber import VideoTranscriber
from src.core.chunking import ChunkingProcessor
from src.database.vector_store import VectorDatabase
from src.core.rag_engine import RAGEngine

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
        with st.status("Procesando vídeo...", expanded=True) as status:
            # 1. Transcription (Whisper)
            status.write("👂 Transcribiendo audio...")
            transcriber = VideoTranscriber()
            raw_text = transcriber.transcribe(url)
        
            # 2. Chunking
            status.write("✂️ Fragmentando contenido...")
            processor = ChunkingProcessor()
            chunks = processor.create_chunks(raw_text)
        
            # 3. Vector Storage (Qdrant)
            status.write("🧠 Guardando en base de datos vectorial...")
            db = VectorDatabase()
            db.upsert_chunks(chunks, url)
        
            status.update(label="✅ Vídeo Indexado y Listo", state="complete")

# --- LAYOUT DEFINITION ---
# We split the screen: 60% for video (1.2), 40% for chat (0.8)
col_vid, col_chat = st.columns([1.2, 0.8])

with col_vid:
    st.subheader("Reproductor Inteligente")
    if url and url.startswith("http"):
        # key hack is applied to the container, not the video
        with st.container(key=st.session_state.video_key):
            st.video(
                url, 
                start_time=st.session_state.video_start_time
                # key is removed to avoid TypeError
            )
    else:
        st.info("Introduce una URL válidapara comenzar.")


with col_chat:
    st.subheader("Consulta a la IA")
    query = st.text_input("Escribe tu pregunta:")
    
    if query and url:
        print(f"RAG: Processing query '{query}'") # Internal log
    
    # Initialize the engine
    engine = RAGEngine()
    
    with st.spinner("La IA está analizando el vídeo..."):
        # 1. Execute the RAG pipeline
        # We use 'url' as the temporary video_id for simplicity
        answer, sources = engine.answer_question(query, url)
        
        # 2. Display the AI response
        st.markdown("### Respuesta:")
        st.write(answer)
        
        st.markdown("---")
        st.caption("Fuentes y momentos clave:")
        
        # 3. Create interactive jump buttons
        # We use a grid or columns for a cleaner UI
        cols = st.columns(2)
        for i, seg in enumerate(sources):
            col_idx = i % 2
            with cols[col_idx]:
                # Button text in Spanish
                if st.button(f"⏱️ Seg. {seg['start']:.1f}", key=f"btn_{i}"):
                    seek_video(seg['start'])