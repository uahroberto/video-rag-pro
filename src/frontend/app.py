import streamlit as st
import uuid
from src.core.transcriber import VideoTranscriber
from src.core.chunking import ChunkingProcessor
from src.database.vector_store import VectorDatabase
from src.core.rag_engine import RAGEngine

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Video RAG Assistant", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" # Force sidebar open for video player
)

# --- HELPER FUNCTIONS ---
def format_time(seconds: float) -> str:
    """Converts seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# --- SESSION STATE ---
if 'video_start_time' not in st.session_state:
    st.session_state.video_start_time = 0
if 'should_autoplay' not in st.session_state:
    st.session_state.should_autoplay = False
if 'video_key' not in st.session_state:
    st.session_state.video_key = str(uuid.uuid4())

# HISTORY STRUCTURE CHANGE:
# We now store: {'role': '...', 'content': '...', 'sources': [...]} 
# This allows rendering buttons for OLD messages too.
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'full_transcript' not in st.session_state:
    st.session_state.full_transcript = ""

def seek_video(seconds):
    """Updates the video pointer and forces a widget refresh."""
    st.session_state.video_start_time = int(seconds)
    st.session_state.video_key = str(uuid.uuid4())
    # Crucial: We don't want to lose the chat history on rerun
    # (Streamlit handles session_state persistence automatically)
    st.session_state.should_autoplay = True
    st.rerun()

# --- MAIN LAYOUT ---

# 1. SIDEBAR: VIDEO PLAYER & CONTROLS (The "Sticky" Element)
with st.sidebar:
    st.title("📺 Panel de Control")
    
    # Input Section
    input_url = st.text_input("URL de YouTube", key="url_input")
    
    if st.button("🚀 Procesar Vídeo", type="primary", use_container_width=True):
        if input_url:
            with st.status("Analizando vídeo...", expanded=True) as status:
                transcriber = VideoTranscriber()
                status.write("📥 Descargando...")
                audio_path, video_title = transcriber.download_audio(input_url)
                
                status.write("⚡ Transcribiendo (Modo Rápido)...")
                raw_segments = transcriber.transcribe(audio_path)
                
                # Save full text for the 'Copy' feature
                st.session_state.full_transcript = " ".join([s['text'] for s in raw_segments])

                processor = ChunkingProcessor()
                chunks = processor.create_chunks(raw_segments)
                
                db = VectorDatabase()
                db.upsert_chunks(chunks, input_url)
                
                status.update(label="✅ Indexado Completo", state="complete")
        else:
            st.error("URL no válida")

    st.markdown("---")
    
    # VIDEO PLAYER (Fixed in Sidebar)
    # This solves the issue of the video scrolling away
    if input_url and input_url.startswith("http"):
        st.subheader("Reproductor")
        with st.container(key=st.session_state.video_key):
            # autoplay=True lets the video start playing as soon as the timestamp button is clicked
            st.video(
                input_url, 
                start_time=st.session_state.video_start_time, 
                autoplay=st.session_state.should_autoplay
            )

            # If autoplay is enabled, disable it and rerun to prevent constant autoplay
            if st.session_state.should_autoplay:
                st.session_state.should_autoplay = False
    # TRANSCRIPT FEATURE (Better UI)
    if st.session_state.full_transcript:
        with st.expander("📄 Ver Transcripción"):
           # UX FIX: Replaced st.code with st.text_area to avoid horizontal scrolling.
            # height=400 gives a good reading viewport.
            st.text_area(
                "Texto Completo", 
                st.session_state.full_transcript, 
                height=400,
                help="Puedes copiar el texto seleccionándolo."
            )

# 2. MAIN AREA: CHAT INTERFACE
st.title("🧠 Asistente de Vídeo")

# Render History
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # TIMESTAMPS LOGIC (Integrated in history)
        # Check if this specific message has associated sources
        if "sources" in message and message["sources"]:
            st.caption("📍 Momentos clave:")
            # We use a small grid for buttons
            cols = st.columns(4)
            for j, seg in enumerate(message["sources"]):
                time_label = format_time(seg['start'])
                # Unique key is essential: msg_index + btn_index
                if cols[j % 4].button(f"▶ {time_label}", key=f"hist_{i}_{j}"):
                    seek_video(seg['start'])

# Chat Input (Always at bottom)
if prompt := st.chat_input("Pregunta sobre el vídeo..."):
    if not input_url:
        st.toast("⚠️ Por favor, carga un vídeo primero.", icon="🚨")
    else:
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Assistant Response
        engine = RAGEngine()
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                answer, sources = engine.answer_question(prompt, input_url)
                st.markdown(answer)
                
                # Render buttons immediately for the new response
                if sources:
                    st.caption("📍 Momentos clave:")
                    cols = st.columns(4)
                    for j, seg in enumerate(sources):
                        time_label = format_time(seg['start'])
                        if cols[j % 4].button(f"▶ {time_label}", key=f"new_{j}"):
                            seek_video(seg['start'])

        # 3. Save to History (Including Sources!)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources # Persist sources for future re-renders
        })
        
        # No st.rerun() needed here usually, but if buttons don't work on first click, ill  add it.
        # st.rerun()