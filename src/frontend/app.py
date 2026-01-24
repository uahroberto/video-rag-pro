import streamlit as st
import uuid
from src.core.transcriber import VideoTranscriber
from src.core.chunking import ChunkingProcessor
from src.database.vector_store import VectorDatabase
from src.core.rag_engine import RAGEngine

# --- CONFIGURATION ---
st.set_page_config(page_title="Video RAG Pro", layout="wide")

# --- HELPER FUNCTIONS ---
def format_time(seconds: float) -> str:
    """Converts seconds to MM:SS format for better UX."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# --- SESSION STATE INITIALIZATION ---
# Created because Streamlit reruns the app.py file on every interaction

# 1. Video Player State
if 'video_start_time' not in st.session_state:
    st.session_state.video_start_time = 0
if 'video_key' not in st.session_state:
    st.session_state.video_key = str(uuid.uuid4())

# 2. Chat History State (New for Modern UI)
# We store the entire conversation history here to emulate a real chatbot experience
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 3. Persistence for Interactive Buttons
# Stores the sources of the LAST answer to keep buttons visible during reruns
if 'current_sources' not in st.session_state:
    st.session_state.current_sources = []

# 4. Raw Transcript Storage
# To enable the 'View Full Transcript' feature without re-processing
if 'full_transcript' not in st.session_state:
    st.session_state.full_transcript = ""

def seek_video(seconds):
    """Updates the video pointer and forces a widget refresh."""
    # Internal log to track user interaction
    print(f"UI: Seeking video to {seconds} seconds") 
    st.session_state.video_start_time = seconds
    st.session_state.video_key = str(uuid.uuid4())

# --- MAIN TITLE ---
st.title("🧠 Video RAG Assistant")

# --- SIDEBAR: CONTENT INGESTION ---
with st.sidebar:
    st.header("1. Cargar Contenido")
    # Using unique keys to prevent input swapping between URL and Chat
    input_url = st.text_input("Enlace de YouTube", key="url_input")

    # Logic Gate: Processing only triggers on button click to save resources
    if st.button("Procesar Vídeo", type="primary"):
        if input_url:
            with st.status("Analizando contenido...", expanded=True) as status:
                # Step 1: Download Audio
                status.write("📥 Descargando audio de YouTube...")
                transcriber = VideoTranscriber() # Created in src/core/transcriber.py
                audio_path, video_title = transcriber.download_audio(input_url)
                
                # Step 2: Transcription (Whisper)
                # Note: Using 'base' model for speed as per Phase 1 optimization
                status.write("⚡ Transcribiendo audio...")
                raw_segments = transcriber.transcribe(audio_path)
                
                # Feature: Save full text for manual inspection
                full_text = " ".join([seg['text'] for seg in raw_segments])
                st.session_state.full_transcript = full_text

                # Step 3: Chunking
                status.write("🧩 Generando fragmentos de conocimiento...")
                processor = ChunkingProcessor()
                chunks = processor.create_chunks(raw_segments)
                
                # Step 4: Vector Storage (Qdrant)
                status.write("💾 Vectorizando en Qdrant...")
                db = VectorDatabase()
                db.upsert_chunks(chunks, input_url)
                
                status.update(label="✅ Vídeo Indexado y Listo", state="complete")
        else:
            st.error("Por favor, introduce una URL válida.")

    # Extra Feature: Raw Text Viewer
    # Useful if the user wants to read instead of chat
    if st.session_state.full_transcript:
        with st.expander("📄 Ver Transcripción Completa"):
            st.text_area("", st.session_state.full_transcript, height=300)

# --- LAYOUT DEFINITION ---
# We split the screen: 60% for video (1.5), 40% for chat (1.0)
col_vid, col_chat = st.columns([1.5, 1.0])

# --- VIDEO COLUMN: Only depends on URL ---
with col_vid:
    # Condition: Only requires a valid URL string to render the player
    if input_url and input_url.startswith("http"):
        # We wrap the video in a container with a dynamic key to force refreshes
        with st.container(key=st.session_state.video_key):
            st.video(input_url, start_time=st.session_state.video_start_time)
    else:
        st.info("👈 Carga un vídeo para comenzar.")

# --- CHAT COLUMN: Modern Chatbot UI ---
with col_chat:
    st.subheader("Chat Inteligente")
    
    # 1. DISPLAY CHAT HISTORY
    # Iterate through session state to render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. PERSISTENT TIMESTAMPS (Interactive Layer)
    # These buttons appear below the history but above the input box.
    # They relate to the LAST answer provided by the AI.
    if st.session_state.current_sources:
        st.caption("📍 Momentos clave relacionados:")
        cols = st.columns(3) # Grid layout for buttons
        for i, seg in enumerate(st.session_state.current_sources):
            # UX: Format time to MM:SS
            time_label = format_time(seg['start'])
            with cols[i % 3]:
                # Logic: Clicking triggers 'seek_video' which refreshes the player
                if st.button(f"▶ {time_label}", key=f"btn_{i}", help="Saltar a este momento"):
                    seek_video(seg['start'])

    # 3. CHAT INPUT (The Trigger)
    # st.chat_input replaces st.form for a more modern feel
    if prompt := st.chat_input("Pregunta sobre el vídeo..."):
        if not input_url:
            st.error("⚠️ Primero debes procesar un vídeo.")
        else:
            # A. Append User Message to History
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # B. Processing Logic
            engine = RAGEngine() # Initialize the engine created in src/rag_engine.py
            
            with st.spinner("La IA está pensando..."):
                # Check for "Summary" intent or standard question
                if "resumen" in prompt.lower() or "summar" in prompt.lower():
                    # Specialized prompt for summarization
                    answer, sources = engine.answer_question("Haz un resumen conciso de los puntos clave", input_url)
                else:
                    # Standard RAG Retrieval
                    answer, sources = engine.answer_question(prompt, input_url)
                
                # C. Append Assistant Message to History
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # D. Update Sources for Buttons (Persistence)
                st.session_state.current_sources = sources
                
                # E. Force Rerun to render the new message and update buttons immediately
                st.rerun()