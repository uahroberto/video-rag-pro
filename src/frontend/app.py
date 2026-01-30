import streamlit as st
import uuid
import os
import sys
import yt_dlp

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
# Añadimos '# noqa: E402' para decir al linter: "Ignora esto, sé lo que hago"
from src.core.transcriber import VideoTranscriber  # noqa: E402
from src.database.vector_store import VectorDatabase  # noqa: E402
from src.core.rag_engine import RAGEngine  # noqa: E402
from src.services.visual_ingestion import VisualIngestionService  # noqa: E402

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Video RAG Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- HELPER FUNCTIONS ---
def format_time(seconds: float) -> str:
    """Converts seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def download_video_files(url: str, video_id: str) -> tuple[str, str, str]:
    """
    Downloads BOTH Video (.mp4) and Audio (.mp3).
    Includes Anti-Bot measures.
    """
    video_dir = "data/videos"
    audio_dir = "data/tmp"
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # 1. Download Video (MP4)
    video_path = f"{video_dir}/{video_id}.mp4"

    # ANTI-BOT CONFIGURATION
    common_opts = {
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "source_address": "0.0.0.0",
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
                "player_skip": ["webpage", "configs", "js"],
            }
        },
    }

    ydl_opts_video = {
        **common_opts,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": video_path,
    }

    # 2. Download Audio (MP3)
    audio_path = f"{audio_dir}/{video_id}.mp3"
    ydl_opts_audio = {
        **common_opts,
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": f"{audio_dir}/{video_id}.%(ext)s",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
            if not os.path.exists(video_path):
                ydl.download([url])
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "Video")

        with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
            if not os.path.exists(audio_path):
                ydl.download([url])

        return video_path, audio_path, title

    except Exception as e:
        raise e


# --- SESSION STATE ---
if "video_start_time" not in st.session_state:
    st.session_state.video_start_time = 0
if "should_autoplay" not in st.session_state:
    st.session_state.should_autoplay = False
if "video_key" not in st.session_state:
    st.session_state.video_key = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = ""


def seek_video(seconds):
    st.session_state.video_start_time = int(seconds)
    st.session_state.video_key = str(uuid.uuid4())
    st.session_state.should_autoplay = True
    st.rerun()


# --- MAIN LAYOUT ---

# 1. SIDEBAR
with st.sidebar:
    st.title("📺 Panel de Control")

    input_id = st.text_input(
        "ID del Vídeo (o URL)",
        value="",
        key="url_input",
        help="Introduce una URL de YouTube o un ID corto.",
    )

    if st.button(
        "🚀 Procesar Vídeo Completo (Web)", type="primary", use_container_width=True
    ):
        if input_id:
            with st.status(
                "🏗️ Iniciando Pipeline Multimodal...", expanded=True
            ) as status:
                # A. SETUP
                transcriber = VideoTranscriber()
                visual_service = VisualIngestionService()
                db = VectorDatabase()

                if input_id.startswith("http"):
                    target_url = input_id
                    process_id = "web_download"
                else:
                    target_url = None
                    process_id = input_id

                # B. DOWNLOAD
                status.write("📥 Descargando Archivos (Audio + Video)...")
                if target_url:
                    video_path, audio_path, title = download_video_files(
                        target_url, process_id
                    )
                else:
                    video_path = f"data/videos/{process_id}.mp4"
                    audio_path = f"data/tmp/{process_id}.mp3"
                    title = process_id

                if not os.path.exists(video_path) or not os.path.exists(audio_path):
                    status.update(label="❌ Faltan archivos locales", state="error")
                    st.error(f"No encuentro {video_path} o {audio_path}.")
                    st.stop()

                # C. AUDIO PROCESSING
                status.write("🎙️ Transcribiendo Audio (Whisper)...")
                audio_chunks = transcriber.transcribe(audio_path)
                st.session_state.full_transcript = " ".join(
                    [s["text"] for s in audio_chunks]
                )

                status.write(
                    f"💾 Guardando {len(audio_chunks)} fragmentos de audio en Qdrant..."
                )
                db.upsert_chunks(audio_chunks, process_id)

                # D. VISUAL PROCESSING
                status.write("👁️ Analizando Vídeo (OCR / Pantalla)...")
                # Interval 5s for HIGH RESOLUTION scanning (Fixes the missed code issue)
                visual_chunks = visual_service.process_video(
                    video_path, process_id, interval=5
                )

                if visual_chunks:
                    status.write(
                        f"💾 Guardando {len(visual_chunks)} fragmentos visuales en Qdrant..."
                    )
                    db.upsert_chunks(visual_chunks, process_id)
                else:
                    status.write("⚠️ No se detectó texto relevante en el vídeo.")

                status.update(
                    label="✅ Ingesta Multimodal Completada", state="complete"
                )
                st.session_state.video_key = str(uuid.uuid4())

        else:
            st.error("Por favor, introduce una URL o ID.")

    st.markdown("---")

    # VIDEO PLAYER
    if input_id:
        if input_id.startswith("http"):
            video_file_path = "data/videos/web_download.mp4"
            if os.path.exists(video_file_path):
                player_source = video_file_path
            else:
                player_source = input_id
        else:
            video_file_path = f"data/videos/{input_id}.mp4"
            player_source = video_file_path if os.path.exists(video_file_path) else ""

        st.subheader("Reproductor")
        if player_source:
            with st.container(key=st.session_state.video_key):
                st.video(
                    player_source,
                    start_time=st.session_state.video_start_time,
                    autoplay=st.session_state.should_autoplay,
                )
        else:
            st.info("Esperando vídeo...")

        if st.session_state.should_autoplay:
            st.session_state.should_autoplay = False

    if st.session_state.full_transcript:
        with st.expander("📄 Ver Transcripción"):
            st.text_area("Texto Completo", st.session_state.full_transcript, height=400)

# 2. MAIN CHAT AREA
st.title("🧠 Asistente Multimodal")

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "sources" in message and message["sources"]:
            st.divider()
            st.caption("🔍 Fuentes Consultadas:")
            cols = st.columns(3)
            for j, seg in enumerate(message["sources"]):
                col = cols[j % 3]
                with col:
                    time_label = format_time(seg["start"])
                    source_type = seg.get("type", "audio")

                    if source_type == "visual":
                        st.markdown(f"**📸 Visual ({time_label})**")
                        frame_path = seg.get("frame_path")
                        if frame_path and os.path.exists(frame_path):
                            st.image(frame_path, use_container_width=True)
                        st.code(seg.get("text", "")[:60] + "...", language="text")
                    else:
                        st.markdown(f"**🎙️ Audio ({time_label})**")
                        st.info(f"\"{seg.get('text', '')[:80]}...\"")

                    if st.button(
                        f"▶ Ir al min {time_label}",
                        key=f"hist_btn_{i}_{j}",
                        use_container_width=True,
                    ):
                        seek_video(seg["start"])

# Chat Input
if prompt := st.chat_input("Pregunta sobre el vídeo..."):
    # FIX FOR MYPY: Ensure explicit string type
    query_id = (
        "web_download"
        if input_id and input_id.startswith("http")
        else str(input_id or "")
    )

    if not query_id:
        st.toast("⚠️ Primero procesa un vídeo.", icon="🚨")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        engine = RAGEngine()
        with st.chat_message("assistant"):
            with st.spinner("🧠 Cerebro Multimodal Pensando..."):
                answer, sources = engine.answer_question(prompt, query_id)
                st.markdown(answer)

                if sources:
                    st.divider()
                    st.caption("🔍 Fuentes encontradas:")
                    cols = st.columns(3)
                    for j, seg in enumerate(sources):
                        col = cols[j % 3]
                        with col:
                            time_label = format_time(seg["start"])
                            source_type = seg.get("type", "audio")

                            if source_type == "visual":
                                st.markdown(f"**📸 Visual ({time_label})**")
                                frame_path = seg.get("frame_path")
                                if frame_path and os.path.exists(frame_path):
                                    st.image(frame_path, use_container_width=True)
                                st.code(
                                    seg.get("text", "")[:60] + "...", language="text"
                                )
                            else:
                                st.markdown(f"**🎙️ Audio ({time_label})**")
                                st.info(f"\"{seg.get('text', '')[:80]}...\"")

                            if st.button(
                                f"▶ Ir al min {time_label}",
                                key=f"new_btn_{j}",
                                use_container_width=True,
                            ):
                                seek_video(seg["start"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
