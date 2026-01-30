import os
import argparse
import logging

# Pointing to the correct location in src/core
from src.core.transcriber import VideoTranscriber
from src.services.visual_ingestion import VisualIngestionService
from src.database.vector_store import VectorDatabase

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(video_url: str, video_id: str):
    """
    Main Orchestration Pipeline for Multimodal RAG Ingestion.
    """
    logger.info(f"🚀 STARTING MULTIMODAL INGESTION PIPELINE FOR: {video_id}")

    # 1. Initialize Components
    transcriber = VideoTranscriber()
    visual_service = VisualIngestionService()
    db = VectorDatabase()

    # Define Paths
    video_path = f"data/videos/{video_id}.mp4"
    audio_path = f"data/tmp/{video_id}.mp3"

    # Ensure directories exist
    os.makedirs("data/videos", exist_ok=True)
    os.makedirs("data/tmp", exist_ok=True)

    # --- PHASE 1: AUDIO PROCESSING ---
    logger.info("\n--- 🔊 PHASE 1: AUDIO PROCESSING ---")

    # A. Download Audio
    if not os.path.exists(audio_path):
        logger.info(f"📥 Downloading audio from {video_url}...")
        try:
            # FIX: Capture the actual downloaded filename (usually YouTubeID.mp3)
            downloaded_path, _ = transcriber.download_audio(
                video_url, output_path="data/tmp"
            )

            # FIX: Rename it to match our internal video_id
            if downloaded_path != audio_path:
                logger.info(f"🔄 Renaming {downloaded_path} to {audio_path}...")
                os.rename(downloaded_path, audio_path)

        except Exception as e:
            logger.error(f"❌ Failed to download audio: {e}")
            return
    else:
        logger.info("⏩ Audio file already exists. Skipping download.")

    # B. Transcribe Audio
    logger.info(f"🎙️ Transcribing audio from {audio_path}...")
    try:
        # Now it will definitely find the file
        audio_chunks = transcriber.transcribe(audio_path)
        logger.info(f"✅ Generated {len(audio_chunks)} audio segments.")

        # C. Index Audio
        logger.info("💾 Indexing Audio Chunks into Qdrant...")
        db.upsert_chunks(audio_chunks, video_id)
    except Exception as e:
        logger.error(f"❌ Audio processing failed: {e}")

    # --- PHASE 2: VISUAL PROCESSING ---
    logger.info("\n--- 👁️ PHASE 2: VISUAL PROCESSING ---")

    if os.path.exists(video_path):
        logger.info(f"🎞️ Processing video file: {video_path}")
        try:
            visual_chunks = visual_service.process_video(
                video_path, video_id, interval=30
            )

            if visual_chunks:
                logger.info(f"✅ Generated {len(visual_chunks)} visual chunks.")
                logger.info("💾 Indexing Visual Chunks into Qdrant...")
                db.upsert_chunks(visual_chunks, video_id)
            else:
                logger.warning("⚠️ No text found in video frames.")
        except Exception as e:
            logger.error(f"❌ Visual processing failed: {e}")
    else:
        logger.warning(f"⚠️ VIDEO FILE NOT FOUND: {video_path}")
        logger.warning("Skipping visual ingestion.")

    logger.info(f"\n✅ PIPELINE FINISHED FOR {video_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a video into the Multimodal RAG System."
    )
    parser.add_argument(
        "--url", type=str, required=True, help="YouTube URL of the video"
    )
    parser.add_argument("--id", type=str, required=True, help="Unique ID for the video")

    args = parser.parse_args()

    main(args.url, args.id)
