import cv2  # OpenCV library for computer vision tasks
import os  # Operating system interactions
import logging  # Logging for debugging and monitoring
from typing import List  # Type hinting for better code readability

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str, output_dir: str, interval_seconds: int = 30
) -> List[str]:
    """
    Extracts frames from a video file at fixed intervals using a seeking strategy.

    Strategy: Pointer Hopping (Sparse Sampling).
    Why: It provides O(1) performance relative to the skipped frames, avoiding
    unnecessary decoding of the footage between intervals (it would add excessive computational cost).

    Args:
        video_path (str): Absolute or relative path to the input video file.
        output_dir (str): Base directory to save the extracted images.
        interval_seconds (int): Time gap between extractions.

    Returns:
        List[str]: List of file paths for the saved images.
    """

    # 1. Basic Validations
    if not os.path.exists(video_path):
        logger.error(f"Video no encontrado: {video_path}")
        return []

    # Extract video name to create a dedicated subfolder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # 2. Load Video Resource
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(
            "No se pudo abrir el archivo de video (Codec error o archivo corrupto)."
        )
        return []

    # 3. Key Metadata Extraction
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Calculate the jump step in frames
    # Logic: 30 FPS * 30 seconds = 900 frames to skip per iteration
    frame_step = int(fps * interval_seconds)

    if frame_step == 0:
        logger.error("Error matemático: frame_step es 0. Revisa los FPS del video.")
        cap.release()
        return []

    logger.info(f"Procesando: {video_name} | Duración: {duration/60:.2f} min")
    logger.info(
        f"Estrategia: Extracción cada {interval_seconds}s (salto de {frame_step} frames)"
    )

    saved_files = []
    current_frame = 0

    # 4. Fast Extraction Loop
    while current_frame < total_frames:
        # Manually set the video pointer to the desired frame (Seeking). As i said
        # This is much faster than reading frames sequentially (excessive computational cost)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        success, frame = cap.read()

        # If read fails (e.g., end of file or corrupted frame), stop
        if not success:
            break

        # Calculate exact timestamp for filename (Synchronization Key)
        timestamp_sec = int(current_frame / fps)

        # Naming: frame_00030.jpg -> Easy to parse "30 seconds" later
        filename = f"frame_{timestamp_sec:05d}.jpg"
        file_path = os.path.join(video_output_dir, filename)

        # Save with compression optimization (JPEG Quality 80 is standard for ML)
        cv2.imwrite(file_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        saved_files.append(file_path)

        # Jump to the next interval
        current_frame += frame_step

    cap.release()
    logger.info(
        f"✅ Extracción completada: {len(saved_files)} imágenes guardadas en '{video_output_dir}'"
    )
    return saved_files


# --- Execution Block (Testing) ---
if __name__ == "__main__":
    # Test configuration
    # Ensure you have a valid video at this path for testing
    TEST_VIDEO = "data/videos/test_video.mp4"
    OUTPUT_DIR = "data/frames"

    print("--- Iniciando prueba de extracción ---")
    if not os.path.exists(TEST_VIDEO):
        print(
            f"⚠️  AVISO: No encuentro '{TEST_VIDEO}'. \n"
            f"Por favor, descarga un video o cambia la ruta en el bloque 'if __name__' para probar."
        )
    else:
        extract_frames(TEST_VIDEO, OUTPUT_DIR, interval_seconds=30)
