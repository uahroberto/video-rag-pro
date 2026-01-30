import os
import logging
import re
from typing import List, Dict, Any

# Import existing components
from src.video_processing.frame_extractor import extract_frames
from src.video_processing.ocr_service import OCRService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VisualIngestionService:
    """
    Orchestrator service that manages the full visual indexing pipeline:
    1. Extracts frames from video.
    2. Runs OCR on frames.
    3. Structures data for Qdrant ingestion.
    """

    def __init__(self):
        """
        Initializes the service with necessary sub-components.
        """
        self.ocr_service = OCRService()
        # Base directory for temporary frame storage
        self.frames_base_dir = "data/frames"

    def process_video(
        self, video_path: str, video_id: str, interval: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Runs the visual pipeline and returns a list of 'documents' ready for embedding.

        Args:
            video_path (str): Path to the .mp4 file.
            video_id (str): Unique identifier for the video (to link in Qdrant).
            interval (int): Seconds between frames.

        Returns:
            List[Dict]: A list of payloads ready for the Vector Store.
            Structure example:
            [
                {
                    "page_content": "def my_func(): return True...",
                    "metadata": {
                        "video_id": "123",
                        "timestamp": 30.0,
                        "source_type": "visual",
                        "frame_path": "data/frames/..."
                    }
                },
                ...
            ]
        """
        logger.info(f"🎬 Starting visual ingestion for video: {video_id}")

        # 1. Extraction Phase: Get images from video
        frame_paths = extract_frames(
            video_path, self.frames_base_dir, interval_seconds=interval
        )

        if not frame_paths:
            logger.warning("No frames extracted. Check video path or codec.")
            return []

        visual_documents = []

        # 2. Analysis Phase: Run OCR on each extracted frame
        logger.info(f"👁️ Analyzing {len(frame_paths)} frames with OCR...")

        for frame_path in frame_paths:
            # Extract text using the OCR service
            text = self.ocr_service.extract_text(frame_path)

            # Optimization: Skip empty frames to save DB space and reduce noise
            if not text.strip():
                continue

            # Calculate timestamp from filename (e.g., frame_00030.jpg -> 30.0)
            timestamp = self._parse_timestamp_from_filename(frame_path)

            # 3. Structuring Phase: Create the payload
            # We create a dictionary compatible with LangChain/Qdrant schemas
            doc = {
                "page_content": text,  # This is the text that will be embedded (Vectorized)
                "metadata": {
                    "video_id": video_id,
                    "timestamp": float(timestamp),
                    "source_type": "visual",  # Crucial for filtering (Audio vs Visual)
                    "frame_path": frame_path,  # Useful for displaying the source image in the UI
                },
            }
            visual_documents.append(doc)

        logger.info(
            f"✅ Visual processing complete. Generated {len(visual_documents)} searchable visual chunks."
        )
        return visual_documents

    def _parse_timestamp_from_filename(self, filename: str) -> float:
        """
        Helper method to extract seconds from filename 'frame_00030.jpg'.
        Returns 0.0 if parsing fails.
        """
        try:
            basename = os.path.basename(filename)
            # Regex to capture the digits after 'frame_'
            match = re.search(r"frame_(\d+)", basename)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception:
            return 0.0


# --- MAIN EXECUTION BLOCK (FOR TESTING PURPOSES) ---
if __name__ == "__main__":
    # Test configuration
    TEST_VIDEO = "data/videos/test_video.mp4"
    TEST_ID = "video_test_001"

    print("🚀 Initializing Visual Ingestion Service...")
    service = VisualIngestionService()

    if os.path.exists(TEST_VIDEO):
        docs = service.process_video(TEST_VIDEO, TEST_ID, interval=30)

        print("\n--- SAMPLE VISUAL CHUNKS (Ready for Qdrant) ---")
        # Display the first 3 chunks to verify structure
        for i, doc in enumerate(docs[:3]):
            print(f"Chunk #{i+1} (Time: {doc['metadata']['timestamp']}s):")
            print(f"Text Preview: {doc['page_content'][:100]}...")
            print("-" * 20)
    else:
        print(f"⚠️ Test video not found at: {TEST_VIDEO}")
