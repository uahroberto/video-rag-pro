import logging
import os
from rapidocr import RapidOCR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OCRService:
    """
    Service responsible for extracting text from images using RapidOCR (ONNX).
    Optimized for CPU usage and technical text detection (code, slides).
    """

    def __init__(self) -> None:
        """
        Initializes the OCR engine.
        We load the model once during instantiation to avoid overhead on every call.
        """
        try:
            self.engine = RapidOCR()
            logger.info("✅ OCR Engine initialized successfully (RapidOCR/ONNX)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OCR Engine: {e}")
            self.engine = None

    def extract_text(self, image_path: str) -> str:
        """
        Extracts clean text from a given image path.

        Handles the specific RapidOCROutput object structure (txts/scores
        attributes).

        Args:
            image_path (str): Absolute or relative path to the .jpg file.

        Returns:
            str: Combined text found in the image, joined by newlines.
                Returns empty string on failure.
        """
        # 1. Guard Clauses
        if not self.engine:
            logger.error("OCR Engine is not running.")
            return ""

        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return ""

        try:
            # 2. Inference
            prediction = self.engine(image_path)

            # 3. Data Extraction (Adapter for RapidOCROutput object)
            # Returns an object with separate attributes for text and confidence scores.
            # Use getattr to safely access them
            raw_texts = getattr(prediction, "txts", [])
            raw_scores = getattr(prediction, "scores", [])

            # Edge case: If the model returns None or empty lists
            if not raw_texts:
                return ""

            detected_texts = []

            # 4. Filtering and Merging
            # Use zip to iterate over text and confidence simultaneously
            for text, score in zip(raw_texts, raw_scores):
                # Ensure confidence is a float
                confidence = float(score)

                # Quality Filter:
                # - Confidence > 0.6: Filters out noise/hallucinations.
                # - Length > 1: Filters out single stray characters.
                if confidence > 0.6 and len(str(text).strip()) > 1:
                    detected_texts.append(str(text))

            # Join with newlines to preserve vertical structure
            full_text = "\n".join(detected_texts)
            return full_text

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            logger.exception("Traceback details:")
            return ""


# --- MAIN EXECUTION BLOCK (FOR TESTING PURPOSES) ---
if __name__ == "__main__":
    # Test configuration: Ensure this file exists before running
    TEST_IMAGE = "data/frames/test_video/frame_00570.jpg"

    print("🚀 Starting OCR Service...")
    ocr = OCRService()

    print(f"📄 Analyzing image: {TEST_IMAGE}")
    if os.path.exists(TEST_IMAGE):
        text_result = ocr.extract_text(TEST_IMAGE)
        print("-" * 40)
        print("DETECTED TEXT:")
        print(text_result)
        print("-" * 40)
    else:
        print(f"❌ ERROR: Image {TEST_IMAGE} does not exist. Please check the path.")
