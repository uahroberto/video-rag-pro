import unittest
from unittest.mock import MagicMock, patch
from src.video_processing.ocr_service import OCRService


class TestOCRService(unittest.TestCase):
    def setUp(self):
        # Patch RapidOCR initialization
        self.patcher = patch("src.video_processing.ocr_service.RapidOCR")
        self.MockRapidOCR = self.patcher.start()

        self.ocr_service = OCRService()

    def tearDown(self):
        self.patcher.stop()

    @patch("src.video_processing.ocr_service.os.path.exists")
    def test_extract_text_success(self, mock_exists):
        mock_exists.return_value = True
        mock_instance = self.MockRapidOCR.return_value

        # Mock prediction result structure (Object with txts and scores attributes)
        MockPrediction = MagicMock()
        MockPrediction.txts = ["Detected Text", "Line 2"]
        MockPrediction.scores = [0.95, 0.88]

        mock_instance.return_value = MockPrediction

        text = self.ocr_service.extract_text("dummy.jpg")

        self.assertIn("Detected Text", text)
        self.assertIn("Line 2", text)

    def test_extract_text_empty(self):
        self.MockRapidOCR.return_value.return_value = None
        text = self.ocr_service.extract_text("dummy.jpg")
        self.assertEqual(text, "")
