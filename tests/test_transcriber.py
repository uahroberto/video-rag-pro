import unittest
from unittest.mock import MagicMock, patch
from src.core.transcriber import VideoTranscriber


class TestVideoTranscriber(unittest.TestCase):
    def setUp(self):
        # We patch the WhisperModel class where it is IMPORTED in the transcriber
        self.patcher = patch("src.core.transcriber.WhisperModel")
        self.MockWhisperModel = self.patcher.start()

        # Instantiate transcriber
        self.transcriber = VideoTranscriber()

    def tearDown(self):
        self.patcher.stop()

    @patch("src.core.transcriber.yt_dlp.YoutubeDL")
    def test_download_audio_success(self, mock_ytdl):
        # Setup mock behavior
        mock_instance = mock_ytdl.return_value.__enter__.return_value
        mock_instance.extract_info.return_value = {
            "id": "test_id",
            "title": "Test Title",
        }
        mock_instance.prepare_filename.return_value = "data/tmp/test_id.mp3"  # Simulated path

        # This method in source uses 'output_path' argument for directory,
        # but internal logic might construct filename.
        # Let's trust logic but mock os.path interactions if mostly external.

        path, title = self.transcriber.download_audio("http://youtube.com/test")

        self.assertEqual(title, "Test Title")
        # The transcriber returns f"{output_path}/{info['id']}.mp3"
        self.assertTrue(path.endswith("test_id.mp3"))

    def test_transcribe_segments(self):
        # Logic:
        # 1. Mock self.transcriber.model.transcribe
        # 2. It returns (segments, info)

        MockSegment = MagicMock()
        MockSegment.start = 0.0
        MockSegment.end = 5.0
        MockSegment.text = "Hello world"

        self.transcriber.model.transcribe.return_value = ([MockSegment], None)

        results = self.transcriber.transcribe("dummy/path.mp3")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Hello world")
        self.assertEqual(results[0]["start"], 0.0)
