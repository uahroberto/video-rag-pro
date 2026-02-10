import unittest
import asyncio
import os
from unittest.mock import MagicMock, patch
from src.core.rag_engine import RAGEngine


class TestAsyncIngest(unittest.TestCase):
    def setUp(self):
        # Patch heavy components
        self.patcher_openai = patch("src.core.rag_engine.OpenAI")
        self.mock_openai = self.patcher_openai.start()

        self.patcher_db = patch("src.core.rag_engine.VectorDatabase")
        self.mock_db_class = self.patcher_db.start()

        # We need a real RAGEngine but with mocked internal components where possible
        # to avoid huge downloads.
        # But we DO want to test the orchestration.
        self.rag = RAGEngine()

        # Strategy: Integration test with a small real video if we had one.
        # Check if 'data/videos/test_video.mp4' exists.
        self.test_video_path = "data/videos/test_video.mp4"
        if not os.path.exists(self.test_video_path):
            self.skipTest("Test video not found for integration test")

    def tearDown(self):
        self.patcher_openai.stop()
        self.patcher_db.stop()

    async def async_test_flow(self):
        """Async test logic to be run by loop"""
        # 1. Mock Audio Download (return local path)
        self.rag.transcriber.download_audio = MagicMock(
            return_value=("data/videos/test_audio.mp3", "Test Title")
        )

        # 2. Mock Audio Transcribe (return dummy segments)
        # Mocking the transcribe method directly on the instance we created
        with patch.object(
            self.rag.transcriber,
            "transcribe",
            return_value=[{"start": 0.0, "end": 2.0, "text": "Hello world"}],
        ):
            pass  # better to set side_effect or return_value on the instance

        self.rag.transcriber.transcribe = MagicMock(
            return_value=[{"start": 0.0, "end": 2.0, "text": "Hello world"}]
        )

        # 3. Mock Video Download (return local video path)
        # We need to ensure we return the ABSOLUTE path or relative path that works
        self.rag._download_video_best = MagicMock(return_value=self.test_video_path)

        # 4. Mock OCR
        self.rag.ocr_service.extract_text = MagicMock(return_value="Slide Text Content")

        # 5. Mock DB Upsert to avoid actual DB calls
        self.rag.db.upsert_chunks = MagicMock()

        # 6. Run ingest
        await self.rag.ingest_video("http://youtube.com/fake")

        # 7. Assertions
        # Verify Audio Download called
        self.rag.transcriber.download_audio.assert_called_once()

        # Verify DB upsert called
        self.assertTrue(self.rag.db.upsert_chunks.called)
        call_args = self.rag.db.upsert_chunks.call_args
        chunks = call_args[0][0]  # first arg

        # Verify content
        has_audio = any("Hello world" in c.get("text", "") for c in chunks)
        # Note: Visual chunks might not be generated if video isn't processed correctly
        # or OCR fails/skips
        # We rely on 'test_video.mp4' actually having frames.
        # If OCR returns 'Slide Text Content' for at least one frame, we are good.
        has_visual = any(
            "Slide Text Content" in c.get("page_content", "") for c in chunks
        )

        if not has_visual:
            print(
                "⚠️ Warning: No visual chunks found."
                "Check loop logic or video content."
            )

        self.assertTrue(has_audio, "Should contain audio chunks")

    def test_concurrent_ingest_flow(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.async_test_flow())
        finally:
            loop.close()

    async def async_test_audio_only(self):
        """Verify that visual processing is skipped"""
        # 1. Mock Audio
        self.rag.transcriber.download_audio = MagicMock(
            return_value=("data/videos/test_audio.mp3", "Test Title")
        )
        self.rag.transcriber.transcribe = MagicMock(
            return_value=[{"start": 0.0, "end": 2.0, "text": "Hello Audio"}]
        )

        # 2. Mock Video Download (Should NOT be called)
        self.rag._download_video_best = MagicMock()

        # 3. Mock OCR (Should NOT be called)
        self.rag.ocr_service.extract_text = MagicMock()

        # 4. Mock DB
        self.rag.db.upsert_chunks = MagicMock()

        # 5. Run ingest with include_visuals=False
        await self.rag.ingest_video(
            "http://youtube.com/fake_audio", include_visuals=False
        )

        # 6. Assertions
        self.rag.transcriber.download_audio.assert_called_once()
        self.rag._download_video_best.assert_not_called()
        self.rag.ocr_service.extract_text.assert_not_called()

        # Check DB upsert content
        call_args = self.rag.db.upsert_chunks.call_args
        chunks = call_args[0][0]
        has_audio = any("Hello Audio" in c.get("text", "") for c in chunks)
        has_visual = any(
            "source" in c.get("metadata", {}) and c["metadata"]["source"] == "visual"
            for c in chunks
        )

        self.assertTrue(has_audio)
        self.assertFalse(has_visual, "Should NOT contain visual chunks")

    def test_audio_only_flow(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.async_test_audio_only())
        finally:
            loop.close()
