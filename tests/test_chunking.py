import unittest
from src.core.chunking import ChunkingProcessor


class TestChunkingProcessor(unittest.TestCase):
    def setUp(self):
        self.chunker = ChunkingProcessor(min_chunk_size=10)  # Small size for testing

    def test_create_chunks_basic(self):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello"},
            {"start": 2.0, "end": 4.0, "text": "World"},
        ]

        chunks = self.chunker.create_chunks(segments)
        # Should combine if logic aims for min_chunk_size=10
        # "Hello" (5) + "World" (5) = "Hello World" (11 chars with space?)

        self.assertTrue(len(chunks) >= 1)
        self.assertIn("Hello", chunks[0]["text"])

    def test_create_chunks_empty(self):
        chunks = self.chunker.create_chunks([])
        self.assertEqual(len(chunks), 0)
