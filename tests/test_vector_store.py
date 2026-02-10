import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.database.vector_store import VectorDatabase


class TestVectorDatabase(unittest.TestCase):
    def setUp(self):
        # Patch the dependencies: QdrantClient, FastEmbed, SentenceTransformer
        # We need to patch where they are used.
        self.patcher_qdrant = patch("src.database.vector_store.QdrantClient")
        self.MockQdrant = self.patcher_qdrant.start()
        self.MockQdrant.return_value.query_points.return_value.points = []

        self.patcher_fastembed = patch("src.database.vector_store.SparseTextEmbedding")
        self.MockFastEmbed = patch(
            "src.database.vector_store.SparseTextEmbedding"
        ).start()

        self.patcher_sentence = patch("src.database.vector_store.SentenceTransformer")
        self.MockSentence = self.patcher_sentence.start()

        # Initialize DB with mocks
        self.db = VectorDatabase()

    def tearDown(self):
        self.patcher_qdrant.stop()
        self.patcher_fastembed.stop()
        self.patcher_sentence.stop()

    def test_upsert_chunks(self):
        # Setup mocks
        mock_client = self.MockQdrant.return_value

        # Mock embedding generation
        self.MockSentence.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_sparse_vec = MagicMock()
        mock_sparse_vec.indices = np.array([0, 1])
        mock_sparse_vec.values = np.array([0.5, 0.8])

        self.MockFastEmbed.return_value.embed.return_value = [mock_sparse_vec]

        chunks = [{"text": "Test chunk", "metadata": {"source": "audio"}}]

        self.db.upsert_chunks(chunks, "video_123")

        # Verify client.upsert called
        self.assertTrue(mock_client.upsert.called)

        # We can check args if we want precise validation
        call_args = mock_client.upsert.call_args
        self.assertEqual(call_args[1]["collection_name"], "video_knowledge_hybrid")

    def test_search_filtering(self):
        """Verify that search passes the video_id filter to Qdrant."""
        mock_client = self.MockQdrant.return_value

        # Mock embeddings for search query
        self.MockSentence.return_value.encode.return_value = np.array([0.1, 0.2, 0.3])

        mock_sparse_vec = MagicMock()
        mock_sparse_vec.indices = np.array([0, 1])
        mock_sparse_vec.values = np.array([0.5, 0.8])
        self.MockFastEmbed.return_value.embed.return_value = [mock_sparse_vec]

        # Call search with video_id
        target_video_id = "test_vid_abc"
        self.db.search("test query", limit=5, video_id=target_video_id)

        # Check query_points call
        self.assertTrue(mock_client.query_points.called)
        call_kwargs = mock_client.query_points.call_args[1]

        # Inspect prefetch filters
        prefetches = call_kwargs.get("prefetch", [])
        self.assertTrue(len(prefetches) > 0)

        # Retrieve the filter from the first prefetch
        actual_filter = prefetches[0].filter
        self.assertIsNotNone(
            actual_filter, "Filter should not be None when video_id is provided"
        )

        # We can inspect the filter structure if we want deep verification,
        # but verifying it's not None and passed is a good start.
        # Structure: Filter(must=[FieldCondition(key='video_id',
        #   match=MatchValue(value='test_vid_abc'))])

        must_conditions = actual_filter.must
        self.assertTrue(len(must_conditions) > 0)
        self.assertEqual(must_conditions[0].key, "video_id")
        self.assertEqual(must_conditions[0].match.value, target_video_id)

    def test_search_no_filter(self):
        """Verify that search works without video_id (no filter applied)."""
        mock_client = self.MockQdrant.return_value

        # Mock embeddings
        self.MockSentence.return_value.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_sparse_vec = MagicMock()
        mock_sparse_vec.indices = np.array([0, 1])
        mock_sparse_vec.values = np.array([0.5, 0.8])
        self.MockFastEmbed.return_value.embed.return_value = [mock_sparse_vec]

        self.db.search("test query", limit=5, video_id=None)

        self.assertTrue(mock_client.query_points.called)
        call_kwargs = mock_client.query_points.call_args[1]
        prefetches = call_kwargs.get("prefetch", [])

        # Filter should be None
        self.assertIsNone(prefetches[0].filter)
