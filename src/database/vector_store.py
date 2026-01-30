import os
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv

load_dotenv()


class VectorDatabase:
    """
    Manages Hybrid Search (Dense + Sparse) interaction with Qdrant.
    Updated to support Multimodal ingestion (Audio + Visual payloads).

    Architecture:
    - Dense Vector: 'all-MiniLM-L6-v2' (Semantic understanding)
    - Sparse Vector: 'Qdrant/bm25' (Keyword matching)
    """

    def __init__(self, collection_name: str = "video_knowledge_hybrid"):
        """
        Initializes the database connection and loads embedding models.
        """
        self.collection_name = collection_name

        # 1. Load Dense Model (Semantic)
        # Why? Allows searching by meaning ("how to loop" finds "while True")
        print("🤖 Loading Dense model (all-MiniLM-L6-v2)...")
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")

        # 2. Load Sparse Model (Keywords/BM25)
        # Why? Allows searching by exact terms ("error 404", variable names)
        print("🤖 Loading Sparse model (Qdrant/bm25)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

        print(f"🌐 Database Mode: Client-Server ({qdrant_host}:{qdrant_port})")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

        self._ensure_collection()

    def _ensure_collection(self):
        """
        Ensures the collection exists with Hybrid Schema (Dense + Sparse configuration).
        """
        if not self.client.collection_exists(self.collection_name):
            print(f"🛠️ Creating Hybrid Collection: {self.collection_name}")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text-dense": models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )
        else:
            print(f"✅ Collection '{self.collection_name}' ready.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], video_id: str):
        """
        Generates vectors and uploads them in batch.

        POLYMORPHISM EXPLAINED:
        This method acts as an Adapter. It accepts data in two shapes:
        1. Audio Chunks (Flat dict): {'text': '...', 'start': 0.0}
        2. Visual Chunks (Nested dict): {'page_content': '...', 'metadata': {...}}

        It normalizes both into a standard Qdrant Payload.
        """
        if not chunks:
            print("⚠️ No chunks to upsert.")
            return

        # --- 1. DATA NORMALIZATION STRATEGY ---
        # We extract the raw text for embedding, regardless of the source format.
        texts_to_vectorize = []

        for chunk in chunks:
            # Case A: Visual Chunk (from VisualIngestionService)
            if "page_content" in chunk:
                texts_to_vectorize.append(chunk["page_content"])

            # Case B: Audio Chunk (from Transcriber)
            elif "text" in chunk:
                texts_to_vectorize.append(chunk["text"])

            else:
                print(f"⚠️ Skipping malformed chunk keys: {chunk.keys()}")
                texts_to_vectorize.append(
                    ""
                )  # Maintain index alignment with empty string

        print(f"🧠 Vectorizing {len(texts_to_vectorize)} chunks (Hybrid Mode)...")

        # --- 2. EMBEDDING GENERATION ---
        # Dense: Creates a 384-dimensional vector capturing "meaning"
        dense_embeddings = self.dense_model.encode(texts_to_vectorize)
        # Sparse: Creates a vector capturing "specific keywords"
        sparse_embeddings = list(self.sparse_model.embed(texts_to_vectorize))

        points = []

        # --- 3. PAYLOAD CONSTRUCTION ---
        for i, chunk in enumerate(chunks):
            text_content = texts_to_vectorize[i]

            # Optimization: Skip empty content to keep index clean
            if not text_content.strip():
                continue

            # Base Payload (Common fields)
            payload = {"video_id": video_id, "text": text_content, "type": "unknown"}

            # Map specific fields based on source type (The Adapter Logic)
            if "page_content" in chunk:
                # IS VISUAL DATA
                meta = chunk["metadata"]
                payload["type"] = "visual"
                # Visual events are instantaneous, so start == end
                payload["start"] = meta.get("timestamp", 0.0)
                payload["end"] = meta.get("timestamp", 0.0)
                payload["frame_path"] = meta.get("frame_path", "")

            elif "text" in chunk:
                # IS AUDIO DATA
                payload["type"] = "audio"
                payload["start"] = chunk.get("start", 0.0)
                payload["end"] = chunk.get("end", 0.0)

            # Create the PointStruct required by Qdrant
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID for the vector
                    payload=payload,
                    vector={
                        "text-dense": dense_embeddings[i].tolist(),
                        "text-sparse": models.SparseVector(
                            indices=sparse_embeddings[i].indices.tolist(),
                            values=sparse_embeddings[i].values.tolist(),
                        ),
                    },
                )
            )

        # Batch Upload
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"✅ Indexed {len(points)} hybrid vectors for video {video_id}.")
        else:
            print("⚠️ No valid points created.")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search using Reciprocal Rank Fusion (RRF).
        combines semantic search results with keyword match results.
        """
        # 1. Vectorize Query (Dense + Sparse)
        query_dense = self.dense_model.encode(query).tolist()
        query_sparse = list(self.sparse_model.embed([query]))[0]

        # 2. Hybrid Query Execution
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="text-dense",
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(),
                        values=query_sparse.values.tolist(),
                    ),
                    using="text-sparse",
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        return [hit.payload for hit in search_result.points]
