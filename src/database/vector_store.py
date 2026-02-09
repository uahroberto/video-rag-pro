import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv

load_dotenv()


class VectorDatabase:
    """
    Manages Hybrid Search (Dense + Sparse) interaction with Qdrant.
    Updated to support Multimodal ingestion (Audio + Visual payloads).
    """

    def __init__(self, collection_name: str = "video_knowledge_hybrid"):
        self.collection_name = collection_name

        print("🤖 Loading Dense model (all-MiniLM-L6-v2)...")
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("🤖 Loading Sparse model (Qdrant/bm25)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

        print(f"🌐 Database Mode: Client-Server ({qdrant_host}:{qdrant_port})")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
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
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
        else:
            print(f"✅ Collection '{self.collection_name}' ready.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], video_id: str) -> None:
        """
        Generates vectors and uploads them in batch.
        Handles polymorphism:
        - Audio Chunks: {'text': '...', 'start': 0.0, ...}
        - Visual Chunks: {'page_content': '...', 'metadata': {...}}
        """
        if not chunks:
            print("⚠️ No chunks to upsert.")
            return

        # --- 1. DATA NORMALIZATION ---
        texts_to_vectorize = []

        for chunk in chunks:
            # Type A: Visual Chunk (LangChain style)
            if "page_content" in chunk:
                texts_to_vectorize.append(chunk["page_content"])
            # Type B: Audio Chunk (Simple dict)
            elif "text" in chunk:
                texts_to_vectorize.append(chunk["text"])
            else:
                print(f"⚠️ Skipping malformed chunk keys: {chunk.keys()}")
                texts_to_vectorize.append("")

        print(f"🧠 Vectorizing {len(texts_to_vectorize)} chunks (Hybrid Mode)...")

        # --- 2. EMBEDDING GENERATION ---
        dense_embeddings = self.dense_model.encode(texts_to_vectorize)
        sparse_embeddings = list(self.sparse_model.embed(texts_to_vectorize))

        points = []

        # --- 3. PAYLOAD CONSTRUCTION ---
        for i, chunk in enumerate(chunks):
            text_content = texts_to_vectorize[i]
            if not text_content.strip():
                continue

            # Base Payload
            payload = {
                "video_id": video_id,
                "text": text_content,
                "type": "unknown",  # Default
            }

            # Map specific fields based on source type
            if "page_content" in chunk:
                # IS VISUAL
                meta = chunk["metadata"]
                payload["type"] = "visual"  # <--- ESTO ES LA CLAVE
                payload["start"] = meta.get("timestamp", 0.0)
                payload["end"] = meta.get("timestamp", 0.0)
                payload["frame_path"] = meta.get("frame_path", "")  # <--- Y ESTO

            elif "text" in chunk:
                # IS AUDIO
                payload["type"] = "audio"
                payload["start"] = chunk.get("start", 0.0)
                payload["end"] = chunk.get("end", 0.0)

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
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

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"✅ Indexed {len(points)} hybrid vectors for video {video_id}.")

    def search(
        self, query: str, limit: int = 5, video_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_dense = self.dense_model.encode(query).tolist()
        query_sparse = list(self.sparse_model.embed([query]))[0]

        # Construct Filter if video_id provided
        query_filter = None
        if video_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id", match=models.MatchValue(value=video_id)
                    )
                ]
            )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="text-dense",
                    limit=limit * 2,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(),
                        values=query_sparse.values.tolist(),
                    ),
                    using="text-sparse",
                    limit=limit * 2,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        return [hit.payload for hit in search_result.points]
