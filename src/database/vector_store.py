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

    Architecture:
    - Dense Vector: 'all-MiniLM-L6-v2' (Semantic understanding)
    - Sparse Vector: 'Qdrant/bm25' (Keyword matching)
    """

    def __init__(self, collection_name: str = "video_knowledge_hybrid"):
        self.collection_name = collection_name

        # 1. Load Dense Model (Semantic)
        print("🤖 Loading Dense model (all-MiniLM-L6-v2)...")
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")

        # 2. Load Sparse Model (Keywords/BM25)
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
        Generates both Dense and Sparse vectors and uploads them in batch.
        """
        texts = [chunk["text"] for chunk in chunks]
        print(f"🧠 Vectorizing {len(chunks)} chunks (Hybrid Mode)...")

        # 1. Generate Dense Embeddings
        dense_embeddings = self.dense_model.encode(texts)

        # 2. Generate Sparse Embeddings
        sparse_embeddings = list(self.sparse_model.embed(texts))

        points = []
        for i, text in enumerate(texts):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    payload={
                        "video_id": video_id,
                        "text": text,
                        "start": chunks[i]["start"],
                        "end": chunks[i]["end"],
                    },
                    vector={
                        "text-dense": dense_embeddings[i].tolist(),
                        "text-sparse": models.SparseVector(
                            indices=sparse_embeddings[i].indices.tolist(),
                            values=sparse_embeddings[i].values.tolist(),
                        ),
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ Indexed {len(points)} hybrid vectors for video {video_id}.")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search using Reciprocal Rank Fusion (RRF).
        """
        # 1. Vectorize Query
        query_dense = self.dense_model.encode(query).tolist()
        query_sparse = list(self.sparse_model.embed([query]))[0]

        # 2. Hybrid Query (PREFETCH + FUSION)
        # Here was the error: We used FusionQuery instead of NamedVector
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="text-dense",
                    limit=limit * 2,  # Traemos más candidatos para fusionar mejor
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
            # THE KEY CHANGE: We used FusionQuery instead of NamedVector
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        return [hit.payload for hit in search_result.points]
