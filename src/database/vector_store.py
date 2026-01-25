import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class VectorDatabase:
    """
    Handles local/remote vector storage and semantic search.
    Uses a local embedding model to keep costs at zero.
    """

    def __init__(self, collection_name: str = "video_knowledge"):
        self.collection_name = collection_name

        # Internal log in English for professional server monitoring
        print("🤖 Loading local embedding model (all-MiniLM-L6-v2)...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Logic: Docker uses QDRANT_HOST, local dev uses QDRANT_PATH
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        # I commented this line out  to remove the dependency on QDRANT_PATH
        # qdrant_path = os.getenv("QDRANT_PATH")

        # Previous logic was for local dev, now we only use Docker
        print(f"🌐 Database Mode: Client-Server ({qdrant_host}:{qdrant_port})")

        try:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.client.get_collections()  # Test connection
        except Exception as e:
            print(f"Error al conectar con Qdrant: {e}")
            raise

        self._ensure_collection()

    def _ensure_collection(self):
        """
        Checks for collection existence atomically to avoid 409 Conflicts.
        This ensures idempotency during Streamlit reruns.
        """
        # Use the native method to check existence and prevent race conditions
        if not self.client.collection_exists(self.collection_name):
            # Frontend-facing or log messages in Spanish
            print(f"🛠️ Creando colección: {self.collection_name}")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        else:
            # Success message for the user/operator
            print(
                f"✅ La colección '{self.collection_name}' ya está lista para su uso."
            )

    def upsert_chunks(self, chunks: list[dict], video_id: str):
        """Converts chunks to vectors and stores them with temporal metadata."""
        print(f"🧠 Vectorizing {len(chunks)} chunks for video {video_id}...")

        points = []
        for chunk in chunks:
            vector = self.encoder.encode(chunk["text"]).tolist()

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "video_id": video_id,
                        "text": chunk["text"],
                        "start": chunk["start"],
                        "end": chunk["end"],
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        print("✅ Vectors successfully stored in Qdrant.")

    def search(self, query: str, limit: int = 3):
        """Finds relevant segments using the modern query_points API."""
        query_vector = self.encoder.encode(query).tolist()

        # Using modern unified API for high performance
        response = self.client.query_points(
            collection_name=self.collection_name, query=query_vector, limit=limit
        )

        return [hit.payload for hit in response.points]
