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
        
        # Load local embedding model (runs on CPU)
        # Size 384 matches the collection config
        print("🤖 Loading local embedding model (all-MiniLM-L6-v2)...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        qdrant_path = os.getenv("QDRANT_PATH")
        qdrant_host = os.getenv("QDRANT_HOST")
        
        if qdrant_path:
            self.client = QdrantClient(path=qdrant_path)
        else:
            self.client = QdrantClient(host=qdrant_host, port=int(os.getenv("QDRANT_PORT", 6333)))

        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            print(f"🛠️ Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[dict], video_id: str):
        """
        Converts chunks to vectors and stores them with temporal metadata.
        """
        print(f"🧠 Vectorizing {len(chunks)} chunks for video {video_id}...")
        
        points = []
        for chunk in chunks:
            # Generate the semantic vector
            vector = self.encoder.encode(chunk['text']).tolist()
            
            # Create a unique point with metadata (Payload)
            # Metadata is critical for the front-end to jump to the right timestamp
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "video_id": video_id,
                    "text": chunk['text'],
                    "start": chunk['start'],
                    "end": chunk['end']
                }
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("✅ Vectors successfully stored in Qdrant.")

    def search(self, query: str, limit: int = 3):
        """
        Finds the most relevant video segments using the modern query_points API.
        """
        # Generate the embedding for the user's question
        query_vector = self.encoder.encode(query).tolist()
        
        # In modern Qdrant, query_points is the preferred method over search()
        # It provides better performance and a unified interface [Source 8, 16]
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        
        # We extract the payload from the resulting points
        return [hit.payload for hit in response.points]