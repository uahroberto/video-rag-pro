from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv

load_dotenv()

# Configuración
COLLECTION_NAME = "video_knowledge_hybrid"
DENSE_MODEL_ID = "all-MiniLM-L6-v2"
SPARSE_MODEL_ID = "Qdrant/bm25"


def main():
    print("--- ⚖️ INICIANDO COMPARATIVA: DENSE VS HYBRID ---")

    # 1. Direct Connection (Bypass of your class to have total control)
    client = QdrantClient(host="localhost", port=6333)

    # 2. Load Models
    print("🤖 Loading models for the test...")
    dense_model = SentenceTransformer(DENSE_MODEL_ID)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_ID)

    # 3. TEST QUERIES (Based on your logs)
    # Cases where Hybrid should win: Exact names, dates, codes.
    test_queries = [
        "2016",  # Exact date (Dense usually ignores numbers)
        "Linux Torvalds",  # Proper name
        "documento.corees",  # Rare technical token (Dense will fail here)
        "selector",  # Technical term
    ]

    for query_text in test_queries:
        print(f"\n\n🔎 PREGUNTA: '{query_text}'")

        # Generate vectors
        q_dense = dense_model.encode(query_text).tolist()
        q_sparse = list(sparse_model.embed([query_text]))[0]

        # --- ROUND 1: ONLY DENSO (What you had before) ---
        dense_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_dense,
            using="text-dense",  # Force using only the semantic vector
            limit=3,
        ).points

        print("  🔴 [DENSE ONLY] Top 3:")
        for i, hit in enumerate(dense_results):
            print(
                f"     {i+1}. Score: {hit.score:.4f} | Texto: {hit.payload['text'][:60]}..."
            )

        # --- ROUND 2: HYBRID (What you have now) ---
        hybrid_results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=q_dense, using="text-dense", limit=10),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=q_sparse.indices.tolist(),
                        values=q_sparse.values.tolist(),
                    ),
                    using="text-sparse",
                    limit=10,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),  # La magia del RRF
            limit=3,
        ).points

        print("  🟢 [HYBRID RRF] Top 3:")
        for i, hit in enumerate(hybrid_results):
            # Note: RRF does not give a similarity score 0-1, it gives a ranking score
            print(
                f"     {i+1}. Score: {hit.score:.4f} | Texto: {hit.payload['text'][:60]}..."
            )


if __name__ == "__main__":
    main()
