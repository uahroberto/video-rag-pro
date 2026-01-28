import time
from src.core.transcriber import VideoTranscriber
from src.core.chunking import ChunkingProcessor
from src.database.vector_store import VectorDatabase


def main():
    # Initialization
    print("--- 🛠️ Initializing full local pipeline ---")
    transcriber = VideoTranscriber()
    chunker = ChunkingProcessor(min_chunk_size=600)

    db = VectorDatabase()

    url = "https://youtu.be/7r2xz7tKY24?si=J5Oj-VJQ8u9ZOADe"
    video_id = url.split("v=")[-1].split("?")[0]

    print("\n--- 📥 Phase 1: Processing Video ---")
    start_time = time.time()

    # 1. Download and Transcribe
    audio_path, title = transcriber.download_audio(url)
    segments = transcriber.transcribe(audio_path)

    # 2. Chunking (Structured Aggregation)
    # Essential to preserve temporal metadata
    chunks = chunker.process(segments)

    # 3. Vectorization & Storage
    # This uses your CPU to create embeddings at zero cost
    print(f"\n--- 🧠 Phase 2: Vectorizing {len(chunks)} chunks ---")
    db.upsert_chunks(chunks, video_id)

    processing_time = time.time() - start_time
    print(f"\n✅ Video fully indexed in {processing_time:.2f}s")

    # 4. SEMANTIC SEARCH TEST
    print("\n--- 🔍 Phase 3: Testing Semantic Search ---")
    query = "What is the ultimate goal of physics?"

    results = db.search(query, limit=3)

    for i, res in enumerate(results):
        print(f"\nResult {i+1} [Starts at {res['start']:.2f}s]:")
        print(f"Text: {res['text'][:150]}...")


if __name__ == "__main__":
    main()
