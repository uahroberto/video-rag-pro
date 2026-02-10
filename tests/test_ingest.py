from src.core.transcriber import VideoTranscriber
from src.core.chunking import ChunkingProcessor
from src.database.vector_store import VectorDatabase


def main():
    # 1. Setup
    transcriber = VideoTranscriber()
    chunker = ChunkingProcessor(min_chunk_size=600)
    db = VectorDatabase()

    url = "https://www.youtube.com/watch?v=7r2xz7tKY24"
    video_id = url.split("v=")[-1]

    # 2. Ingestion Pipeline
    print("\n--- Starting Ingestion ---")
    audio_path, title = transcriber.download_audio(url)
    segments = transcriber.transcribe(audio_path)

    # 3. Processing
    chunks = chunker.process(segments)

    # 4. Storage (Vectorization happens here)
    db.upsert_chunks(chunks, video_id)

    # 5. TEST: Let's try a semantic search!
    print("\n--- Testing Search ---")
    query = "What is the main goal of physics?"
    relevant_segments = db.search(query)

    for i, res in enumerate(relevant_segments):
        print(f"\nResult {i + 1} (Starts at {res['start']:.2f}s):")
        print(f"Content: {res['text'][:150]}...")


if __name__ == "__main__":
    main()
