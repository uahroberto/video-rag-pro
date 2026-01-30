from src.services.visual_ingestion import VisualIngestionService
from src.database.vector_store import VectorDatabase


def run_integration_test():
    """
    Test script to verify the connection between Visual Extraction and Vector DB.
    """
    video_id = "test_integration_001"
    # Asegúrate de que este video existe (ya lo descargamos antes)
    video_path = "data/videos/test_video.mp4"

    print("--- 1. STARTING VISUAL PIPELINE ---")
    visual_service = VisualIngestionService()
    # Procesamos el video (extracción + OCR)
    chunks = visual_service.process_video(video_path, video_id, interval=30)

    if not chunks:
        print("❌ No chunks generated. Test aborted.")
        return

    print(f"✅ Generated {len(chunks)} visual chunks.")

    print("\n--- 2. STARTING DATABASE INGESTION ---")
    db = VectorDatabase()
    # Aquí probamos el Polimorfismo: Le pasamos chunks visuales y debe aceptarlos
    db.upsert_chunks(chunks, video_id)

    print("\n--- 3. VERIFICATION (SEARCH) ---")
    # Buscamos algo que sabemos que sale en los créditos del video Big Buck Bunny
    query = "Blender Foundation"
    results = db.search(query, limit=3)

    print(f"🔎 Query: '{query}'")
    for res in results:
        # Si vemos type='visual', ¡hemos triunfado!
        print(f"🎯 Found match! Type: {res.get('type')} | Time: {res.get('start')}s")
        print(f"   Text: {res.get('text')[:50]}...")


if __name__ == "__main__":
    run_integration_test()
