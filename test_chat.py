from src.core.rag_engine import RAGEngine


def main():
    chat = RAGEngine()

    video_id = "7r2xz7tKY24"
    question = "¿Cuál es el objetivo de la física y qué herramientas menciona el vídeo?"

    print(f"🤔 Pregunta: {question}")
    answer, sources = chat.answer_question(question, video_id)

    print("\n🤖 Respuesta de la IA:")
    print(answer)

    print("\n📍 Fuentes para verificar:")
    for i, s in enumerate(sources):
        print(f"[{i+1}] Segundo {s['start']:.2f}: {s['text'][:70]}...")


if __name__ == "__main__":
    main()
