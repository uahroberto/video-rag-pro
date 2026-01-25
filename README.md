# Video RAG Pro: AI-Powered Video Knowledge Base

A high-performance RAG (Retrieval-Augmented Generation) system that allows users to "chat" with YouTube videos. It features local transcription and semantic search with precise timestamp citations.

## 🛠️ Technical Stack
- **Inference:** Faster-Whisper (int8 quantization) for CPU-optimized transcription.
- **Embeddings:** `all-MiniLM-L6-v2` (Local execution).
- **Vector Database:** Qdrant (Local persistence).
- **LLM:** OpenAI GPT-4o-mini (Reasoning & Citations).
- **Frontend:** Streamlit with "Key Hack" for interactive video seeking.

## 🚀 Engineering Highlights
- **Structured Aggregation:** Custom chunking logic that preserves temporal metadata for 100% accurate video citations.
- **Hybrid Architecture:** Local processing for privacy and cost-efficiency, cloud-based intelligence for final answers.
- **Hardware Optimized:** Designed to run on consumer hardware (Ryzen 7) and low-power servers (Intel N100).
