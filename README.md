# 📹 Video RAG Pro

![Python](https://img.shields.io/badge/Python-3.12-blue) ![AI](https://img.shields.io/badge/Multimodal-Audio%20%2B%20Visual-purple) ![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-orange)

**Video RAG Pro** is a **Multimodal AI engine** that transforms passive video consumption into an interactive knowledge retrieval experience. Unlike standard tools that only "listen" to audio, Video RAG Pro **"watches" the video too**, reading code snippets, slides, and diagrams on screen.

It allows users to "chat" with video content, retrieving precise **Audio & Visual citations** in seconds using a Hybrid Search architecture.

## 🚀 Key Features

* **👁️ Visual RAG (OCR):** Automatically extracts and indexes text from video frames (code, slides, diagrams) using **RapidOCR**. Captures information that is *shown* but not *spoken* (e.g., specific variable names or config values).
* **🔍 Hybrid Search:** Powered by **Qdrant**, combining **Dense Vectors** (Semantic Search) with **Sparse Vectors (BM25)** (Keyword Search). This ensures you find concepts by meaning ("how to loop") AND by exact syntax (`for i in range`).
* **⚡ High-Performance Ingestion:** Optimized pipeline using `faster-whisper` (Int8 quantization) for audio and intelligent frame sampling (5s interval) for video, achieving high accuracy with low latency.
* **🛡️ Reliability Layer:** CI/CD integration with `pre-commit` hooks enforcing strict typing (`mypy`) and linting (`ruff`) standards.
* **🐳 Containerized Architecture:** Fully dockerized environment ensuring reproducibility across development and production.
* **🔒 Privacy First:** Local embeddings, OCR, and transcription; video data never leaves your infrastructure during processing (except for the final LLM reasoning step).

## 🛠️ Architecture

The system follows a modular **Multimodal ETL** pipeline:

1.  **Extract:** Downloads Video (MP4) and Audio (MP3) using a robust anti-bot `yt-dlp` wrapper.
2.  **Transform (Audio):** Transcribes speech to text using `faster-whisper`.
3.  **Transform (Visual):** Scans video frames every 5 seconds, detecting text/code via `RapidOCR`.
4.  **Load:** Vectorizes both streams (Audio & Visual) using **Hybrid Embeddings** (Dense + Sparse) and stores them in **Qdrant** with metadata pointing to the exact source type and timestamp.

## 📊 Performance Benchmarks

Engineering decisions are data-driven to balance latency and accuracy.

| Configuration | Latency | RTF | Outcome |
| :--- | :--- | :--- | :--- |
| **Audio Baseline** | 26.0s | ~0.25 | High precision, slow UX. |
| **Audio Optimized** | 14.5s | ~0.14 | **44% Latency Reduction** with `int8` quantization. |
| **Visual OCR** | ~0.8s/frame | N/A | Real-time capable extraction on CPU (ONNX Runtime). |

## 💻 Tech Stack

* **Core:** Python 3.12
* **AI & NLP:** `faster-whisper`, `sentence-transformers`, `fastembed` (Sparse Vectors), OpenAI GPT-4o-mini.
* **Computer Vision:** `RapidOCR`, `opencv-python-headless`.
* **Database:** Qdrant (Hybrid Vector Store).
* **Frontend:** Streamlit (with Custom Media Cards).
* **DevOps:** Docker, Docker Compose, Pre-commit, Ruff, Mypy.

## runner️ Quick Start

### Prerequisites

* Docker & Docker Compose
* OpenAI API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/video-rag-pro.git](https://github.com/yourusername/video-rag-pro.git)
    cd video-rag-pro
    ```

2.  **Configure Environment:**
    ```bash
    cp .env.example .env
    # Edit .env with your OpenAI API Key and Qdrant settings
    ```

3.  **Launch with Docker:**
    ```bash
    docker-compose up --build
    ```

4.  **Access the App:** Navigate to `http://localhost:8501`.

5.  **Usage:**
    * Paste a YouTube URL (e.g., a coding tutorial).
    * Click **"🚀 Procesar Vídeo Completo"**.
    * Ask questions like *"What variable name is used in the code?"* or *"What does the slide say about architecture?"*.

## 🧪 Quality Assurance

We enforce code quality gates to prevent technical debt:
* **Type Safety:** 100% type coverage required via `mypy`.
* **Linting:** PEP 8 compliance enforced by `ruff`.
* **Testing:** Run local checks with: `pre-commit run --all-files`

## 🔮 Roadmap

* [x] **Hybrid Search:** Implement BM25 + Dense Vector fusion.
* [x] **Visual RAG:** Index slides and visual code snippets via OCR.
* [ ] **VLM Integration:** Upgrade from OCR to Vision Language Models (e.g., LLaVA) to "describe" images, not just read text.
* [ ] **RAGAS Evaluation:** Automated pipeline to measure Faithfulness and Answer Relevancy.
