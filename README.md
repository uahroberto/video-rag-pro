# 📹 Video RAG Pro

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red?style=for-the-badge)
![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-black?style=for-the-badge)
![Type Checked](https://img.shields.io/badge/Type%20Checked-Mypy-blueviolet?style=for-the-badge)

**Video RAG Pro** is an AI-powered engine that transforms passive video consumption into an interactive knowledge retrieval experience. It allows users to "chat" with video content, retrieving precise temporal citations from hours of footage in seconds.

Unlike standard summarizers, this system uses **Retrieval-Augmented Generation (RAG)** to ground answers in specific video timestamps, enabling non-linear consumption of technical lectures and conferences.

---

## 🚀 Key Features

* **⚡ High-Performance Ingestion:** Optimized local transcription pipeline using `faster-whisper` with custom quantization parameters, achieving a **0.14 Real-Time Factor (RTF)** on CPU.
* **🧠 Semantic Search:** Powered by **Qdrant** vector database to retrieve context based on meaning, not just keyword matching.
* **🛡️ Reliability Layer:** CI/CD integration with `pre-commit` hooks enforcing strict typing (`mypy`) and linting (`ruff`) standards.
* **🐳 Containerized Architecture:** Fully dockerized environment ensuring reproducibility across development and production.
* **🔒 Privacy First:** Local embeddings and transcription; video data never leaves the infrastructure during processing.

---

## 🛠️ Architecture

The system follows a modular ETL (Extract, Transform, Load) pipeline pattern:

```mermaid
graph LR
    A[Video Input] --> B(Audio Extraction via yt-dlp)
    B --> C{Transcriber Engine}
    C -->|Output: Text + Timestamps| D[Chunking Strategy]
    D --> E[Vector Embedding]
    E --> F[(Qdrant Vector DB)]
    G[User Query] --> H[Semantic Search]
    H <--> F
    H --> I[LLM Synthesis]
    I --> J[Answer + Time Buttons]


---

## 📊 Performance Benchmarks

Engineering decisions are data-driven. We optimized the inference pipeline to balance latency and accuracy.

**Test Environment:** Fedora Linux, CPU Inference, 1m45s Technical Sample.

| Configuration | Latency | Real-Time Factor (RTF) | Outcome |
| :--- | :--- | :--- | :--- |
| **Baseline** (Beam Size 5) | 26.0s | ~0.25 | High precision, slow UX. |
| **Optimized** (Beam Size 1) | **14.5s** | **~0.14** | **44% Latency Reduction** with maintained semantic integrity. |

> *Optimization Details:* By switching to Greedy Search (`beam_size=1`) and `int8` quantization, we achieved a sub-second response feel for end-users while maintaining entity recognition accuracy for RAG tasks.

---

## 💻 Tech Stack

* **Core:** Python 3.12
* **AI & NLP:** `faster-whisper`, `sentence-transformers`, OpenAI GPT-4o-mini
* **Database:** Qdrant (Vector Store)
* **Frontend:** Streamlit
* **DevOps:** Docker, Docker Compose, Pre-commit

---

## 🏃‍♂️ Quick Start

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
    # Edit .env with your OpenAI API Key and settings
    ```

3.  **Launch with Docker:**
    ```bash
    docker-compose up --build
    ```

4.  **Access the App:**
    Navigate to `http://localhost:8501`

---

## 🧪 Quality Assurance

We enforce code quality gates to prevent technical debt:

* **Type Safety:** 100% type coverage required via `mypy`.
* **Linting:** PEP 8 compliance enforced by `ruff`.
* **Testing:** Run local checks with:
    ```bash
    pre-commit run --all-files
    ```

---

## 🔮 Roadmap

* [ ] **Hybrid Search:** Implement BM25 + Dense Vector fusion for better keyword retrieval.
* [ ] **Visual RAG:** Integrate VLM (Vision Language Models) to index slides and visual code snippets.
* [ ] **RAGAS Evaluation:** Automated pipeline to measure Faithfulness and Answer Relevancy.
