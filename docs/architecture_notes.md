video-rag-pro/
│
├── .env                    # Variables de entorno (API Keys, IP del servidor)
├── .gitignore              # Para no subir basura a GitHub
├── requirements.txt        # Librerías del proyecto
├── docker-compose.yml      # (Opcional) Por si quieres correr DB en local también
├── README.md               # Tu carta de presentación
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/               # LÓGICA PURA (El "Cerebro")
│   │   ├── __init__.py
│   │   ├── transcriber.py  # Lógica de Whisper (Local)
│   │   ├── chunking.py     # Tu algoritmo custom de timestamps
│   │   └── rag_engine.py   # Lógica de OpenAI + Prompts
│   │
│   ├── database/           # PERSISTENCIA (La "Memoria")
│   │   ├── __init__.py
│   │   └── vector_store.py # Cliente de Qdrant
│   │
│   ├── api/                # COMUNICACIÓN (La "Boca")
│   │   ├── __init__.py
│   │   └── main.py         # Endpoints FastAPI
│   │
│   └── frontend/           # INTERFAZ (La "Cara")
│       └── app.py          # Streamlit UI
│
└── data/                   # Almacén temporal (ignorada en git)
    └── tmp/                # Audios descargados temporalmente
