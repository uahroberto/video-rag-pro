import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    # 1. Detect model size from environment variable
    model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny")
    repo_id = f"systran/faster-whisper-{model_size}"

    print(f"🛠️ Starting robust download for: {repo_id}")
    print("⏳ This may take a while, but it only happens once...")

    # 2. Robust download
    # max_workers=1 is the key: obliga a bajar archivo por archivo en fila india.
    # Detects the "CAS service error" and firewall blocks.
    try:
        path = snapshot_download(
            repo_id=repo_id,
            max_workers=1,  # <--- ESTO ARREGLA TU ERROR
            resume_download=True,
        )
        print(f"\n✅ Model downloaded successfully to cache: {path}")
        print("🚀 Now you can run ./run.sh without problems.")

    except Exception as e:
        print(f"\n❌ Error fatal en la descarga: {e}")


if __name__ == "__main__":
    main()
