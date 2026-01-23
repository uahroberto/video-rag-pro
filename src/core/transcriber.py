import os
from faster_whisper import WhisperModel
import yt_dlp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VideoTranscriber:
    """
    Handles downloading audio from YouTube and transcribing it using a local Whisper model.
    Optimized for CPU usage via CTranslate2 (faster-whisper).
    """

    def __init__(self):
        # Load configuration from .env
        self.model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
        self.device = os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        
        print(f"🚀 Loading Whisper model '{self.model_size}' on {self.device} with precision {self.compute_type}...")
        
        # Initialize the model once (Singleton pattern)
        self.model = WhisperModel(
            self.model_size, 
            device=self.device, 
            compute_type=self.compute_type
        )

    def download_audio(self, youtube_url: str, output_path: str = "data/tmp") -> tuple[str, str]:
        """
        Downloads the best available audio from a YouTube URL.
        
        Args:
            youtube_url (str): The valid YouTube URL.
            output_path (str): Directory to save the temporary audio file.
            
        Returns:
            tuple: (file_path, video_title)
        """
        os.makedirs(output_path, exist_ok=True)
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            # Output template: data/tmp/VIDEO_ID.mp3
            'outtmpl': f'{output_path}/%(id)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            # Youtube blocked the acces (403) so we need to add this options
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'referer': 'https://www.google.com/',
            'nocheckcertificate': True
        }

        print(f"📥 Downloading audio from: {youtube_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = f"{output_path}/{info['id']}.mp3"
            return filename, info.get('title', 'Unknown Title')

    def transcribe(self, audio_path: str) -> list[dict]:
        """
        Transcribes an audio file and returns segments with precise timestamps.
        
        Args:
            audio_path (str): Path to the .mp3 file.
            
        Returns:
            list[dict]: A list of segments like {'start': 0.0, 'end': 2.0, 'text': 'Hello'}
        """
        print(f"🎙️ Transcribing {audio_path}... (Running locally on CPU)")
        
        # Optimized inference with Faster-Whisper
        # We force 'es' (Spanish) for testing, but can be set to None for auto-detection
        segments, _ = self.model.transcribe(
            audio_path, 
            beam_size=5,
            language="es", 
            vad_filter=True # Filters out silence to speed up processing
        )

        results = []
        # Convert the generator to a list to persist data
        # CRITICAL: We preserve start/end times here for the RAG citation feature later.
        # Citation feature is key. It may make it harder but it's a must.
        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            # Real-time feedback in console
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        return results

    # Just to keep the server clean
    def cleanup_temp_files(file_path: str):
        """Removes temporary audio files to save disk space."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 Maintenance: Cleaned up {file_path}")
        except Exception as e:
            print(f"⚠️ Maintenance Warning: Could not delete {file_path}: {e}")