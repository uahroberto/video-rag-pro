import os
import asyncio
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv
from typing import Tuple, Dict, Any, Optional
from src.database.vector_store import VectorDatabase
from src.core.transcriber import VideoTranscriber
from src.core.chunking import ChunkingProcessor
from src.video_processing.ocr_service import OCRService

# Load environment variables for the API Key
load_dotenv()


class RAGEngine:
    """
    Orchestrates the Retrieval-Augmented Generation process.
    Connects the local semantic search with OpenAI's intelligence.
    Updated to support Multimodal Context (Audio + Visual) & Concurrent Ingestion.
    """

    def __init__(self) -> None:
        # We use gpt-4o-mini: high reasoning capability at a very low cost
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.db = VectorDatabase()
        self.model = "gpt-4o-mini"

        # Components for Ingestion
        self.transcriber = VideoTranscriber()
        self.chunker = ChunkingProcessor(min_chunk_size=600)
        self.ocr_service = OCRService()

        # Executor for CPU-bound tasks (Whisper, RapidOCR)
        # OPTIMIZATION: Use all available cores to maximize throughput
        max_workers = os.cpu_count() or 4
        print(f"🚀 Initializing RAGEngine with {max_workers} worker threads")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def ingest_video(
        self, youtube_url: str, include_visuals: bool = True
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        High-Performance Concurrent Ingestion Pipeline.
        Orchestrates:
        1. Download & Transcribe (Audio)
        2. Frame Extraction & OCR (Visual) [Optional]
        All running in parallel.
        """
        print(f"🚀 Starting Concurrent Ingestion for: {youtube_url}")
        if not include_visuals:
            print("ℹ️  Audio-Only Mode Enabled: Skipping visual processing.")

        loop = asyncio.get_running_loop()

        # Phase 1: Parallel Downloads (IO-bound but fast)
        # We start both downloads concurrently if visual is needed.

        future_audio = loop.run_in_executor(
            self.executor, self.transcriber.download_audio, youtube_url
        )

        future_video = None
        if include_visuals:
            future_video = loop.run_in_executor(
                self.executor, self._download_video_best, youtube_url
            )

        print("⏳ Waiting for downloads...")
        if future_video:
            (audio_res, video_path) = await asyncio.gather(future_audio, future_video)
            audio_path, video_title = audio_res
        else:
            audio_path, video_title = await future_audio
            video_path = None

        print(f"✅ Downloads Ready. Audio: {audio_path}, Video: {video_path}")

        # Phase 2: Scatter-Gather (Parallel Execution)
        # task_audio: Transcribe Audio
        # task_visual: Process Video Frames (OCR) - Now fully parallelized internally

        task_audio = loop.run_in_executor(
            self.executor, self._process_audio_task, audio_path
        )

        task_visual = None
        if include_visuals and video_path:
            # We run _process_video_task in the executor to avoid blocking the loop
            # during the frame extraction phase, even though it spawns its own tasks.
            task_visual = loop.run_in_executor(
                self.executor, self._process_video_task, video_path
            )

        # Wait for both to complete
        if task_visual:
            audio_chunks, visual_chunks = await asyncio.gather(task_audio, task_visual)
        else:
            audio_chunks = await task_audio
            visual_chunks = []

        # Phase 3: Aggregation & Storage
        all_chunks = audio_chunks + visual_chunks
        print(f"💾 Upserting {len(all_chunks)} combined chunks to VectorDB...")

        video_id = youtube_url.split("v=")[-1].split("&")[0]
        video_id = youtube_url.split("v=")[-1].split("&")[0]
        await loop.run_in_executor(
            self.executor, self.db.upsert_chunks, all_chunks, video_id
        )
        print("✅ Ingestion Complete!")

        return video_path, audio_path, video_title

    def _process_audio_task(self, audio_path: str) -> list[Dict[str, Any]]:
        """Wrapper for audio transcription and chunking."""
        print("🔊 Starting Audio Transcription...")
        segments = self.transcriber.transcribe(audio_path)
        chunks = self.chunker.create_chunks(segments)
        print(f"🔊 Audio processing complete: {len(chunks)} chunks.")
        return chunks

    def _process_video_task(self, video_path: str) -> list[Dict[str, Any]]:
        """
        High-Performance Video Processing Loop.

        Phase 1 (Sequential):
        - Iterate video frames.
        - Downscale & Deduplicate (Fast CPU ops).
        - Save valid frames to temp storage.

        Phase 2 (Parallel):
        - Submit all valid frames to ThreadPoolExecutor for OCR.
        - This saturates the CPU as RapidOCR releases GIL for ONNX runtime.
        """
        print(f"👁️ Starting Visual Processing: {video_path}")
        chunks = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Performance Tuning: Increase interval to reduce total OCR workload.
        # User feedback: "Extraction is too long".
        # 15 seconds (User Request)
        step_seconds = 15
        step_frames = int(fps * step_seconds)

        current_frame_pos = (
            0  # Use a different name to avoid confusion with frame_count
        )
        last_processed_frame_hash = None

        # Store tasks for Phase 2
        ocr_tasks = []
        temp_files_to_clean = []

        print(f"👁️ Phase 1: Extracting & Filtering Frames (Every {step_seconds}s)...")

        frame_count = 0
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 == 0:
                timestamp = current_frame_pos / fps
                print(f"   ... Scanning frame {frame_count} at {timestamp:.1f}s ...")

            timestamp = current_frame_pos / fps

            # --- OPTIMIZATION 1: Downscaling ---
            h, w = frame.shape[:2]
            if h > 720:
                scale = 720 / h
                new_w = int(w * scale)
                frame = cv2.resize(frame, (new_w, 720))

            # --- OPTIMIZATION 2: Smart Deduplication ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (8, 8))
            current_hash = small

            is_duplicate = False
            if last_processed_frame_hash is not None:
                score = np.mean(np.abs(current_hash - last_processed_frame_hash))  # type:ignore[unreachable]
                if score < 5.0:
                    is_duplicate = True

            if not is_duplicate:
                # Prepare for OCR
                temp_path = f"data/tmp/temp_frame_{timestamp:.2f}.jpg"
                cv2.imwrite(temp_path, frame)
                temp_files_to_clean.append(temp_path)

                # We collect data needed for the task
                ocr_tasks.append({"path": temp_path, "timestamp": timestamp})

                last_processed_frame_hash = current_hash

            current_frame_pos += step_frames

        cap.release()
        print(f"👁️ Phase 1 Complete. Found {len(ocr_tasks)} unique frames.")
        print("👁️ Phase 2: Running Parallel OCR...")

        # Phase 2: Parallel OCR
        # We use a helper function to make it pickle-able/map-able
        #  if we switched to ProcessPool,
        # but for ThreadPool with GIL-releasing libs (like onnxruntime), this is fine.
        # However, 'self.ocr_service.extract_text' is bound method.
        # Ideally we run it in the executor.

        # Since we are already INSIDE a thread (this method called via run_in_executor),
        # we can submit new tasks to the SAME executor or a new one?
        # RAGEngine.executor is used for High-level orchestration.
        # It's better to reuse it or have a dedicated one.
        # Using `self.executor.map` within a task running
        # in `self.executor` might cause deadlock
        # if max_workers is low and we occupy one slot waiting for others.
        # But we set max_workers to cpu_count.
        # To be safe, let's just map synchronously here if we are already in a thread,
        # OR better: use `concurrent.futures.wait`
        # on futures submitted to `self.executor`.

        # Deadlock Risk: If we have 4 cores, and we use
        # one for Audio and one for Video (this function),
        # we have 2 left. If we submit 100 OCR tasks, they will run 2 at a time.
        # We wait for them. This is fine. NO DEADLOCK.

        futures = []
        for task in ocr_tasks:
            # We submit the EXTRACT TEXT task
            future = self.executor.submit(self.ocr_service.extract_text, task["path"])
            futures.append((future, task))  # Keep track of metadata

        for future, task in futures:
            # Result is blocking, but we are in a thread so it's fine.
            text = future.result()
            if text:
                chunks.append(
                    {
                        "page_content": text,
                        "metadata": {
                            "source": "visual",
                            "timestamp": task["timestamp"],
                            "frame_path": task["path"],
                        },
                    }
                )

        # Cleanup
        # for path in temp_files_to_clean:
        #    if os.path.exists(path): os.remove(path)

        print(f"👁️ Phase 2 Complete. Generated {len(chunks)} visual chunks.")
        return chunks

    def _download_video_best(self, url: str) -> str:
        """
        Temporary helper to download video for visual processing.
        Leverages yt-dlp via os.system or specialized library call if possible.
        """

        # For this refactor, let's assume we use yt-dlp directly.
        import yt_dlp

        # FIX: Force h264 (avc1) codec to ensure OpenCV compatibility.
        # AV1 (av01) or VP9 often fail on systems without HW acceleration
        ydl_opts = {
            "format": "bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/"
            "best[ext=mp4]/best",
            "outtmpl": "data/videos/%(id)s.%(ext)s",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)

    def answer_question(
        self, question: str, video_id: str
    ) -> tuple[str, list[Dict[str, Any]]]:
        """
        Main logic: Retrieve chunks, build context, and generate a cited answer.
        Handles both text/audio segments and visual/OCR segments.
        """
        if not video_id:
            print("⚠️ No Video ID provided for search context. Results might be mixed.")

        # 1. RETRIEVAL: Find top matches in Qdrant
        # TUNING: Increased limit from 5 to 10 to handle multi-part questions better.
        # This ensures we get context for "Question A & B" if they are far apart.
        raw_context_segments = self.db.search(question, limit=15, video_id=video_id)
        print(
            f"🔎 Raw segments found: {len(raw_context_segments)} for video {video_id}"
        )

        # DEBUG: Inspect the first few results to confirm visual data presence
        for idx, seg in enumerate(raw_context_segments[:5]):
            source_type = seg.get("type", "unknown")
            print(
                f"  [{idx}] {source_type.upper()} "
                f"{seg['start']}s: {seg['text'][:50]}..."
            )

        if not raw_context_segments:
            return "No encontré información relevante en el vídeo.", []

        # 1.1 FILTERING LOGIC (For UI Buttons Only)
        # We filter buttons to avoid overcrowding the UI
        # We use ALL segments for the LLM context.
        context_segments = []
        seen_time_windows = set()

        for seg in raw_context_segments:
            # Relaxed window to 10s to allow more granular buttons
            time_window = int(seg["start"] // 10)
            if time_window not in seen_time_windows:
                context_segments.append(seg)
                seen_time_windows.add(time_window)

            if len(context_segments) >= 7:
                break

        context_segments.sort(key=lambda x: x["start"])

        # 1.2 PROMPT CONTEXT LOGIC (All relevant segments)
        # Sort all retrieved segments by time for the LLM to read a coherent story
        prompt_segments = sorted(raw_context_segments, key=lambda x: x["start"])

        # 2. CONTEXT PREPARATION (Multimodal Aware - XML Structured)
        context_text = "<video_context>\n"
        for i, seg in enumerate(prompt_segments):
            start_m, start_s = divmod(int(seg["start"]), 60)

            # Handle cases where 'end' might be missing
            _end_time = seg.get("end", seg["start"])

            # Label the source type explicitly
            source_type = seg.get("type", "audio").upper()

            # Build XML-like structure for the segment
            context_text += f"""
    <context_slice id="{i + 1}">
        <source_type>{source_type}</source_type>
        <timestamp>{start_m:02d}:{start_s:02d}</timestamp>
        <content>{seg["text"]}</content>
    </context_slice>
"""
        context_text += "</video_context>"

        # 3. PROMPT ENGINEERING (Refined for reasoning & visual accuracy)
        system_prompt = (
            "You are an Expert Technical Tutor. "
            "Answer based on the provided video segments.\n"
            "RULES:\n"
            "1. FORMATTING: Always use Markdown code blocks "
            "for code found in the context.\n"
            "2. VISUAL SUPREMACY: If information comes from a "
            "<source_type>VISUAL</source_type> tag, "
            "prioritize it for syntax/code accuracy.\n"
            "3. CITATIONS: Cite the specific timestamp (e.g., 04:15) "
            "for every claim.\n"
            "4. UNKNOWN: If the answer is not in the context, say you don't know."
        )

        user_prompt = (
            f"User Question: {question}\n\n"
            f"Video Content (XML Structured):\n{context_text}"
        )

        # 4. GENERATION: Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,  # Zero temperature ensures consistent, factual answers
        )

        # Return the generated answer AND the filtered segments for the buttons
        # The segments now contain 'frame_path' if they are visual
        return response.choices[0].message.content, context_segments
