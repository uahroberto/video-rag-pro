import os
from openai import OpenAI
from src.database.vector_store import VectorDatabase
from dotenv import load_dotenv

# Load environment variables for the API Key
load_dotenv()


class RAGEngine:
    """
    Orchestrates the Retrieval-Augmented Generation process.
    Connects the local semantic search with OpenAI's intelligence.
    Updated to support Multimodal Context (Audio + Visual).
    """

    def __init__(self):
        # We use gpt-4o-mini: high reasoning capability at a very low cost
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.db = VectorDatabase()
        self.model = "gpt-4o-mini"  # gpt-4o-mini | input: 0.15$, output: 0.60$

    def answer_question(self, question: str, video_id: str) -> tuple[str, list[dict]]:
        """
        Main logic: Retrieve chunks, build context, and generate a cited answer.
        Handles both text/audio segments and visual/OCR segments.
        """
        # 1. RETRIEVAL: Find top matches in Qdrant
        # TUNING: Increased limit from 5 to 10 to handle multi-part questions better.
        # This ensures we get context for "Question A" AND "Question B" if they are far apart.
        raw_context_segments = self.db.search(question, limit=15)
        print(f"🔎 Raw segments found: {len(raw_context_segments)}")

        # DEBUG: Inspect the first few results to confirm visual data presence
        for idx, seg in enumerate(raw_context_segments[:5]):
            source_type = seg.get("type", "unknown")
            print(
                f"  [{idx}] {source_type.upper()} {seg['start']}s: {seg['text'][:50]}..."
            )

        if not raw_context_segments:
            return "No encontré información relevante en el vídeo.", []

        # 1.1 FILTERING LOGIC (The Button Fix)
        context_segments = []
        seen_time_windows = set()

        for seg in raw_context_segments:
            # Relaxed window to 10s to allow more granular buttons
            time_window = int(seg["start"] // 10)
            if time_window not in seen_time_windows:
                context_segments.append(seg)
                seen_time_windows.add(time_window)

            # Increased visual button limit
            if len(context_segments) >= 7:
                break

        context_segments.sort(key=lambda x: x["start"])

        # 2. CONTEXT PREPARATION (Multimodal Aware)
        context_text = ""
        for i, seg in enumerate(context_segments):
            start_m, start_s = divmod(int(seg["start"]), 60)

            # Handle cases where 'end' might be missing in visual chunks (points in time)
            end_time = seg.get("end", seg["start"])
            end_m, end_s = divmod(int(end_time), 60)

            time_str = f"{start_m:02d}:{start_s:02d}"

            # Label the source type explicitly for the LLM
            # This allows the AI to say "As shown on the screen..." vs "The speaker says..."
            source_type = seg.get("type", "audio")
            source_tag = (
                "[VISUAL/SCREEN]" if source_type == "visual" else "[AUDIO/SPEECH]"
            )

            context_text += (
                f"\n[Source {i+1}] {source_tag} ({time_str}): {seg['text']}\n"
            )

        # 3. PROMPT ENGINEERING (Refined for multi-questions & visual context)
        system_prompt = (
            "You are a professional assistant. Answer questions based on the provided video segments. "
            "INSTRUCTIONS:\n"
            "1. You MUST cite the specific timestamp (e.g., 04:15) for every claim.\n"
            "2. If the information comes from a [VISUAL/SCREEN] source, explicitly mention it "
            "(e.g., 'As shown in the code example on screen...').\n"
            "3. If the answer is not in the context, say you don't know."
        )

        user_prompt = f"User Question: {question}\n\nVideo Content:\n{context_text}"

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
