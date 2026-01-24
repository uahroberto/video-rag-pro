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
    """

    def __init__(self):
        # We use gpt-4o-mini: high reasoning capability at a very low cost
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.db = VectorDatabase()
        self.model = "gpt-4o-mini" # gpt-4o-mini | input: 0.15$, output: 0.60$

    def answer_question(self, question: str, video_id: str) -> tuple[str, list[dict]]:
        """
        Main logic: Retrieve chunks, build context, and generate a cited answer.
        """
        # 1. RETRIEVAL: Find top matches in Qdrant
        # We fetch 5 candidates to allow for filtering duplicates later
        raw_context_segments = self.db.search(question, limit=5)
        
        if not raw_context_segments:
            return "No relevant information found in the video.", []

        # 1.1 FILTERING LOGIC (The Button Fix)
        # We process 'raw_context_segments' into 'context_segments' to remove redundancy
        context_segments = []
        seen_time_windows = set()
        
        for seg in raw_context_segments:
            # We group timestamps by 30-second windows to avoid buttons like 4:05 and 4:08
            time_window = int(seg['start'] // 30)
            
            if time_window not in seen_time_windows:
                context_segments.append(seg)
                seen_time_windows.add(time_window)
            
            # Limit to 3 distinct buttons for a clean UI
            if len(context_segments) >= 3:
                break
        
        # Sort chronologically so the buttons appear in order (e.g., 0:30, then 4:15)
        context_segments.sort(key=lambda x: x['start'])

        # 2. CONTEXT PREPARATION: Build a string with timestamps for the LLM
        # This follows the 'Structured Aggregation' logic to maintain accuracy
        context_text = ""
        for i, seg in enumerate(context_segments):
            # UX FIX: Convert seconds to MM:SS directly in the Context
            # This forces the LLM to see and use human-readable timestamps
            start_m, start_s = divmod(int(seg['start']), 60)
            end_m, end_s = divmod(int(seg['end']), 60)
            
            time_str = f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}"
            
            # The context now looks like: [Source 1] (04:15 - 04:30): ...
            context_text += f"\n[Source {i+1}] ({time_str}): {seg['text']}\n"

        # 3. PROMPT ENGINEERING: Setting the rules for the AI
        system_prompt = (
            "You are a professional assistant. Answer questions based on the provided video segments. "
            "You MUST cite the specific timestamp (e.g., 04:15) for every claim you make. "
            "If the answer is not in the context, say you don't know. Be concise."
        )
        
        user_prompt = f"User Question: {question}\n\nVideo Content:\n{context_text}"

        # 4. GENERATION: Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0  # Zero temperature ensures consistent, factual answers
        )

        # Return the generated answer AND the filtered segments for the buttons
        return response.choices[0].message.content, context_segments