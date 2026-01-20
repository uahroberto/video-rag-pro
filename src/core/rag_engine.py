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
        self.model = "gpt-4o-mini"

    def answer_question(self, question: str, video_id: str) -> tuple[str, list[dict]]:
        """
        Main logic: Retrieve chunks, build context, and generate a cited answer.
        """
        # 1. RETRIEVAL: Find the top 4 most relevant chunks in Qdrant
        context_segments = self.db.search(question, limit=4)
        
        if not context_segments:
            return "No relevant information found in the video.", []

        # 2. CONTEXT PREPARATION: Build a string with timestamps for the LLM
        # This follows the 'Structured Aggregation' logic to maintain accuracy
        context_text = ""
        for i, seg in enumerate(context_segments):
            context_text += f"\n[Source {i+1}] ({seg['start']:.2f}s - {seg['end']:.2f}s): {seg['text']}\n"

        # 3. PROMPT ENGINEERING: Setting the rules for the AI
        system_prompt = (
            "You are a professional assistant. Answer questions based on the provided video segments. "
            "You MUST cite the specific timestamp for every claim you make. "
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

        return response.choices[0].message.content, context_segments