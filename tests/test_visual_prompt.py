import unittest
from unittest.mock import MagicMock, patch
from src.core.rag_engine import RAGEngine


class TestVisualPrompt(unittest.TestCase):
    def setUp(self):
        # Patch environment variables to avoid real API keys if needed,
        # though RAGEngine loads them in __init__.
        # We will mock the OpenAI client anyway.
        self.patcher_openai = patch("src.core.rag_engine.OpenAI")
        self.mock_openai = self.patcher_openai.start()

        self.patcher_db = patch("src.core.rag_engine.VectorDatabase")
        self.mock_db_class = self.patcher_db.start()

        # Setup RAGEngine
        self.rag = RAGEngine()
        # self.rag.db is now the return value of the mocked VectorDatabase class
        # We can configure it if needed, but for now it's a MagicMock

    def tearDown(self):
        self.patcher_openai.stop()
        self.patcher_db.stop()

    def test_visual_context_extraction(self):
        """
        Test that the RAG engine correctly formats visual context and
        the system prompt encourages code formatting.
        """
        # 1. Mock Database Search Results
        # Simulating a scenario where the answer is ONLY in the visual context
        mock_results = [
            {
                "id": "chunk_1",
                "text": "In this video we are configuring the retry logic.",
                "start": 10.0,
                "end": 15.0,
                "type": "audio",
            },
            {
                "id": "chunk_2",
                "text": "const MAX_RETRIES = 5;",
                "start": 12.0,
                "end": 12.0,
                "type": "visual",
            },
        ]
        self.rag.db.search.return_value = mock_results

        # 2. Mock OpenAI Response
        # We mock the response to avoid actual API calls,
        # but we check the call arguments
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "The variable is `MAX_RETRIES` set to 5."
        self.rag.client.chat.completions.create.return_value = mock_response

        # 3. Method Call
        question = "What is the variable name for retries?"
        video_id = "test_vid"
        answer, sources = self.rag.answer_question(question, video_id)

        # 4. Assertions on the PROMPT sent to OpenAI
        # We need to verify that our new formatting logic is present in the prompt
        call_args = self.rag.client.chat.completions.create.call_args
        _, kwargs = call_args
        messages = kwargs["messages"]

        system_msg = next(m for m in messages if m["role"] == "system")["content"]
        user_msg = next(m for m in messages if m["role"] == "user")["content"]

        # Check System Prompt Upgrades
        self.assertIn(
            "Expert Technical Tutor",
            system_msg,
            "System prompt should have the new role",
        )
        self.assertIn(
            "Markdown code blocks",
            system_msg,
            "System prompt should enforce Markdown formatting",
        )

        # Check Context Injection Formatting (XML-like)
        self.assertIn("<context_slice", user_msg, "User prompt should use XML-like context tags")
        self.assertIn(
            "<source_type>VISUAL</source_type>",
            user_msg,
            "Visual sources should be explicitly tagged",
        )
        self.assertIn("const MAX_RETRIES = 5;", user_msg, "The visual content should be present")


if __name__ == "__main__":
    unittest.main()
