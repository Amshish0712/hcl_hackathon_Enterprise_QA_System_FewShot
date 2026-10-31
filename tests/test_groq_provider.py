import json
import os
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import groq

from core import database
from core.auth import login
from core.chat_config import ChatError, ChatSetupError, GeminiSettings, settings
from core.documents import ingest_document
from core.groq_provider import groq_completion
from core.indexing import index_documents
from core.rag import NO_EVIDENCE, answer_question
from test_rag import controlled_embeddings, query_vector


TEST_SECRET = "groq-unit-test-credential-not-real"


def groq_reply(text, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=tool_calls))])


def status_error(cls, code):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(code, request=request, json={"error": {"message": "details " + TEST_SECRET}})
    return cls("Provider said " + TEST_SECRET, response=response, body=None)


class GroqSettingsTests(unittest.TestCase):
    def test_provider_selection_defaults_and_compound_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch("core.chat_config.PROJECT_ROOT", Path(directory)):
            with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "your_groq_api_key_here", "GROQ_MODEL": ""}):
                self.assertEqual(settings().provider, "groq")
                self.assertEqual(settings().model, "openai/gpt-oss-20b")
                self.assertEqual(settings().key_variable, "GROQ_API_KEY")
                self.assertFalse(settings().configured)
            with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": TEST_SECRET, "GROQ_MODEL": "groq/compound"}):
                # Compound models have built-in web search, so they fall back to the default model.
                self.assertEqual(settings().model, "openai/gpt-oss-20b")
                self.assertTrue(settings().configured)
                self.assertNotIn(TEST_SECRET, repr(settings()))
            with patch.dict(os.environ, {"LLM_PROVIDER": "unknown", "GEMINI_API_KEY": TEST_SECRET}):
                self.assertEqual(settings().provider, "gemini")


class GroqCompletionTests(unittest.TestCase):
    def setUp(self):
        self.config = GeminiSettings("openai/gpt-oss-20b", TEST_SECRET, provider="groq")
        self.messages = [{"role": "system", "content": "System"}, {"role": "user", "content": "Question and passages"}]

    def test_request_has_no_tools_and_low_temperature(self):
        with patch("groq.Groq") as factory:
            create = factory.return_value.chat.completions.create
            create.return_value = groq_reply("Answer [S1]")
            self.assertEqual(groq_completion(self.config, self.messages, "Refusal"), "Answer [S1]")
            parameters = create.call_args.kwargs
            self.assertEqual(parameters["model"], "openai/gpt-oss-20b")
            self.assertEqual(parameters["temperature"], 0.1)
            self.assertEqual(parameters["messages"], self.messages)
            self.assertNotIn("tools", parameters)
            self.assertEqual(factory.call_args.kwargs["timeout"], 30.0)

    def test_tool_calls_and_empty_responses_are_safe(self):
        with patch("groq.Groq") as factory:
            create = factory.return_value.chat.completions.create
            create.return_value = groq_reply("", tool_calls=[object()])
            self.assertEqual(groq_completion(self.config, self.messages, "Refusal"), "Refusal")
            create.return_value = groq_reply(" ")
            with self.assertRaisesRegex(ChatError, "Groq returned no answer"):
                groq_completion(self.config, self.messages, "Refusal")

    def test_errors_are_allowlisted_and_never_expose_the_credential(self):
        cases = [
            (status_error(groq.AuthenticationError, 401), "authentication failed", "HTTP 401"),
            (status_error(groq.RateLimitError, 429), "rate limit", "HTTP 429"),
            (status_error(groq.NotFoundError, 404), "model is unavailable", "HTTP 404"),
            (status_error(groq.InternalServerError, 503), "temporarily unavailable", "HTTP 503"),
            (groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com")), "network request failed", "APIConnectionError"),
            (RuntimeError("raw " + TEST_SECRET), "could not complete", "GroqError"),
        ]
        for error, explanation, label in cases:
            with self.subTest(error=type(error).__name__), patch("groq.Groq") as factory:
                factory.return_value.chat.completions.create.side_effect = error
                with self.assertRaises(ChatError) as caught:
                    groq_completion(self.config, self.messages, "Refusal")
                message = str(caught.exception)
                self.assertIn(explanation, message)
                self.assertIn(label, message)
                self.assertNotIn(TEST_SECRET, message)
                self.assertNotIn(TEST_SECRET, "".join(traceback.format_exception(caught.exception)))


class GroqRagTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for patcher in (
            patch.object(database, "DB_PATH", Path(self.directory.name) / "app.db"),
            patch("core.chat_config.PROJECT_ROOT", Path(self.directory.name)),
            patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": TEST_SECRET, "GROQ_MODEL": "", "RAG_MIN_RELEVANCE_SCORE": "0.30"}),
            patch("core.indexing.embed_chunks", side_effect=controlled_embeddings),
            patch("core.retrieval.embed_question", return_value=query_vector()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        database.initialize_database()
        self.admin = login("admin@enterprise.com", "Admin@123")
        document = ingest_document("admin.md", b"# Retention\nRecords are retained for 30 days.", "text/markdown", self.admin["id"])
        index_documents(self.admin["id"], [document["id"]])
        groq_patch = patch("groq.Groq")
        self.factory = groq_patch.start()
        self.addCleanup(groq_patch.stop)
        self.create = self.factory.return_value.chat.completions.create
        gemini_patch = patch("google.genai.Client")
        self.gemini = gemini_patch.start()
        self.addCleanup(gemini_patch.stop)

    def test_groq_answers_with_the_same_grounding_and_s1_citation(self):
        self.create.return_value = groq_reply("Records are retained for 30 days. [S1]")
        result = answer_question(self.admin["id"], "What is the retention period?")
        self.assertEqual(result["answer"], "Records are retained for 30 days. [S1]")
        self.assertEqual(result["sources"][0]["filename"], "admin.md")
        self.gemini.assert_not_called()
        messages = self.create.call_args.kwargs["messages"]
        self.assertIn("Never cite [S2]", messages[0]["content"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["retrieved_passages_untrusted"][0]["source_label"], "S1")
        self.assertNotIn(TEST_SECRET, json.dumps(messages))

    def test_unsupported_groq_answer_falls_back(self):
        self.create.return_value = groq_reply("Records are retained for 90 days. [S1]")
        self.assertEqual(answer_question(self.admin["id"], "What is the retention period?")["answer"], NO_EVIDENCE)

    def test_missing_groq_key_names_the_right_variable(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaisesRegex(ChatSetupError, "GROQ_API_KEY"):
                answer_question(self.admin["id"], "What is the retention period?")
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
