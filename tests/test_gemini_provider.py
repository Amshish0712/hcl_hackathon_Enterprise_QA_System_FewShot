from io import StringIO
import logging
import os
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from google.genai import errors
import httpx

from core.chat_config import ChatError, GeminiSettings, protect_provider_logs, settings
from core.gemini_provider import gemini_completion, safe_error_message


TEST_SECRET = "provider-test-placeholder-not-a-real-key"


class GeminiProviderTests(unittest.TestCase):
    def test_typed_api_errors_have_safe_actionable_status(self):
        cases = [
            (400, "INVALID_ARGUMENT", "Gemini authentication failed", [{"reason": "API_KEY_INVALID"}]),
            (401, "UNAUTHENTICATED", "Gemini authentication failed", []),
            (403, "PERMISSION_DENIED", "Gemini authentication failed", []),
            (429, "RESOURCE_EXHAUSTED", "quota or rate limit", []),
            (404, "NOT_FOUND", "model is unavailable", []),
            (503, "UNAVAILABLE", "temporarily unavailable", []),
            (400, "INVALID_ARGUMENT", "rejected the request", []),
        ]
        for code, status, expected, details in cases:
            with self.subTest(code=code, status=status):
                error_class = errors.ServerError if code >= 500 else errors.ClientError
                error = error_class(code, {"error": {"code": code, "status": status,
                                    "message": TEST_SECRET, "details": details}})
                with patch("google.genai.Client") as factory:
                    factory.return_value.__enter__.return_value.models.generate_content.side_effect = error
                    with self.assertRaises(ChatError) as caught:
                        gemini_completion(GeminiSettings("gemini-3.1-flash-lite", TEST_SECRET),
                                          [{"content": "Grounded system instruction"}, {"content": "Question and evidence"}], "No evidence")
                text = str(caught.exception)
                self.assertIn(expected, text)
                self.assertIn(f"HTTP {code}", text)
                self.assertIn(status, text)
                self.assertIn(error_class.__name__, text)
                self.assertNotIn(TEST_SECRET, text)
                self.assertNotIn(TEST_SECRET, "".join(traceback.format_exception(caught.exception)))

    def test_network_failures_have_safe_type_without_request_details(self):
        for error in (httpx.ConnectError(TEST_SECRET), httpx.ReadTimeout(TEST_SECRET), ConnectionError(TEST_SECRET)):
            text = safe_error_message(error)
            self.assertIn("network request failed or timed out", text)
            self.assertIn(type(error).__name__, text)
            self.assertNotIn(TEST_SECRET, text)

    def test_untrusted_status_or_type_is_not_echoed(self):
        error = RuntimeError(TEST_SECRET)
        error.code = TEST_SECRET
        error.status = TEST_SECRET
        error.message = TEST_SECRET
        text = safe_error_message(error)
        self.assertNotIn(TEST_SECRET, text)
        self.assertIn("GeminiError", text)

    def test_raw_sdk_and_transport_logs_are_suppressed(self):
        for name in ("google.genai._api_client", "google.genai.models", "httpx", "httpcore.connection"):
            output = StringIO()
            handler = logging.StreamHandler(output)
            logger = logging.getLogger(name)
            logger.addHandler(handler)
            try:
                protect_provider_logs(TEST_SECRET)
                logger.warning("Request diagnostic %s", TEST_SECRET)
            finally:
                logger.removeHandler(handler)
            self.assertEqual(output.getvalue(), "")

    def test_default_model_placeholder_and_environment_override(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.chat_config.PROJECT_ROOT", Path(directory)), patch.dict(os.environ, {
                "LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "your_gemini_api_key_here", "GEMINI_MODEL": "",
            }):
                self.assertEqual(settings().model, "gemini-3.1-flash-lite")
                self.assertFalse(settings().configured)
                with patch.dict(os.environ, {"GEMINI_API_KEY": TEST_SECRET, "GEMINI_MODEL": "gemini-2.5-flash"}):
                    self.assertTrue(settings().configured)
                    self.assertEqual(settings().model, "gemini-2.5-flash")
                    self.assertNotIn(TEST_SECRET, repr(settings()))

    def test_tool_response_is_not_executed_and_empty_response_is_safe(self):
        with patch("google.genai.Client") as factory:
            client = factory.return_value.__enter__.return_value
            config = GeminiSettings("gemini-3.1-flash-lite", TEST_SECRET)
            messages = [{"content": "System"}, {"content": "Question and passages"}]
            client.models.generate_content.return_value = SimpleNamespace(function_calls=[object()], text="")
            self.assertEqual(gemini_completion(config, messages, "Exact refusal"), "Exact refusal")
            client.models.generate_content.return_value = SimpleNamespace(function_calls=None, text=" ")
            with self.assertRaisesRegex(ChatError, "Gemini returned no answer"):
                gemini_completion(config, messages, "Exact refusal")


if __name__ == "__main__":
    unittest.main()
