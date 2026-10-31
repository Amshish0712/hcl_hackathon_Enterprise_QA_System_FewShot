import os
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from streamlit.testing.v1 import AppTest

from core import database
from core.auth import login, register
from core.document_management import delete_document
from core.documents import ingest_document
from core.indexing import index_documents
from core.trust import CONFLICT_WARNING, NO_EVIDENCE
from test_rag import TEST_CREDENTIAL, completion, controlled_embeddings, query_vector


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


class ChatUITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for patcher in (
            patch.object(database, "DB_PATH", Path(self.directory.name) / "app.db"),
            patch("core.chat_config.PROJECT_ROOT", Path(self.directory.name)),
            patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": TEST_CREDENTIAL, "GEMINI_MODEL": "gemini-3.1-flash-lite", "RAG_MIN_RELEVANCE_SCORE": "0.30"}),
            patch("core.indexing.embed_chunks", side_effect=controlled_embeddings),
            patch("core.retrieval.embed_question", side_effect=lambda question: controlled_embeddings([
                {"document_name": "editor" if "approval" in question.lower() else "admin"}
            ])),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        database.initialize_database()
        self.admin = login("admin@enterprise.com", "Admin@123")
        self.editor = register("Test Editor", "editor@example.com", "Strong@123", "editor")
        self.viewer = register("Test Viewer", "viewer@example.com", "Strong@123", "viewer")
        self.admin_doc = ingest_document("admin.md", b"# Retention\nRecords are retained for 30 days.", "text/markdown", self.admin["id"])
        self.editor_doc = ingest_document("editor.md", b"# Approvals\nManagers approve requests.", "text/markdown", self.editor["id"])
        index_documents(self.admin["id"], [self.admin_doc["id"], self.editor_doc["id"]])
        factory = patch("google.genai.Client")
        self.client_factory = factory.start()
        self.addCleanup(factory.stop)
        self.client = self.client_factory.return_value.__enter__.return_value
        self.client.models.generate_content.return_value = completion()

    def app(self, user, page="Chat"):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20)
        app.session_state["user"] = user
        app.session_state["navigation"] = page
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_all_roles_chat_evidence_panels_and_editor_scope(self):
        for user in (self.admin, self.editor, self.viewer):
            with self.subTest(role=user["role"]):
                app = self.app(user)
                question = "Which approval steps are required?" if user["role"] == "editor" else "What is the retention period?"
                self.client.models.generate_content.return_value = completion(
                    "Managers approve requests. [S1]" if user["role"] == "editor" else "Records are retained for 30 days. [S1]")
                app.chat_input[0].set_value(question).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.chat_message), 2)
                self.assertEqual(app.expander[0].label, "Retrieved evidence")
                self.assertFalse(app.expander[0].proto.expanded)
                evidence = app.session_state["chat_messages"][1]["evidence"]
                self.assertTrue(evidence)
                stored = app.session_state["chat_messages"][1]
                self.assertEqual(len(stored["sources"]), 1)
                self.assertTrue(any("(#source-" in item.value for item in app.chat_message[1].markdown))
                self.assertEqual(app.chat_message[1].markdown[1].value, "#### Sources")
                self.assertTrue(any("Top retrieval score:" in item.value for item in app.expander[0].caption))
                if user["role"] == "editor":
                    self.assertEqual({item["document_id"] for item in evidence}, {self.editor_doc["id"]})
                self.assertNotIn(TEST_CREDENTIAL, str(app))

    def test_clear_conversation_action_and_dashboard_message_count(self):
        app = self.app(self.viewer)
        app.chat_input[0].set_value("What is the retention period?").run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value("Dashboard").run()
        self.assertFalse(app.exception)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Session messages"), "2")
        app.sidebar.radio[0].set_value("Chat").run()
        app.button(key="clear_conversation").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["chat_messages"], [])
        self.assertEqual(len(app.chat_message), 0)

    def test_missing_key_and_no_index_warnings_do_not_break_app(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            app = self.app(self.viewer)
            self.assertIn("GEMINI_API_KEY", app.error[0].value)
            self.assertTrue(app.chat_input[0].disabled)
        delete_document(self.admin["id"], self.admin_doc["id"], confirmed=True)
        delete_document(self.admin["id"], self.editor_doc["id"], confirmed=True)
        app = self.app(self.viewer)
        self.assertIn("No indexed documents", app.warning[0].value)
        self.assertTrue(app.chat_input[0].disabled)
        self.client_factory.assert_not_called()

    def test_gemini_failure_is_friendly_and_never_shows_the_credential(self):
        self.client.models.generate_content.side_effect = RuntimeError("Provider details " + TEST_CREDENTIAL)
        app = self.app(self.viewer)
        app.chat_input[0].set_value("What is the retention period?").run()
        self.assertFalse(app.exception)
        self.assertIn("Gemini could not complete", app.error[0].value)
        self.assertNotIn(TEST_CREDENTIAL, str(app))

    def test_suggested_question_buttons_submit_a_real_chat_turn(self):
        self.client.models.generate_content.return_value = completion("Managers approve requests. [S1]")
        app = self.app(self.editor)
        next(button for button in app.button if button.label == "Which approval steps are required?").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["chat_messages"][0]["content"], "Which approval steps are required?")
        self.assertTrue(self.client.models.generate_content.called)

    def test_low_relevance_displays_exact_fallback_and_no_sources(self):
        vector = np.zeros((1, 384), dtype="float32")
        vector[0, 2] = 1
        with patch("core.retrieval.embed_question", return_value=vector):
            app = self.app(self.viewer)
            app.chat_input[0].set_value("An unrelated question").run()
        self.assertFalse(app.exception)
        stored = app.session_state["chat_messages"][1]
        self.assertEqual(stored["content"], NO_EVIDENCE)
        self.assertEqual(stored["sources"], [])
        self.assertFalse(any(item.value == "#### Sources" for item in app.markdown))
        self.client_factory.assert_not_called()

    def test_conflict_warning_and_single_s1_source_are_shown(self):
        other = ingest_document("admin-new.md", b"# Retention\nRecords are retained for 60 days.", "text/markdown", self.admin["id"])
        index_documents(self.admin["id"], [other["id"]])
        def reply(**kwargs):
            top = json.loads(kwargs["contents"])["retrieved_passages_untrusted"][0]
            return completion(top["chunk_text"].splitlines()[-1] + " [S1]")
        self.client.models.generate_content.side_effect = reply
        app = self.app(self.viewer)
        app.chat_input[0].set_value("What is the retention period?").run()
        self.assertFalse(app.exception)
        self.assertTrue(any(message.value == CONFLICT_WARNING for message in app.warning))
        stored = app.session_state["chat_messages"][1]
        self.assertEqual(len(stored["sources"]), 1)
        assistant = app.chat_message[1]
        self.assertEqual(re.findall(r"\[S[1-4]\]", assistant.markdown[0].value), ["[S1]"])
        cards = [item.value for item in assistant.text if re.match(r"^\[S\d+\] ", item.value)]
        self.assertEqual(cards, [f"[S1] {stored['evidence'][0]['filename']}"])
        self.assertTrue(any(f"Chunk ID: {stored['evidence'][0]['chunk_id']} · Retrieval rank: 1" in item.value for item in assistant.caption))
        self.assertFalse(any(re.search(r"\[S[2-4]\]", item.value) for item in assistant.text))
        self.assertEqual(len(app.expander[0].code), len(stored["evidence"]))

    def test_citing_a_lower_ranked_chunk_falls_back_instead_of_showing_it(self):
        other = ingest_document("admin-approvals.md", b"# Approvals\nManagers approve requests.", "text/markdown", self.admin["id"])
        index_documents(self.admin["id"], [other["id"]])
        def reply(**kwargs):
            chosen = json.loads(kwargs["contents"])["retrieved_passages_untrusted"][1]
            return completion(f"{chosen['chunk_text'].splitlines()[-1]} [{chosen['source_label']}]")
        self.client.models.generate_content.side_effect = reply
        app = self.app(self.viewer)
        app.chat_input[0].set_value("What do the documents state?").run()
        self.assertFalse(app.exception)
        stored = app.session_state["chat_messages"][1]
        self.assertEqual(stored["content"], NO_EVIDENCE)
        self.assertEqual(stored["sources"], [])
        self.assertFalse(any(item.value == "#### Sources" for item in app.markdown))

    def test_debug_document_list_is_hidden_and_evidence_remains_secret_free(self):
        for user in (self.admin, self.viewer):
            with self.subTest(role=user["role"]):
                app = self.app(user)
                app.chat_input[0].set_value("What is the retention period?").run()
                self.assertFalse(app.exception)
                stored = app.session_state["chat_messages"][1]
                self.assertEqual(len(app.expander[0].dataframe), 0)
                self.assertEqual(len(app.expander[0].code), len(stored["evidence"]))
                if user["role"] == "admin":
                    self.assertTrue(stored["retrieval_debug"])
                else:
                    self.assertNotIn("retrieval_debug", stored)
                self.assertNotIn(TEST_CREDENTIAL, str(app))


if __name__ == "__main__":
    unittest.main()
