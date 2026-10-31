from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from core import database
from core.auth import login, register
from core.documents import ingest_document
from core.index_errors import IndexingError
from core.indexing import index_status
from test_indexing import fake_embeddings


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


class IndexingUITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        patcher = patch.object(database, "DB_PATH", Path(self.directory.name) / "app.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.initialize_database()
        self.admin = login("admin@enterprise.com", "Admin@123")
        self.editor = register("Test Editor", "editor@example.com", "Strong@123", "editor")
        self.viewer = register("Test Viewer", "viewer@example.com", "Strong@123", "viewer")
        embedder = patch("core.indexing.embed_chunks", side_effect=fake_embeddings)
        self.embedder = embedder.start()
        self.addCleanup(embedder.stop)

    def app(self, user, page="Index Documents"):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20)
        app.session_state["user"] = user
        app.session_state["navigation"] = page
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_empty_index_page_has_clear_error_and_no_model_load(self):
        app = self.app(self.admin)
        self.assertIn("no successfully extracted documents", app.error[0].value)
        self.embedder.assert_not_called()

    def test_editor_inventory_selection_index_results_and_dashboard_status(self):
        ingest_document("admin.txt", b"Admin document", "text/plain", self.admin["id"])
        document = ingest_document("editor.md", b"# Topic\nEditor content", "text/markdown", self.editor["id"])
        app = self.app(self.editor)
        self.assertEqual(list(app.dataframe[0].value["Filename"]), ["editor.md"])
        self.embedder.assert_not_called()
        app.multiselect(key="index_selection").set_value([document["id"]]).run()
        next(button for button in app.button if button.label == "Create or refresh index").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Indexed successfully" in message.value for message in app.success))
        self.assertEqual(app.dataframe[0].value.iloc[0]["Indexing status"], "Indexed")
        self.assertEqual(index_status(self.editor["id"])["indexed_documents"], 1)
        app.sidebar.radio[0].set_value("Dashboard").run()
        self.assertFalse(app.exception)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Indexed documents"), "1")
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Project phase"), "06")

    def test_viewer_has_dashboard_status_but_no_index_page(self):
        app = self.app(self.viewer, "Dashboard")
        self.assertNotIn("Index Documents", app.sidebar.radio[0].options)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Total chunks"), "0")
        self.assertTrue(any("sentence-transformers/all-MiniLM-L6-v2" in item.value for item in app.caption))

    def test_model_error_is_shown_and_can_be_retried(self):
        document = ingest_document("report.txt", b"Document content", "text/plain", self.admin["id"])
        self.embedder.side_effect = IndexingError("The embedding model could not be downloaded or loaded.")
        app = self.app(self.admin)
        app.multiselect(key="index_selection").set_value([document["id"]]).run()
        next(button for button in app.button if button.label == "Create or refresh index").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("could not be downloaded" in item.value for item in app.error))
        self.assertEqual(index_status(self.admin["id"])["total_chunks"], 0)

    def test_rebuild_button_indexes_only_accessible_documents(self):
        ingest_document("admin.txt", b"Admin document", "text/plain", self.admin["id"])
        ingest_document("editor.txt", b"Editor document", "text/plain", self.editor["id"])
        app = self.app(self.editor)
        next(button for button in app.button if button.label == "Rebuild accessible index").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(index_status(self.editor["id"])["indexed_documents"], 1)
        self.assertEqual(index_status(self.admin["id"])["indexed_documents"], 1)


if __name__ == "__main__":
    unittest.main()
