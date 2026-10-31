from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from core import database
from core.auth import login, register
from core.documents import ingest_document
from core.document_management import get_inventory


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


class ManagementUITests(unittest.TestCase):
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
        self.admin_doc = ingest_document("admin.txt", b"a" * 5001, "text/plain", self.admin["id"])
        self.editor_doc = ingest_document("editor.md", b"# Editor", "text/markdown", self.editor["id"])

    def app(self, user):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20)
        app.session_state["user"] = user
        app.session_state["navigation"] = "Document Management"
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_viewer_can_preview_but_has_no_download_delete_or_upload_controls(self):
        app = self.app(self.viewer)
        self.assertNotIn("Upload Documents", app.sidebar.radio[0].options)
        self.assertEqual(len(app.dataframe[0].value), 2)
        app.selectbox(key="management_document").set_value(self.admin_doc["id"]).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.code[0].value, "a" * 5000)
        self.assertTrue(any("5,000" in message.value for message in app.info))
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertEqual(len(app.checkbox), 0)
        self.assertEqual([button.label for button in app.button], ["Log out"])
        self.assertTrue(any(caption.value == "PHASES 1 TO 6 · ACCESS, INGESTION, MANAGEMENT, INDEXING, RAG AND TRUST" for caption in app.sidebar.caption))

    def test_editor_inventory_is_own_only_and_manipulated_selection_is_reset(self):
        app = self.app(self.editor)
        self.assertEqual(list(app.dataframe[0].value["Filename"]), ["editor.md"])
        # Inject an invalid selection before creating widgets. Mutating a live
        # selectbox to an out-of-options value fails inside AppTest serialization.
        app = AppTest.from_file(str(APP_PATH), default_timeout=20)
        app.session_state["user"] = self.editor
        app.session_state["navigation"] = "Document Management"
        app.session_state["management_document"] = self.admin_doc["id"]
        app.run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.selectbox(key="management_document").value)
        self.assertEqual(len(app.code), 0)
        app.selectbox(key="management_document").set_value(self.editor_doc["id"]).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.code[0].value, "# Editor")
        self.assertEqual(len(app.get("download_button")), 1)
        self.assertTrue(app.button(key=f"delete_{self.editor_doc['id']}").disabled)

    def test_admin_delete_requires_checkbox_then_removes_document_and_refreshes(self):
        app = self.app(self.admin)
        app.selectbox(key="management_document").set_value(self.editor_doc["id"]).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button(key=f"delete_{self.editor_doc['id']}").disabled)
        app.checkbox(key=f"confirm_delete_{self.admin['id']}_{self.editor_doc['id']}").check().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.button(key=f"delete_{self.editor_doc['id']}").disabled)
        app.button(key=f"delete_{self.editor_doc['id']}").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Document deleted" in message.value for message in app.success))
        self.assertEqual([item["id"] for item in get_inventory(self.admin["id"])], [self.admin_doc["id"]])
        self.assertEqual(len(app.dataframe[0].value), 1)

    def test_search_and_type_filters_update_inventory_and_metrics(self):
        app = self.app(self.viewer)
        app.text_input(key="inventory_search").set_value("admin.txt").run()
        self.assertFalse(app.exception)
        self.assertEqual(list(app.dataframe[0].value["Filename"]), ["admin.txt"])
        self.assertEqual(app.metric[0].value, "1")
        app.text_input(key="inventory_search").set_value("")
        app.selectbox(key="inventory_type").set_value(".md").run()
        self.assertFalse(app.exception)
        self.assertEqual(list(app.dataframe[0].value["Filename"]), ["editor.md"])
        app.selectbox(key="inventory_status").set_value("failed").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "0")
        self.assertEqual(len(app.dataframe), 0)

    def test_missing_files_do_not_break_details_or_confirmed_deletion(self):
        root = database.DB_PATH.parent
        (root / "uploads" / self.editor_doc["stored_filename"]).unlink()
        (root / "extracted_text" / (Path(self.editor_doc["stored_filename"]).stem + ".txt")).unlink()
        app = self.app(self.editor)
        app.selectbox(key="management_document").set_value(self.editor_doc["id"]).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.warning), 2)
        self.assertEqual(len(app.get("download_button")), 0)
        app.checkbox(key=f"confirm_delete_{self.editor['id']}_{self.editor_doc['id']}").check().run()
        app.button(key=f"delete_{self.editor_doc['id']}").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(get_inventory(self.editor["id"])), 0)
        self.assertEqual(len(app.warning), 2)


if __name__ == "__main__":
    unittest.main()
