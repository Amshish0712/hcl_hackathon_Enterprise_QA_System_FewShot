from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core import database
from core.auth import login, register
from core.documents import ingest_document
from core.document_management import (
    ManagementError, delete_document, download_original, get_document, get_inventory,
    preview_document, summarize_inventory,
)


class DocumentManagementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        patcher = patch.object(database, "DB_PATH", Path(self.directory.name) / "app.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.initialize_database()
        self.admin = login("admin@enterprise.com", "Admin@123")
        self.editor = register("First Editor", "first@example.com", "Strong@123", "editor")
        self.other_editor = register("Second Editor", "second@example.com", "Strong@123", "editor")
        self.viewer = register("Read Only", "viewer@example.com", "Strong@123", "viewer")
        self.admin_doc = ingest_document("admin.txt", b"Admin content", "text/plain", self.admin["id"])
        self.editor_doc = ingest_document("editor.md", b"# Editor content", "text/markdown", self.editor["id"])
        self.other_doc = ingest_document("other.txt", b"Other content", "text/plain", self.other_editor["id"])

    def paths(self, document):
        root = database.DB_PATH.parent
        return (root / "uploads" / document["stored_filename"],
                root / "extracted_text" / (Path(document["stored_filename"]).stem + ".txt"))

    def assert_retained(self, document):
        self.assertEqual(get_document(self.admin["id"], document["id"])["id"], document["id"])
        for path in self.paths(document):
            self.assertTrue(path.exists())

    def test_inventory_scope_for_every_role_and_manipulated_uploader_filter(self):
        all_ids = {self.admin_doc["id"], self.editor_doc["id"], self.other_doc["id"]}
        for user in (self.admin, self.viewer):
            self.assertEqual({item["id"] for item in get_inventory(user["id"])}, all_ids)
        self.assertEqual([item["id"] for item in get_inventory(self.editor["id"])], [self.editor_doc["id"]])
        self.assertEqual(get_inventory(self.editor["id"], uploader_id=self.other_editor["id"]), [])
        self.assertEqual(get_inventory(self.editor["id"], search="admin"), [])

    def test_details_and_preview_enforce_editor_ownership(self):
        for user in (self.admin, self.viewer, self.editor, self.other_editor):
            for document in (self.admin_doc, self.editor_doc, self.other_doc):
                allowed = user["role"] != "editor" or document["uploaded_by"] == user["id"]
                if allowed:
                    details = get_document(user["id"], document["id"])
                    self.assertEqual(details["can_download"], user["role"] != "viewer")
                    self.assertEqual(details["can_delete"], user["role"] != "viewer")
                    self.assertTrue(preview_document(user["id"], document["id"])["text"])
                else:
                    with self.assertRaises(ManagementError):
                        get_document(user["id"], document["id"])
                    with self.assertRaises(ManagementError):
                        preview_document(user["id"], document["id"])

    def test_search_filters_dates_and_role_scoped_summary(self):
        with database.connection() as db:
            db.execute("UPDATE documents SET uploaded_at = '2026-10-01 23:59:59' WHERE id = ?", (self.admin_doc["id"],))
            db.execute("UPDATE documents SET uploaded_at = '2026-10-02 00:00:00' WHERE id = ?", (self.editor_doc["id"],))
            db.execute("UPDATE documents SET extraction_status = 'failed', extracted_characters = 0 WHERE id = ?", (self.other_doc["id"],))
        admin_id = self.admin["id"]
        self.assertEqual(len(get_inventory(admin_id, extension=".txt")), 2)
        self.assertEqual(get_inventory(admin_id, status="failed")[0]["id"], self.other_doc["id"])
        self.assertEqual(get_inventory(admin_id, uploader_id=self.editor["id"])[0]["id"], self.editor_doc["id"])
        self.assertEqual(get_inventory(admin_id, search="FIRST@EXAMPLE")[0]["id"], self.editor_doc["id"])
        self.assertEqual(get_inventory(admin_id, date_from=date(2026, 10, 1), date_to=date(2026, 10, 1))[0]["id"], self.admin_doc["id"])
        self.assertEqual(len(get_inventory(admin_id, extension=".md", status="completed", search="editor")), 1)
        summary = summarize_inventory(get_inventory(self.viewer["id"]))
        self.assertEqual(summary, {"total_documents": 3, "successful_extractions": 2, "failed_extractions": 1,
                                   "total_extracted_characters": len("Admin content") + len("# Editor content")})
        self.assertEqual(summarize_inventory(get_inventory(self.editor["id"]))["total_documents"], 1)
        self.assertEqual(summarize_inventory([])["total_documents"], 0)
        with self.assertRaises(ManagementError):
            get_inventory(admin_id, date_from="2026-11-01", date_to="2026-10-01")
        with self.assertRaises(ManagementError):
            get_inventory(admin_id, date_from="not-a-date")

    def test_search_uses_literal_wildcards_and_parameterized_input(self):
        doc = ingest_document("100%_done.txt", b"literal characters", "text/plain", self.admin["id"])
        self.assertEqual([item["id"] for item in get_inventory(self.admin["id"], search="%_")], [doc["id"]])
        self.assertEqual(get_inventory(self.admin["id"], search="' OR 1=1 --"), [])
        self.assertEqual(get_inventory(self.editor["id"], uploader_id="1 OR 1=1"), [])

    def test_preview_limit_is_characters_and_detects_truncation_exactly(self):
        _, text_path = self.paths(self.editor_doc)
        for length, truncated in ((5000, False), (5001, True)):
            text_path.write_text("é" * length, encoding="utf-8")
            preview = preview_document(self.editor["id"], self.editor_doc["id"])
            self.assertEqual(preview["text"], "é" * 5000)
            self.assertEqual(preview["truncated"], truncated)

    def test_download_requires_admin_or_owning_editor_even_with_manual_ids(self):
        for user, document in ((self.admin, self.editor_doc), (self.editor, self.editor_doc)):
            result = download_original(user["id"], document["id"])
            self.assertEqual(result["content"], b"# Editor content")
            self.assertEqual(result["filename"], "editor.md")
        for user, document in ((self.viewer, self.admin_doc), (self.viewer, self.editor_doc),
                               (self.editor, self.admin_doc), (self.editor, self.other_doc)):
            with self.assertRaises(ManagementError):
                download_original(user["id"], document["id"])

    def test_missing_and_invalid_users_document_ids_and_changed_roles(self):
        for user_id in (None, "1", True, 999999):
            with self.assertRaises(ManagementError):
                get_inventory(user_id)
        for document_id in (None, "1", True, -1, 999999):
            with self.assertRaises(ManagementError):
                download_original(self.admin["id"], document_id)
        with database.connection() as db:
            db.execute("UPDATE users SET role = 'viewer' WHERE id = ?", (self.editor["id"],))
        with self.assertRaises(ManagementError):
            download_original(self.editor["id"], self.editor_doc["id"])
        with self.assertRaises(ManagementError):
            delete_document(self.editor["id"], self.editor_doc["id"], confirmed=True)
        self.assertTrue(preview_document(self.editor["id"], self.other_doc["id"])["text"])

    def test_deletion_requires_explicit_boolean_confirmation(self):
        for confirmed in (False, None, "yes", 1):
            with self.assertRaisesRegex(ManagementError, "Confirm"):
                delete_document(self.admin["id"], self.admin_doc["id"], confirmed=confirmed)
            self.assert_retained(self.admin_doc)

    def test_admin_deletes_any_document_and_both_files_including_txt_name_collision(self):
        for document in (self.other_doc, self.editor_doc, self.admin_doc):
            result = delete_document(self.admin["id"], document["id"], confirmed=True)
            self.assertEqual(result["warnings"], [])
            for path in self.paths(document):
                self.assertFalse(path.exists())
            with self.assertRaises(ManagementError):
                get_document(self.admin["id"], document["id"])
        self.assertEqual(get_inventory(self.admin["id"]), [])
        self.assertFalse((database.DB_PATH.parent / ".deletion_staging").exists())

    def test_editor_can_delete_only_own_document_and_viewer_never_can(self):
        for user, document in ((self.editor, self.admin_doc), (self.editor, self.other_doc),
                               (self.viewer, self.admin_doc), (self.viewer, self.editor_doc)):
            with self.assertRaises(ManagementError):
                delete_document(user["id"], document["id"], confirmed=True)
            self.assert_retained(document)
        delete_document(self.editor["id"], self.editor_doc["id"], confirmed=True)
        self.assertEqual(get_inventory(self.editor["id"]), [])
        self.assert_retained(self.admin_doc)
        self.assert_retained(self.other_doc)

    def test_missing_stored_files_have_clear_errors_and_can_still_be_deleted(self):
        for path in self.paths(self.admin_doc):
            path.unlink()
        with self.assertRaisesRegex(ManagementError, "original file is missing"):
            download_original(self.admin["id"], self.admin_doc["id"])
        with self.assertRaisesRegex(ManagementError, "extracted text file is missing"):
            preview_document(self.viewer["id"], self.admin_doc["id"])
        self.assertEqual(get_document(self.viewer["id"], self.admin_doc["id"])["id"], self.admin_doc["id"])
        result = delete_document(self.admin["id"], self.admin_doc["id"], confirmed=True)
        self.assertEqual(len(result["warnings"]), 2)
        self.assertTrue(all("already missing" in warning for warning in result["warnings"]))

    def test_file_staging_failure_restores_files_and_retains_metadata(self):
        original_rename = Path.rename

        def failing_rename(path, target):
            if path.parent.name == "extracted_text":
                raise PermissionError("Simulated locked file")
            return original_rename(path, target)

        with patch.object(Path, "rename", failing_rename):
            with self.assertRaisesRegex(ManagementError, "retained"):
                delete_document(self.admin["id"], self.admin_doc["id"], confirmed=True)
        self.assert_retained(self.admin_doc)
        self.assertEqual(download_original(self.admin["id"], self.admin_doc["id"])["content"], b"Admin content")
        self.assertFalse((database.DB_PATH.parent / ".deletion_staging").exists())

    def test_database_delete_failure_restores_files_and_metadata(self):
        with database.connection() as db:
            db.execute("""CREATE TRIGGER fail_deletion BEFORE DELETE ON documents
                          BEGIN SELECT RAISE(ABORT, 'Simulated database failure'); END""")
        with self.assertRaisesRegex(ManagementError, "retained"):
            delete_document(self.admin["id"], self.editor_doc["id"], confirmed=True)
        self.assert_retained(self.editor_doc)
        self.assertEqual(preview_document(self.editor["id"], self.editor_doc["id"])["text"], "# Editor content")
        self.assertFalse((database.DB_PATH.parent / ".deletion_staging").exists())

    def test_unsafe_stored_paths_cannot_be_previewed_downloaded_or_deleted(self):
        outside = database.DB_PATH.parent / "outside.txt"
        outside.write_text("Do not read or delete", encoding="utf-8")
        for stored in ("../outside.txt", "..\\outside.txt", "/outside.txt", "not-a-uuid.txt"):
            with database.connection() as db:
                db.execute("UPDATE documents SET stored_filename = ? WHERE id = ?", (stored, self.admin_doc["id"]))
            with self.assertRaises(ManagementError):
                preview_document(self.admin["id"], self.admin_doc["id"])
            with self.assertRaises(ManagementError):
                download_original(self.admin["id"], self.admin_doc["id"])
            with self.assertRaises(ManagementError):
                delete_document(self.admin["id"], self.admin_doc["id"], confirmed=True)
            self.assertEqual(outside.read_text(encoding="utf-8"), "Do not read or delete")


if __name__ == "__main__":
    unittest.main()
