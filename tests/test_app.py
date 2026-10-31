import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from core import database
from core.auth import register
from core.permissions import allowed_pages
from core.documents import list_documents


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_patch = patch.object(database, "DB_PATH", Path(self.directory.name) / "app.db")
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)
        database.initialize_database()

    def app(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        self.assertFalse(app.exception)
        return app

    def sign_in(self, app, email, password):
        app.text_input[0].set_value(email)
        app.text_input[1].set_value(password)
        app.button[0].click().run()
        self.assertFalse(app.exception)

    def test_login_required_wrong_password_and_registration_role_options(self):
        app = self.app()
        self.assertEqual(app.selectbox[0].options, ["Viewer", "Editor"])
        self.assertEqual(len(app.sidebar.radio), 0)
        self.sign_in(app, "admin@enterprise.com", "wrong")
        self.assertEqual(app.error[0].value, "Invalid email or password.")
        self.assertEqual(len(app.sidebar.radio), 0)

    def test_all_roles_navigation_placeholders_and_logout(self):
        for role in ("admin", "editor", "viewer"):
            with self.subTest(role=role):
                email, password = "admin@enterprise.com", "Admin@123"
                if role != "admin":
                    email, password = f"{role}@example.com", "Strong@123"
                    register(f"Test {role}", email, password, role)
                app = self.app()
                self.sign_in(app, email, password)
                self.assertEqual(app.session_state["user"]["role"], role)
                self.assertNotIn("password_hash", app.session_state["user"])
                self.assertEqual(app.sidebar.radio[0].options, list(allowed_pages(role)))
                for page in allowed_pages(role):
                    app.sidebar.radio[0].set_value(page).run()
                    self.assertFalse(app.exception)
                    self.assertEqual(len(app.get("file_uploader")), 1 if page == "Upload Documents" else 0)
                app.sidebar.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertNotIn("user", app.session_state)
                self.assertEqual(len(app.sidebar.radio), 0)

    def test_registration_form_and_mismatched_passwords(self):
        app = self.app()
        app.text_input[2].set_value("New User")
        app.text_input[3].set_value("new@example.com")
        app.text_input[4].set_value("Strong@123")
        app.text_input[5].set_value("Different@123")
        app.button[1].click().run()
        self.assertEqual(app.error[0].value, "Passwords do not match.")
        app.text_input[2].set_value("New User")
        app.text_input[3].set_value("new@example.com")
        app.text_input[4].set_value("Strong@123")
        app.text_input[5].set_value("Strong@123")
        app.selectbox[0].set_value("editor")
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        self.sign_in(app, "new@example.com", "Strong@123")
        self.assertEqual(app.session_state["user"]["role"], "editor")

    def test_session_role_is_reloaded_from_database(self):
        user = register("Test Editor", "editor@example.com", "Strong@123", "editor")
        app = self.app()
        self.sign_in(app, "editor@example.com", "Strong@123")
        app.sidebar.radio[0].set_value("Upload Documents").run()
        with database.connection() as db:
            db.execute("UPDATE users SET role = 'viewer' WHERE id = ?", (user["id"],))
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["user"]["role"], "viewer")
        self.assertEqual(app.sidebar.radio[0].value, "Dashboard")
        self.assertEqual(app.sidebar.radio[0].options, ["Dashboard", "Document Management", "Chat"])

    def test_deleted_account_loses_access(self):
        user = register("Test Viewer", "viewer@example.com", "Strong@123", "viewer")
        app = self.app()
        self.sign_in(app, "viewer@example.com", "Strong@123")
        with database.connection() as db:
            db.execute("DELETE FROM users WHERE id = ?", (user["id"],))
        app.run()
        self.assertFalse(app.exception)
        self.assertNotIn("user", app.session_state)
        self.assertEqual(len(app.sidebar.radio), 0)

    def test_direct_renderer_calls_enforce_permissions(self):
        for role, page in (
            ("viewer", "User Management"),
            ("viewer", "Upload Documents"),
            ("viewer", "Index Documents"),
            ("editor", "User Management"),
            ("unknown", "Dashboard"),
        ):
            with self.subTest(role=role, page=page):
                script = (
                    "from ui.pages import render_page\n"
                    f"render_page({page!r}, {{'role': {role!r}}})\n"
                )
                app = AppTest.from_string(script).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.error[0].value, "You do not have permission to access this page.")
                self.assertEqual(len(app.subheader), 0)

    def test_upload_batch_continues_after_rejected_file_and_shows_history(self):
        app = self.app()
        self.sign_in(app, "admin@enterprise.com", "Admin@123")
        files = [
            SimpleNamespace(name="empty.txt", size=0, type="text/plain", getvalue=lambda: b""),
            SimpleNamespace(name="valid.md", size=7, type="text/markdown", getvalue=lambda: b"# Hello"),
        ]
        # AppTest cannot set native file_uploader values; inject upload objects at its boundary.
        import streamlit as st
        real_submit = st.form_submit_button

        def submit(label, *args, **kwargs):
            value = real_submit(label, *args, **kwargs)
            return True if label == "Upload and extract" else value

        with patch("ui.uploads.st.file_uploader", return_value=files), patch("ui.uploads.st.form_submit_button", side_effect=submit):
            app.sidebar.radio[0].set_value("Upload Documents").run()
        self.assertFalse(app.exception)
        self.assertIn("empty", app.error[0].value)
        self.assertIn("successfully", app.success[0].value)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(app.dataframe[0].value.iloc[0]["Filename"], "valid.md")
        self.assertEqual(app.dataframe[0].value.iloc[0]["Extracted characters"], 7)
        self.assertEqual(len(list_documents(app.session_state["user"]["id"])), 1)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(list_documents(app.session_state["user"]["id"])), 1)


if __name__ == "__main__":
    unittest.main()
