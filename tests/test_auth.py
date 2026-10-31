import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import database
from core.auth import ValidationError, get_user, login, register
from core.permissions import allowed_pages, can_access
from core.security import verify_password


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_patch = patch.object(database, "DB_PATH", Path(self.directory.name) / "app.db")
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)
        database.initialize_database()

    def test_admin_bootstrap_is_idempotent_and_preserves_existing_password(self):
        admin = login("ADMIN@ENTERPRISE.COM", "Admin@123")
        self.assertEqual(admin["role"], "admin")
        with database.connection() as db:
            original_hash = db.execute("SELECT password_hash FROM users").fetchone()[0]
        database.initialize_database()
        with database.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM users").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT password_hash FROM users").fetchone()[0], original_hash)
            # An existing administrator's credentials must never be reset.
            from core.security import hash_password
            db.execute("UPDATE users SET password_hash = ?", (hash_password("Changed@123"),))
        database.initialize_database()
        self.assertIsNone(login("admin@enterprise.com", "Admin@123"))
        self.assertIsNotNone(login("admin@enterprise.com", "Changed@123"))

    def test_registration_login_and_hash_storage(self):
        for role in ("viewer", "editor"):
            with self.subTest(role=role):
                user = register("  Test User  ", f" {role.upper()}@Example.com ", "Strong@123", role)
                self.assertEqual(user["name"], "Test User")
                self.assertEqual(login(user["email"], "Strong@123"), user)
                self.assertEqual(get_user(user["id"]), user)
                self.assertNotIn("password_hash", user)
                with database.connection() as db:
                    stored = db.execute("SELECT password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()[0]
                self.assertNotEqual(stored, "Strong@123")
                self.assertTrue(stored.startswith("$2"))
                self.assertTrue(verify_password("Strong@123", stored))

    def test_registration_rejects_admin_unknown_role_and_invalid_fields(self):
        cases = [
            ("Test User", "test@example.com", "Strong@123", "admin"),
            ("Test User", "test@example.com", "Strong@123", "owner"),
            ("A", "test@example.com", "Strong@123", "viewer"),
            ("Test User", "invalid-email", "Strong@123", "viewer"),
            ("Test User", "test@example.com", "short", "viewer"),
            ("Test User", "test@example.com", "é" * 37, "viewer"),
            ("Test User", "admin@enterprise.com", "Strong@123", "editor"),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValidationError):
                    register(*arguments)

    def test_duplicate_email_wrong_password_and_sql_input(self):
        register("Test User", "user@example.com", "Strong@123", "viewer")
        with self.assertRaises(ValidationError):
            register("Another User", "USER@EXAMPLE.COM", "Strong@123", "editor")
        self.assertIsNone(login("user@example.com", "wrong"))
        self.assertIsNone(login("missing@example.com", "Strong@123"))
        self.assertIsNone(login("' OR 1=1 --", "Strong@123"))
        self.assertIsNone(login("user@example.com", "a" * 73))

    def test_every_role_page_combination_and_unknown_inputs(self):
        expected = {
            "admin": {"Dashboard", "User Management", "Upload Documents", "Document Management", "Index Documents", "Chat"},
            "editor": {"Dashboard", "Upload Documents", "Document Management", "Index Documents", "Chat"},
            "viewer": {"Dashboard", "Document Management", "Chat"},
        }
        all_pages = set().union(*expected.values()) | {"Unknown Page"}
        for role in (*expected, "unknown", ""):
            self.assertEqual(set(allowed_pages(role)), expected.get(role, set()))
            for page in all_pages:
                self.assertEqual(can_access(role, page), page in expected.get(role, set()))


if __name__ == "__main__":
    unittest.main()
