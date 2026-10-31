"""Authentication services; public user objects never contain password hashes."""

import re
import sqlite3

from core.database import DEFAULT_ADMIN_EMAIL, connection
from core.security import hash_password, verify_password


class ValidationError(ValueError):
    """A registration error that is safe to display to a user."""


def normalize_email(email: str) -> str:
    return email.strip().casefold()


def public_user(row: sqlite3.Row) -> dict:
    return {field: row[field] for field in ("id", "name", "email", "role")}


def register(name: str, email: str, password: str, role: str) -> dict:
    name = name.strip()
    email = normalize_email(email)
    # Enforce the same policy in the service as in the registration form.
    if role not in ("viewer", "editor"):
        raise ValidationError("Registration is available only for viewer and editor roles.")
    if not 2 <= len(name) <= 80 or any(ord(char) < 32 for char in name):
        raise ValidationError("Enter a full name between 2 and 80 characters.")
    if len(email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise ValidationError("Enter a valid email address.")
    if len(password) < 8 or len(password.encode("utf-8")) > 72:
        raise ValidationError("Use at least 8 characters and at most 72 UTF-8 bytes for your password.")
    if email == DEFAULT_ADMIN_EMAIL:
        raise ValidationError("This email is reserved for the default administrator.")

    password_hash = hash_password(password)
    try:
        with connection() as database:
            cursor = database.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                (name, email, password_hash, role),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise ValidationError("An account with this email already exists.") from exc
    return {"id": user_id, "name": name, "email": email, "role": role}


def login(email: str, password: str) -> dict | None:
    with connection() as database:
        row = database.execute(
            "SELECT * FROM users WHERE email = ?", (normalize_email(email),)
        ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return public_user(row)


def get_user(user_id: int | None) -> dict | None:
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        return None
    with connection() as database:
        row = database.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return public_user(row) if row is not None else None
