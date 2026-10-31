"""Authorized document inventory, bounded previews, downloads, and deletion."""

from datetime import date
import mimetypes
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4

from core import database
from core.documents import MIME_TYPES


PREVIEW_LIMIT = 5000
DOCUMENT_QUERY = """
    SELECT d.*, u.name AS uploader_name, u.email AS uploader_email
    FROM documents d JOIN users u ON u.id = d.uploaded_by
"""


class ManagementError(ValueError):
    """An actionable management error safe to display in the UI."""


def _actor(db: sqlite3.Connection, user_id: int) -> dict:
    if type(user_id) is not int or user_id < 1:
        raise ManagementError("Sign in with a valid account to manage documents.")
    row = db.execute("SELECT id, role FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None or row["role"] not in ("admin", "editor", "viewer"):
        raise ManagementError("Sign in with a valid account to manage documents.")
    return dict(row)


def _document(db: sqlite3.Connection, user_id: int, document_id: int, action="view") -> tuple[dict, dict]:
    actor = _actor(db, user_id)
    if action != "view" and actor["role"] == "viewer":
        raise ManagementError("Viewer accounts can only view metadata and preview extracted text.")
    if type(document_id) is not int or document_id < 1:
        raise ManagementError("Document not found or you do not have permission to access it.")
    conditions, parameters = ["d.id = ?"], [document_id]
    if actor["role"] == "editor":
        conditions.append("d.uploaded_by = ?")
        parameters.append(actor["id"])
    row = db.execute(DOCUMENT_QUERY + " WHERE " + " AND ".join(conditions), parameters).fetchone()
    if row is None:
        raise ManagementError("Document not found or you do not have permission to access it.")
    return dict(row), actor


def _date_filter(value: date | str | None) -> str | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (ValueError, TypeError) as exc:
        raise ManagementError("Use a valid upload date in YYYY-MM-DD format.") from exc


def get_inventory(user_id: int, *, search="", extension=None, status=None,
                  uploader_id=None, date_from=None, date_to=None) -> list[dict]:
    """Filter metadata in SQLite after applying the caller's current access scope."""
    start, end = _date_filter(date_from), _date_filter(date_to)
    if start and end and start > end:
        raise ManagementError("The start upload date must be on or before the end date.")
    with database.connection() as db:
        actor = _actor(db, user_id)
        conditions, parameters = [], []
        if actor["role"] == "editor":
            conditions.append("d.uploaded_by = ?")
            parameters.append(actor["id"])
        if search.strip():
            term = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("(d.original_filename LIKE ? ESCAPE '\\' OR u.name LIKE ? ESCAPE '\\' OR u.email LIKE ? ESCAPE '\\')")
            parameters.extend([f"%{term}%"] * 3)
        for column, value in (("d.extension", extension), ("d.extraction_status", status), ("d.uploaded_by", uploader_id)):
            if value is not None:
                conditions.append(f"{column} = ?")
                parameters.append(value)
        if start:
            conditions.append("date(d.uploaded_at) >= ?")
            parameters.append(start)
        if end:
            conditions.append("date(d.uploaded_at) <= ?")
            parameters.append(end)
        query = DOCUMENT_QUERY
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        rows = db.execute(query + " ORDER BY d.uploaded_at DESC, d.id DESC", parameters).fetchall()
    return [dict(row) for row in rows]


def summarize_inventory(documents: list[dict]) -> dict:
    """Summarize records already scoped and filtered by get_inventory."""
    return {
        "total_documents": len(documents),
        "successful_extractions": sum(document["extraction_status"] == "completed" for document in documents),
        "failed_extractions": sum(document["extraction_status"] == "failed" for document in documents),
        "total_extracted_characters": sum(document["extracted_characters"] for document in documents),
    }


def get_document(user_id: int, document_id: int) -> dict:
    with database.connection() as db:
        document, actor = _document(db, user_id, document_id)
    document["can_download"] = actor["role"] in ("admin", "editor")
    document["can_delete"] = document["can_download"]
    return document


def _safe_path(directory: str, filename: str) -> Path:
    try:
        root = database.DB_PATH.parent.resolve()
        path = root / directory / filename
        # Reject redirected folders, junctions, and links to other documents.
        if path.resolve() != path or path.is_symlink():
            raise ManagementError("The stored document path is unsafe. Contact an administrator.")
    except (OSError, RuntimeError) as exc:
        raise ManagementError("The stored document path could not be verified safely.") from exc
    return path


def _document_paths(document: dict) -> tuple[Path, Path]:
    stored = document["stored_filename"]
    extension = document["extension"]
    if not isinstance(stored, str) or "/" in stored or "\\" in stored or extension not in MIME_TYPES:
        raise ManagementError("The stored document filename is invalid. Contact an administrator.")
    name = Path(stored)
    try:
        UUID(name.stem)
    except ValueError as exc:
        raise ManagementError("The stored document filename is invalid. Contact an administrator.") from exc
    if name.suffix != extension:
        raise ManagementError("The stored document filename does not match its file type.")
    return _safe_path("uploads", stored), _safe_path("extracted_text", name.stem + ".txt")


def preview_document(user_id: int, document_id: int) -> dict:
    with database.connection() as db:
        document, _ = _document(db, user_id, document_id)
        _, text_path = _document_paths(document)
        try:
            with text_path.open(encoding="utf-8") as extracted:
                text = extracted.read(PREVIEW_LIMIT + 1)
        except FileNotFoundError as exc:
            raise ManagementError("The extracted text file is missing. Document metadata is still available.") from exc
        except (OSError, UnicodeError) as exc:
            raise ManagementError("The extracted text file could not be read. Check its encoding and file permissions.") from exc
    return {"text": text[:PREVIEW_LIMIT], "truncated": len(text) > PREVIEW_LIMIT, "limit": PREVIEW_LIMIT}


def download_original(user_id: int, document_id: int) -> dict:
    with database.connection() as db:
        document, _ = _document(db, user_id, document_id, action="download")
        original_path, _ = _document_paths(document)
        try:
            content = original_path.read_bytes()
        except FileNotFoundError as exc:
            raise ManagementError("The original file is missing and cannot be downloaded. Metadata and any extracted text remain available.") from exc
        except OSError as exc:
            raise ManagementError("The original file could not be read. Check file permissions.") from exc
    return {
        "content": content, "filename": document["original_filename"],
        "mime_type": mimetypes.guess_type(document["original_filename"])[0] or "application/octet-stream",
    }


def delete_document(user_id: int, document_id: int, *, confirmed=False) -> dict:
    """Stage files under the data root; restore them if the metadata transaction fails."""
    from core.index_errors import IndexingError
    from core.vector_store import vector_transaction
    if confirmed is not True:
        raise ManagementError("Confirm deletion explicitly before deleting this document.")
    staged = []
    missing = []
    staging_directory = None
    staging_created = False
    try:
        with vector_transaction() as transaction:
            db = transaction.db
            # Keep identity, ownership, file staging, and metadata removal in one write transaction.
            document, _ = _document(db, user_id, document_id, action="delete")
            original_path, text_path = _document_paths(document)
            staging_directory = _safe_path(".deletion_staging", uuid4().hex)
            staging_directory.mkdir(parents=True)
            staging_created = True
            for label, source in (("original", original_path), ("extracted text", text_path)):
                destination = staging_directory / ("original" if label == "original" else "extracted_text")
                try:
                    if source.exists() and not source.is_file():
                        raise ManagementError(f"The stored {label} path is not a file. Deletion was cancelled.")
                    source.rename(destination)
                except FileNotFoundError:
                    missing.append(f"The {label} file was already missing.")
                else:
                    staged.append((source, destination))
            transaction.remove_documents([document_id])
            db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    except (ManagementError, IndexingError, OSError, sqlite3.Error) as exc:
        restore_failed = False
        for source, destination in reversed(staged):
            try:
                destination.rename(source)
            except OSError:
                restore_failed = True
        _remove_empty_staging(staging_directory if staging_created else None)
        if restore_failed:
            raise ManagementError("Deletion failed and metadata was retained. Some stored files need administrator recovery from data/.deletion_staging.") from exc
        if isinstance(exc, ManagementError):
            raise
        raise ManagementError("The document could not be deleted. Metadata and files were retained; check database and file permissions.") from exc

    warnings = list(missing)
    for _, destination in staged:
        try:
            destination.unlink()
        except OSError:
            warnings.append("A temporary deletion file could not be removed. Ask an administrator to clean data/.deletion_staging after checking file permissions.")
    _remove_empty_staging(staging_directory)
    return {"document_id": document_id, "warnings": warnings}


def _remove_empty_staging(directory: Path | None) -> None:
    if directory is not None:
        try:
            directory.rmdir()
            directory.parent.rmdir()
        except OSError:
            pass
