"""Phase 2 file validation, text extraction, and transactional storage."""

from io import BytesIO
from pathlib import Path
import sqlite3
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from bs4 import BeautifulSoup
from docx import Document
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from core import database
from core.permissions import can_access


MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_DOCX_EXPANDED_SIZE = 100 * 1024 * 1024
MIME_TYPES = {
    ".pdf": {"application/pdf", "application/x-pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip", "application/x-zip-compressed",
    },
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain"},
    ".html": {"text/html", "application/xhtml+xml"},
    ".htm": {"text/html", "application/xhtml+xml"},
}
ALLOWED_EXTENSIONS = tuple(extension.lstrip(".") for extension in MIME_TYPES)


class DocumentError(ValueError):
    """An upload or extraction error that can be displayed to the user."""


def validate_file(filename: str, content: bytes, mime_type: str | None = None) -> str:
    """Return the normalized extension after validating the upload envelope."""
    if not filename or len(filename) > 255 or any(ord(char) < 32 for char in filename):
        raise DocumentError("The filename is missing, too long, or contains invalid characters.")
    if "/" in filename or "\\" in filename:
        raise DocumentError("The filename must not contain a directory path.")
    extension = Path(filename).suffix.lower()
    if extension not in MIME_TYPES:
        raise DocumentError("Unsupported format. Use PDF, DOCX, TXT, Markdown, or HTML.")
    if not content:
        raise DocumentError("The file is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise DocumentError("The file exceeds the maximum size of 20 MB.")
    mime_type = (mime_type or "").split(";", 1)[0].strip().lower()
    if mime_type not in ("", "application/octet-stream") and mime_type not in MIME_TYPES[extension]:
        raise DocumentError(f"The MIME type '{mime_type}' does not match {extension}.")
    if extension == ".pdf" and not content.lstrip().startswith(b"%PDF-"):
        raise DocumentError("The file is not a valid PDF document.")
    if extension == ".docx":
        try:
            with ZipFile(BytesIO(content)) as archive:
                if not {"[Content_Types].xml", "word/document.xml"}.issubset(archive.namelist()):
                    raise DocumentError("The file is not a valid DOCX document.")
                if sum(entry.file_size for entry in archive.infolist()) > MAX_DOCX_EXPANDED_SIZE:
                    raise DocumentError("The DOCX expands beyond the supported 100 MB extraction limit.")
        except BadZipFile as exc:
            raise DocumentError("The DOCX is corrupted or is not a valid Word document.") from exc
    return extension


def _decode_text(content: bytes) -> str:
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    else:
        encodings = ("utf-8-sig", "cp1252")
    for encoding in encodings:
        try:
            text = content.decode(encoding)
            break
        except UnicodeError:
            continue
    else:
        raise DocumentError("The file cannot be decoded as supported text (UTF-8, UTF-16 with BOM, or Windows-1252).")
    if any(ord(char) < 32 and char not in "\t\r\n\f" for char in text):
        raise DocumentError("The file contains binary or corrupted text data.")
    return text


def _docx_text(container) -> str:
    parts = []
    for block in container.iter_inner_content():
        if isinstance(block, Paragraph):
            parts.append(block.text)
        else:
            for row in block.rows:
                parts.append("\t".join(_docx_text(cell) for cell in row.cells))
    return "\n".join(parts)


def extract_text(content: bytes, extension: str) -> str:
    """Extract readable text; reject corrupt, encrypted, and textless documents."""
    extension = extension.lower()
    try:
        if extension == ".pdf":
            reader = PdfReader(BytesIO(content), strict=True)
            if reader.is_encrypted:
                raise DocumentError("Encrypted PDFs are not supported. Upload an unencrypted PDF.")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif extension == ".docx":
            text = _docx_text(Document(BytesIO(content)))
        elif extension in (".txt", ".md"):
            text = _decode_text(content)
        elif extension in (".html", ".htm"):
            soup = BeautifulSoup(_decode_text(content), "html.parser")
            if soup.find() is None:
                raise DocumentError("The file does not contain recognizable HTML markup.")
            for element in soup(["script", "style", "noscript", "template"]):
                element.decompose()
            text = (soup.body or soup).get_text(separator="\n", strip=True)
        else:
            raise DocumentError("This file format is not supported for text extraction.")
    except DocumentError:
        raise
    except Exception as exc:
        # Parser exceptions differ across formats. Do not expose library traces in the UI.
        raise DocumentError(f"The {extension.lstrip('.').upper()} is corrupted or could not be read.") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        message = "No readable text was found; the document is empty or contains only images."
        if extension == ".pdf":
            message += " Scanned PDFs require OCR, which is outside Phase 2."
        raise DocumentError(message)
    return text


def _require_uploader(db: sqlite3.Connection, user_id: int) -> None:
    user = db.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
    if user is None or not can_access(user["role"], "Upload Documents"):
        raise DocumentError("Only an authenticated administrator or editor can upload or view ingestion history.")


def ingest_document(filename: str, content: bytes, mime_type: str | None, uploaded_by: int) -> dict:
    """Persist only fully validated, successfully extracted documents."""
    with database.connection() as db:
        _require_uploader(db, uploaded_by)
    extension = validate_file(filename, content, mime_type)
    text = extract_text(content, extension)
    identifier = uuid4().hex
    stored_filename = identifier + extension
    upload_path = database.DB_PATH.parent / "uploads" / stored_filename
    text_path = database.DB_PATH.parent / "extracted_text" / (identifier + ".txt")
    created_paths = []
    try:
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        with database.connection() as db:
            # A role may have changed while a larger document was being extracted.
            _require_uploader(db, uploaded_by)
            with upload_path.open("xb") as uploaded:
                created_paths.append(upload_path)
                uploaded.write(content)
            with text_path.open("x", encoding="utf-8", newline="\n") as extracted:
                created_paths.append(text_path)
                extracted.write(text)
            cursor = db.execute(
                """INSERT INTO documents
                   (original_filename, stored_filename, extension, file_size, uploaded_by,
                    extraction_status, extracted_characters)
                   VALUES (?, ?, ?, ?, ?, 'completed', ?)""",
                (filename, stored_filename, extension, len(content), uploaded_by, len(text)),
            )
            document = dict(db.execute("SELECT * FROM documents WHERE id = ?", (cursor.lastrowid,)).fetchone())
        return document
    except (OSError, sqlite3.Error, DocumentError) as exc:
        for path in created_paths:
            path.unlink(missing_ok=True)
        if isinstance(exc, DocumentError):
            raise
        raise DocumentError("The document could not be saved. Check available disk space and database/directory permissions, then retry.") from exc


def list_documents(user_id: int) -> list[dict]:
    with database.connection() as db:
        _require_uploader(db, user_id)
        rows = db.execute("""
            SELECT documents.*, users.name AS uploader_name, users.email AS uploader_email
            FROM documents JOIN users ON users.id = documents.uploaded_by
            ORDER BY documents.uploaded_at DESC, documents.id DESC
        """).fetchall()
    return [dict(row) for row in rows]
