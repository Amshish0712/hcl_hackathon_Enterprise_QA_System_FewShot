from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from core import database
from core.auth import login, register
from core.documents import (
    MAX_FILE_SIZE, DocumentError, extract_text, ingest_document, list_documents, validate_file,
)


def pdf_bytes(text="Hello PDF", encrypted=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text:
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 30 250 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("secret")
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def docx_bytes():
    document = Document()
    document.add_paragraph("First paragraph")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Left cell"
    table.cell(0, 1).text = "Right cell"
    document.add_paragraph("Last paragraph")
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class FileValidationTests(unittest.TestCase):
    def test_supported_formats_mime_fallback_and_case_insensitive_extensions(self):
        cases = [
            ("REPORT.PDF", pdf_bytes(), "application/pdf"),
            ("report.docx", docx_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("notes.txt", b"Hello", "text/plain; charset=utf-8"),
            ("notes.md", b"# Hello", "text/markdown"),
            ("index.html", b"<p>Hello</p>", "text/html"),
            ("index.htm", b"<p>Hello</p>", "text/html"),
            ("notes.txt", b"Hello", "application/octet-stream"),
            ("notes.md", b"# Hello", None),
        ]
        for filename, content, mime in cases:
            with self.subTest(filename=filename, mime=mime):
                self.assertEqual(validate_file(filename, content, mime), Path(filename).suffix.lower())

    def test_unsupported_empty_oversized_mismatched_and_unsafe_files(self):
        cases = [
            ("program.exe", b"binary", None, "Unsupported"),
            ("empty.txt", b"", "text/plain", "empty"),
            ("large.txt", b"x" * (MAX_FILE_SIZE + 1), "text/plain", "20 MB"),
            ("notes.txt", b"hello", "application/pdf", "MIME"),
            ("fake.pdf", b"not a pdf", "application/pdf", "valid PDF"),
            ("fake.docx", b"not a zip", None, "corrupted"),
            ("../notes.txt", b"hello", None, "directory"),
            ("..\\notes.txt", b"hello", None, "directory"),
        ]
        for filename, content, mime, message in cases:
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(DocumentError, message):
                    validate_file(filename, content, mime)

    def test_exact_size_limit_is_allowed(self):
        self.assertEqual(validate_file("limit.txt", b"x" * MAX_FILE_SIZE), ".txt")


class TextExtractionTests(unittest.TestCase):
    def test_pdf(self):
        self.assertIn("Hello PDF", extract_text(pdf_bytes(), ".pdf"))

    def test_docx_paragraphs_and_tables_preserve_order(self):
        text = extract_text(docx_bytes(), ".docx")
        self.assertIn("Left cell\tRight cell", text)
        self.assertLess(text.index("First paragraph"), text.index("Left cell"))
        self.assertLess(text.index("Right cell"), text.index("Last paragraph"))

    def test_text_markdown_bom_and_encoding_fallback(self):
        for content, extension, expected in [
            ("Hello café".encode("utf-8"), ".txt", "Hello café"),
            ("# Heading\n\nBody".encode("utf-8"), ".md", "# Heading\n\nBody"),
            ("BOM text".encode("utf-8-sig"), ".txt", "BOM text"),
            ("UTF16 text".encode("utf-16"), ".txt", "UTF16 text"),
            ("Legacy café".encode("cp1252"), ".txt", "Legacy café"),
        ]:
            with self.subTest(extension=extension, expected=expected):
                self.assertEqual(extract_text(content, extension), expected)

    def test_html_removes_scripts_styles_and_decodes_entities(self):
        content = b"<html><head><style>hidden-style</style></head><body><h1>A &amp; B</h1><p>Hello</p><script>hidden-script</script></body></html>"
        for extension in (".html", ".htm"):
            text = extract_text(content, extension)
            self.assertEqual(text, "A & B\nHello")
            self.assertNotIn("hidden", text)

    def test_corrupt_binary_encrypted_and_textless_documents(self):
        for content, extension, message in [
            (b"%PDF-1.7\nbroken", ".pdf", "corrupted"),
            (b"invalid", ".docx", "corrupted"),
            (b"\x00\x01binary", ".txt", "binary"),
            (b"  \r\n\t", ".txt", "No readable text"),
            (b"<html><body></body></html>", ".html", "No readable text"),
            (b"not html", ".html", "HTML markup"),
            (pdf_bytes(text=""), ".pdf", "OCR"),
            (pdf_bytes(encrypted=True), ".pdf", "Encrypted"),
        ]:
            with self.subTest(extension=extension, message=message):
                with self.assertRaisesRegex(DocumentError, message):
                    extract_text(content, extension)


class DocumentStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        patcher = patch.object(database, "DB_PATH", Path(self.directory.name) / "app.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.initialize_database()
        self.admin = login("admin@enterprise.com", "Admin@123")

    def test_metadata_uuid_files_extracted_text_and_history(self):
        records = [ingest_document("same.txt", b"Hello documents", "text/plain", self.admin["id"]) for _ in range(2)]
        self.assertNotEqual(records[0]["stored_filename"], records[1]["stored_filename"])
        for record in records:
            UUID(Path(record["stored_filename"]).stem)
            self.assertEqual(record["original_filename"], "same.txt")
            self.assertEqual(record["file_size"], len(b"Hello documents"))
            self.assertEqual(record["uploaded_by"], self.admin["id"])
            self.assertTrue(record["uploaded_at"])
            self.assertEqual(record["extraction_status"], "completed")
            self.assertEqual(record["extracted_characters"], len("Hello documents"))
            root = database.DB_PATH.parent
            self.assertEqual((root / "uploads" / record["stored_filename"]).read_bytes(), b"Hello documents")
            self.assertEqual((root / "extracted_text" / (Path(record["stored_filename"]).stem + ".txt")).read_text(encoding="utf-8"), "Hello documents")
        history = list_documents(self.admin["id"])
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["uploader_email"], "admin@enterprise.com")

    def test_editor_allowed_viewer_and_deleted_user_rejected(self):
        editor = register("Test Editor", "editor@example.com", "Strong@123", "editor")
        viewer = register("Test Viewer", "viewer@example.com", "Strong@123", "viewer")
        ingest_document("editor.txt", b"Editor document", "text/plain", editor["id"])
        self.assertEqual(len(list_documents(editor["id"])), 1)
        for user_id in (viewer["id"], 999999):
            with self.assertRaises(DocumentError):
                ingest_document("denied.txt", b"Denied", "text/plain", user_id)
            with self.assertRaises(DocumentError):
                list_documents(user_id)

    def test_rejected_documents_leave_no_files_or_metadata(self):
        for filename, content in (("bad.pdf", b"%PDF-1.7\nbroken"), ("empty.txt", b""), ("blank.txt", b"   ")):
            with self.assertRaises(DocumentError):
                ingest_document(filename, content, None, self.admin["id"])
        self.assertEqual(list_documents(self.admin["id"]), [])
        self.assertFalse((database.DB_PATH.parent / "uploads").exists())

    def test_text_write_failure_rolls_back_uploaded_file(self):
        original_open = Path.open

        def failing_open(path, *args, **kwargs):
            if path.parent.name == "extracted_text":
                raise OSError("Simulated full disk")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", failing_open):
            with self.assertRaisesRegex(DocumentError, "could not be saved"):
                ingest_document("test.txt", b"Hello", "text/plain", self.admin["id"])
        self.assertEqual(list((database.DB_PATH.parent / "uploads").iterdir()), [])
        self.assertEqual(list_documents(self.admin["id"]), [])

    def test_database_failure_rolls_back_files(self):
        with database.connection() as db:
            db.execute("""CREATE TRIGGER fail_documents BEFORE INSERT ON documents
                          BEGIN SELECT RAISE(ABORT, 'Simulated database failure'); END""")
        with self.assertRaisesRegex(DocumentError, "could not be saved"):
            ingest_document("test.txt", b"Hello", "text/plain", self.admin["id"])
        self.assertEqual(list((database.DB_PATH.parent / "uploads").iterdir()), [])
        self.assertEqual(list((database.DB_PATH.parent / "extracted_text").iterdir()), [])
        self.assertEqual(list_documents(self.admin["id"]), [])

    def test_additive_migration_preserves_phase_one_account(self):
        with database.connection() as db:
            original_hash = db.execute("SELECT password_hash FROM users WHERE id = ?", (self.admin["id"],)).fetchone()[0]
            db.execute("DROP TABLE documents")
        database.initialize_database()
        database.initialize_database()
        self.assertEqual(login("admin@enterprise.com", "Admin@123"), self.admin)
        with database.connection() as db:
            self.assertEqual(db.execute("SELECT password_hash FROM users WHERE id = ?", (self.admin["id"],)).fetchone()[0], original_hash)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
