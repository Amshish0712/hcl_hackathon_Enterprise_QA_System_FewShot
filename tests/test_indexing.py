import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import faiss
import numpy as np

from core import database
from core.auth import login, register
from core.chunking import split_document
from core.document_management import ManagementError, delete_document
from core.documents import ingest_document
from core.embeddings import EMBEDDING_DIMENSION, MODEL_NAME, embed_chunks
from core.index_errors import IndexingError, ModelLoadingError
from core.indexing import index_documents, index_status, indexable_documents, rebuild_accessible_index
from core.vector_store import vector_transaction


def fake_embeddings(chunks, progress=None):
    result = []
    for chunk in chunks:
        digest = hashlib.sha256(chunk["chunk_text"].encode()).digest()
        vector = np.resize(np.frombuffer(digest, dtype="uint8"), EMBEDDING_DIMENSION).astype("float32") + 1
        result.append(vector / np.linalg.norm(vector))
    if progress:
        progress(1, "Embedded test chunks")
    return np.asarray(result, dtype="float32")


class ChunkingTests(unittest.TestCase):
    def test_size_overlap_coverage_and_no_empty_chunks(self):
        text = "".join(chr(0x400 + position % 200) for position in range(2200))
        chunks = split_document(text, "report.txt")
        self.assertTrue(all(0 < item["chunk_length"] <= 800 for item in chunks))
        self.assertEqual([item["chunk_index"] for item in chunks], list(range(len(chunks))))
        rebuilt = chunks[0]["chunk_text"]
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous["chunk_text"][-150:], current["chunk_text"][:150])
            rebuilt += current["chunk_text"][150:]
        self.assertEqual(rebuilt, text)
        self.assertEqual(split_document(" \n\t"), [])
        self.assertTrue(all(item["document_name"] == "report.txt" for item in chunks))

    def test_heading_boundaries_context_and_setext(self):
        chunks = split_document("# Introduction\n" + "First section. " * 90 + "\n## Details\nSecond section.\n\nConclusion\n==========\nFinal text.", "guide.md")
        self.assertTrue(any(item["section_title"] == "Introduction" for item in chunks))
        self.assertTrue(any(item["section_title"] == "Details" for item in chunks))
        self.assertEqual(chunks[-1]["section_title"], "Conclusion")
        self.assertFalse(any("First section" in item["chunk_text"] and "Second section" in item["chunk_text"] for item in chunks))

    def test_overlap_preserves_whitespace_at_word_boundaries(self):
        chunks = split_document("Paragraph words and spacing. " * 100)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous["chunk_text"][-150:], current["chunk_text"][:150])

    def test_explicit_page_markers_formfeeds_and_unavailable_page_metadata(self):
        chunks = split_document("[Page 3]\n# Topic\nFirst page\n--- Page 4 ---\nNext page", "report.pdf")
        self.assertEqual([item["page_number"] for item in chunks], [3, 4])
        self.assertEqual([item["section_title"] for item in chunks], ["Topic", "Topic"])
        self.assertEqual([item["page_number"] for item in split_document("First\fSecond", "report.pdf")], [1, 2])
        self.assertIsNone(split_document("No page marker", "report.pdf")[0]["page_number"])

    def test_invalid_configuration_and_fenced_heading(self):
        with self.assertRaises(ValueError):
            split_document("text", chunk_size=150, overlap=150)
        chunks = split_document("# Actual\n```python\n# code comment\n```\nEnd.")
        self.assertEqual({item["section_title"] for item in chunks}, {"Actual"})


class EmbeddingTests(unittest.TestCase):
    def test_cpu_batching_context_and_normalized_float32_vectors(self):
        chunks = split_document("[Page 2]\n# Topic\n" + "Embedding content. " * 1000, "report.md")
        with patch("core.embeddings.load_model") as loader:
            loader.return_value.encode.side_effect = lambda inputs, **kwargs: np.ones((len(inputs), EMBEDDING_DIMENSION), dtype="float32")
            result = embed_chunks(chunks)
        self.assertEqual(result.shape, (len(chunks), EMBEDDING_DIMENSION))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
        self.assertGreater(loader.return_value.encode.call_count, 1)
        for call in loader.return_value.encode.call_args_list:
            self.assertEqual(call.kwargs["device"], "cpu")
            self.assertTrue(call.kwargs["normalize_embeddings"])
            self.assertIn("Document: report.md\nSection: Topic\nPage: 2", call.args[0][0])

    def test_invalid_embedding_output_and_empty_chunks_are_clear_errors(self):
        with patch("core.embeddings.load_model") as loader:
            with self.assertRaises(IndexingError):
                embed_chunks([])
            loader.assert_not_called()
            loader.return_value.encode.return_value = np.zeros((1, 384), dtype="float32")
            with self.assertRaisesRegex(IndexingError, "Embedding generation failed"):
                embed_chunks(split_document("Text", "report.txt"))


class IndexingTests(unittest.TestCase):
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
        self.admin_doc = ingest_document("admin.md", ("# Overview\n" + "Enterprise content. " * 100).encode(), "text/markdown", self.admin["id"])
        self.editor_doc = ingest_document("editor.txt", b"Editor content. " * 100, "text/plain", self.editor["id"])
        embedder = patch("core.indexing.embed_chunks", side_effect=fake_embeddings)
        self.embedder = embedder.start()
        self.addCleanup(embedder.stop)

    @property
    def index_path(self):
        return database.DB_PATH.parent / "vector_store" / "index.faiss"

    def assert_consistent(self):
        index = faiss.read_index(str(self.index_path))
        with database.connection() as db:
            mappings = db.execute("SELECT * FROM vector_mapping ORDER BY position").fetchall()
            chunks = dict(db.execute("SELECT id, document_id FROM chunks").fetchall())
        self.assertEqual(index.ntotal, len(chunks))
        self.assertEqual(faiss.vector_to_array(index.id_map).tolist(), [item["chunk_id"] for item in mappings])
        self.assertEqual([item["position"] for item in mappings], list(range(index.ntotal)))
        self.assertEqual(set(chunks), {item["chunk_id"] for item in mappings})
        for item in mappings:
            self.assertEqual(item["document_id"], chunks[item["chunk_id"]])
        return index

    def test_safe_migration_and_document_defaults(self):
        database.initialize_database()
        database.initialize_database()
        with database.connection() as db:
            columns = {row["name"] for row in db.execute("PRAGMA table_info(documents)")}
            row = db.execute("SELECT * FROM documents WHERE id = ?", (self.admin_doc["id"],)).fetchone()
        self.assertTrue({"indexing_status", "indexed_at", "chunk_count", "embedding_model"}.issubset(columns))
        self.assertEqual(row["indexing_status"], "not_indexed")
        self.assertEqual(row["chunk_count"], 0)
        self.assertIsNone(row["indexed_at"])
        self.assertIsNotNone(login("admin@enterprise.com", "Admin@123"))

    def test_legacy_database_migration_preserves_existing_users_and_documents(self):
        legacy_path = database.DB_PATH.parent / "legacy.db"
        with database.connection() as db:
            admin = db.execute("SELECT * FROM users WHERE id = ?", (self.admin["id"],)).fetchone()
        with sqlite3.connect(legacy_path) as db:
            db.executescript("""
                CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE,
                    password_hash TEXT NOT NULL, role TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE documents(id INTEGER PRIMARY KEY, original_filename TEXT, stored_filename TEXT UNIQUE,
                    extension TEXT, file_size INTEGER, uploaded_by INTEGER REFERENCES users(id),
                    uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP, extraction_status TEXT, extracted_characters INTEGER);
            """)
            db.execute("INSERT INTO users(id,name,email,password_hash,role) VALUES (?,?,?,?,?)", tuple(admin[key] for key in ("id", "name", "email", "password_hash", "role")))
            db.execute("""INSERT INTO documents(id,original_filename,stored_filename,extension,file_size,uploaded_by,extraction_status,extracted_characters)
                          VALUES (17,'legacy.txt','legacy.txt','.txt',10,?,'completed',10)""", (admin["id"],))
        db.close()
        with patch.object(database, "DB_PATH", legacy_path):
            database.initialize_database()
            database.initialize_database()
            self.assertEqual(login("admin@enterprise.com", "Admin@123")["id"], admin["id"])
            with database.connection() as db:
                row = db.execute("SELECT * FROM documents WHERE id = 17").fetchone()
            self.assertEqual(row["original_filename"], "legacy.txt")
            self.assertEqual(row["indexing_status"], "not_indexed")
            self.assertEqual(row["chunk_count"], 0)

    def test_persistence_metadata_and_dashboard_scope(self):
        progress = []
        result = index_documents(self.admin["id"], [self.admin_doc["id"], self.editor_doc["id"]], lambda fraction, text: progress.append(fraction))
        self.assertEqual(result["documents_indexed"], 2)
        self.assertEqual(result["embedding_model"], MODEL_NAME)
        self.assertEqual(progress[-1], 1)
        self.assertEqual(progress, sorted(progress))
        index = self.assert_consistent()
        self.assertEqual(index.d, EMBEDDING_DIMENSION)
        with database.connection() as db:
            chunk = db.execute("SELECT * FROM chunks WHERE document_id = ? LIMIT 1", (self.admin_doc["id"],)).fetchone()
            document = db.execute("SELECT * FROM documents WHERE id = ?", (self.admin_doc["id"],)).fetchone()
        self.assertEqual(chunk["section_title"], "Overview")
        self.assertEqual(chunk["chunk_length"], len(chunk["chunk_text"]))
        self.assertEqual(document["indexing_status"], "indexed")
        self.assertEqual(document["embedding_model"], MODEL_NAME)
        self.assertTrue(document["indexed_at"])
        self.assertEqual(index_status(self.admin["id"])["indexed_documents"], 2)
        self.assertEqual(index_status(self.viewer["id"])["total_chunks"], index.ntotal)
        self.assertEqual(index_status(self.editor["id"])["indexed_documents"], 1)
        with vector_transaction() as transaction:
            self.assertEqual(transaction.load().ntotal, index.ntotal)

    def test_repeat_indexing_replaces_old_chunks_and_vectors_without_duplicates(self):
        first = index_documents(self.editor["id"], [self.editor_doc["id"], self.editor_doc["id"]])
        with database.connection() as db:
            old_ids = {row[0] for row in db.execute("SELECT id FROM chunks")}
        second = index_documents(self.editor["id"], [self.editor_doc["id"]])
        self.assertEqual(first["documents_indexed"], 1)
        self.assertEqual(first["chunks_created"], second["chunks_created"])
        self.assertEqual(self.assert_consistent().ntotal, second["chunks_created"])
        with database.connection() as db:
            new_ids = {row[0] for row in db.execute("SELECT id FROM chunks")}
        self.assertFalse(old_ids & new_ids)

    def test_access_checks_reject_viewers_other_editors_and_mixed_ids_before_work(self):
        self.assertEqual([doc["id"] for doc in indexable_documents(self.editor["id"])], [self.editor_doc["id"]])
        for user_id, ids in ((self.viewer["id"], [self.admin_doc["id"]]),
                             (self.editor["id"], [self.admin_doc["id"]]),
                             (self.editor["id"], [self.editor_doc["id"], self.admin_doc["id"]]),
                             (999999, [self.admin_doc["id"]])):
            with self.assertRaises(IndexingError):
                index_documents(user_id, ids)
        with self.assertRaises(IndexingError):
            indexable_documents(self.viewer["id"])
        with self.assertRaises(IndexingError):
            rebuild_accessible_index(self.viewer["id"])
        self.embedder.assert_not_called()
        self.assertFalse(self.index_path.exists())

    def test_only_successful_extractions_and_empty_selection_are_allowed(self):
        with database.connection() as db:
            db.execute("UPDATE documents SET extraction_status = 'failed' WHERE id = ?", (self.editor_doc["id"],))
        self.assertEqual(indexable_documents(self.editor["id"]), [])
        with self.assertRaises(IndexingError):
            index_documents(self.editor["id"], [self.editor_doc["id"]])
        with self.assertRaises(IndexingError):
            index_documents(self.admin["id"], [])
        with self.assertRaises(IndexingError):
            rebuild_accessible_index(self.editor["id"])

    def test_editor_rebuild_preserves_other_users_vectors_and_chunk_ids(self):
        index_documents(self.admin["id"], [self.admin_doc["id"], self.editor_doc["id"]])
        with database.connection() as db:
            protected_ids = [row[0] for row in db.execute("SELECT id FROM chunks WHERE document_id = ?", (self.admin_doc["id"],))]
        before = self.assert_consistent()
        protected = {chunk_id: before.reconstruct(chunk_id).copy() for chunk_id in protected_ids}
        self.embedder.reset_mock()
        result = rebuild_accessible_index(self.editor["id"])
        self.assertEqual(result["documents_indexed"], 1)
        self.assertEqual(self.embedder.call_count, 1)
        after = self.assert_consistent()
        for chunk_id in protected_ids:
            np.testing.assert_array_equal(after.reconstruct(chunk_id), protected[chunk_id])

    def test_model_failure_preserves_previous_index_and_allows_retry(self):
        index_documents(self.editor["id"], [self.editor_doc["id"]])
        before = self.index_path.read_bytes()
        self.embedder.side_effect = IndexingError("Model download unavailable")
        result = index_documents(self.editor["id"], [self.editor_doc["id"]])
        self.assertEqual(result["results"][0]["status"], "failed")
        self.assertIn("Model download", result["results"][0]["message"])
        self.assertEqual(self.index_path.read_bytes(), before)
        self.assert_consistent()
        self.embedder.side_effect = fake_embeddings
        self.assertEqual(index_documents(self.editor["id"], [self.editor_doc["id"]])["documents_indexed"], 1)

    def test_unavailable_model_is_loaded_only_once_per_batch(self):
        self.embedder.side_effect = ModelLoadingError("Model download unavailable")
        result = index_documents(self.admin["id"], [self.admin_doc["id"], self.editor_doc["id"]])
        self.assertEqual(self.embedder.call_count, 1)
        self.assertEqual([item["status"] for item in result["results"]], ["failed", "failed"])
        self.assertFalse(self.index_path.exists())

    def test_empty_or_missing_text_fails_safely_and_batch_continues(self):
        path = database.DB_PATH.parent / "extracted_text" / (Path(self.admin_doc["stored_filename"]).stem + ".txt")
        path.write_text(" \n", encoding="utf-8")
        result = index_documents(self.admin["id"], [self.admin_doc["id"], self.editor_doc["id"]])
        self.assertEqual([item["status"] for item in result["results"]], ["failed", "indexed"])
        self.assertIn("empty", result["results"][0]["message"])
        path.unlink()
        result = index_documents(self.admin["id"], [self.admin_doc["id"]])
        self.assertIn("missing", result["results"][0]["message"])
        self.assert_consistent()

    def test_lost_ownership_during_embedding_prevents_commit(self):
        def revoke(chunks, progress=None):
            with database.connection() as db:
                db.execute("UPDATE users SET role = 'viewer' WHERE id = ?", (self.editor["id"],))
            return fake_embeddings(chunks, progress)
        self.embedder.side_effect = revoke
        result = index_documents(self.editor["id"], [self.editor_doc["id"]])
        self.assertEqual(result["documents_indexed"], 0)
        with database.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)

    def test_failed_publication_restores_faiss_file_and_sqlite(self):
        index_documents(self.admin["id"], [self.admin_doc["id"]])
        previous_bytes = self.index_path.read_bytes()
        with database.connection() as db:
            db.execute("""CREATE TRIGGER fail_index_commit BEFORE UPDATE ON index_state
                          BEGIN SELECT RAISE(ABORT, 'Simulated publication failure'); END""")
        result = index_documents(self.admin["id"], [self.editor_doc["id"]])
        self.assertEqual(result["documents_indexed"], 0)
        self.assertEqual(self.index_path.read_bytes(), previous_bytes)
        self.assert_consistent()
        self.assertEqual(index_status(self.admin["id"])["indexed_documents"], 1)

    def test_recovery_after_file_publication_before_database_commit(self):
        index_documents(self.admin["id"], [self.admin_doc["id"]])
        valid = self.index_path.read_bytes()
        self.index_path.with_name("index.faiss.previous").write_bytes(valid)
        self.index_path.write_bytes(b"Uncommitted replacement")
        with vector_transaction() as transaction:
            transaction.load()
        self.assertEqual(self.index_path.read_bytes(), valid)
        self.assert_consistent()

    def test_deleting_indexed_document_removes_chunks_vectors_and_files(self):
        index_documents(self.admin["id"], [self.admin_doc["id"], self.editor_doc["id"]])
        delete_document(self.editor["id"], self.editor_doc["id"], confirmed=True)
        index = self.assert_consistent()
        with database.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (self.editor_doc["id"],)).fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM vector_mapping WHERE document_id = ?", (self.editor_doc["id"],)).fetchone()[0], 0)
        self.assertEqual(index.ntotal, index_status(self.admin["id"])["total_chunks"])
        delete_document(self.admin["id"], self.admin_doc["id"], confirmed=True)
        self.assertEqual(self.assert_consistent().ntotal, 0)

    def test_indexed_deletion_publication_failure_restores_files_chunks_and_vectors(self):
        index_documents(self.admin["id"], [self.admin_doc["id"]])
        previous_bytes = self.index_path.read_bytes()
        with database.connection() as db:
            db.execute("""CREATE TRIGGER fail_index_deletion BEFORE UPDATE ON index_state
                          BEGIN SELECT RAISE(ABORT, 'Simulated publication failure'); END""")
        with self.assertRaises(ManagementError):
            delete_document(self.admin["id"], self.admin_doc["id"], confirmed=True)
        self.assertEqual(self.index_path.read_bytes(), previous_bytes)
        self.assert_consistent()
        self.assertEqual(index_status(self.admin["id"])["indexed_documents"], 1)
        self.assertTrue((database.DB_PATH.parent / "uploads" / self.admin_doc["stored_filename"]).exists())
        self.assertTrue((database.DB_PATH.parent / "extracted_text" / (Path(self.admin_doc["stored_filename"]).stem + ".txt")).exists())


if __name__ == "__main__":
    unittest.main()
