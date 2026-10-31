"""Serialized SQLite/FAISS writes, stable chunk IDs, and recoverable publication."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import shutil
from uuid import uuid4

from filelock import FileLock, Timeout
import numpy as np

from core import database
from core.embeddings import EMBEDDING_DIMENSION, MODEL_NAME
from core.index_errors import IndexingError


def _faiss():
    try:
        import faiss
        return faiss
    except (ImportError, OSError) as exc:
        raise IndexingError("FAISS could not be loaded. Install requirements.txt in the active Python environment and restart the app.") from exc


def _path(name: str) -> Path:
    path = database.DB_PATH.parent.resolve() / "vector_store" / name
    if path.resolve() != path or path.is_symlink():
        raise IndexingError("The vector storage path is unsafe. Check data/vector_store.")
    return path


def _checksum(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VectorTransaction:
    def __init__(self, db):
        self.db = db
        self.index = None
        self.changed = False
        self.published = False
        self.had_original = False
        self.temporary = _path(uuid4().hex + ".tmp")

    def load(self):
        if self.index is not None:
            return self.index
        faiss = _faiss()
        state = self.db.execute("SELECT * FROM index_state WHERE id = 1").fetchone()
        mappings = self.db.execute("SELECT * FROM vector_mapping ORDER BY position").fetchall()
        if state is None:
            if mappings or self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]:
                raise IndexingError("Vector metadata has no matching index state. Ask an administrator to restore the vector-store backup.")
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(EMBEDDING_DIMENSION))
            return self.index
        if state["embedding_model"] != MODEL_NAME:
            raise IndexingError("The stored index uses a different embedding model.")
        current, previous = _path("index.faiss"), _path("index.faiss.previous")
        if _checksum(current) != state["checksum"]:
            if _checksum(previous) != state["checksum"]:
                raise IndexingError("The FAISS index is missing or damaged and does not match SQLite. Restore data/vector_store and data/app.db from the same backup.")
            # A process may have stopped after the file replacement but before SQLite committed.
            shutil.copyfile(previous, self.temporary)
            self.temporary.replace(current)
        try:
            self.index = faiss.read_index(str(current))
            actual_ids = faiss.vector_to_array(self.index.id_map).tolist()
            if self.index.d != EMBEDDING_DIMENSION or actual_ids != [row["chunk_id"] for row in mappings]:
                raise ValueError("Vector-to-chunk mapping mismatch")
            if [row["position"] for row in mappings] != list(range(self.index.ntotal)):
                raise ValueError("Non-contiguous positions")
            chunk_documents = dict(self.db.execute("SELECT id, document_id FROM chunks").fetchall())
            if set(chunk_documents) != set(actual_ids) or any(chunk_documents[row["chunk_id"]] != row["document_id"] for row in mappings):
                raise ValueError("Document mapping mismatch")
        except Exception as exc:
            raise IndexingError("The stored FAISS index and chunk metadata are inconsistent. Restore a matching backup before indexing.") from exc
        return self.index

    def reset(self) -> None:
        """Start an empty index without reading the old one; chunks are replaced on publish."""
        faiss = _faiss()
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(EMBEDDING_DIMENSION))
        self.db.execute("DELETE FROM chunks")
        self.changed = True

    def remove_documents(self, document_ids: list[int]) -> None:
        if not document_ids:
            return
        placeholders = ",".join("?" for _ in document_ids)
        ids = [row[0] for row in self.db.execute(
            f"SELECT id FROM chunks WHERE document_id IN ({placeholders})", document_ids,
        )]
        if not ids:
            return
        self.load().remove_ids(np.asarray(ids, dtype="int64"))
        self.db.execute(f"DELETE FROM chunks WHERE document_id IN ({placeholders})", document_ids)
        self.changed = True

    def add(self, vectors, chunk_ids):
        self.load().add_with_ids(np.ascontiguousarray(vectors, dtype="float32"), np.asarray(chunk_ids, dtype="int64"))
        self.changed = True

    def publish(self):
        if not self.changed:
            return
        faiss = _faiss()
        faiss.write_index(self.index, str(self.temporary))
        # Flush the temporary file before publishing its checksum in SQLite.
        import os
        with self.temporary.open("rb+") as source:
            os.fsync(source.fileno())
        digest = _checksum(self.temporary)
        current, previous = _path("index.faiss"), _path("index.faiss.previous")
        self.had_original = current.exists()
        if self.had_original:
            shutil.copyfile(current, previous)
        self.temporary.replace(current)
        self.published = True
        ids = faiss.vector_to_array(self.index.id_map).tolist()
        chunk_documents = dict(self.db.execute("SELECT id, document_id FROM chunks").fetchall())
        self.db.execute("DELETE FROM vector_mapping")
        self.db.executemany(
            "INSERT INTO vector_mapping(position, chunk_id, document_id) VALUES (?, ?, ?)",
            [(position, chunk_id, chunk_documents[chunk_id]) for position, chunk_id in enumerate(ids)],
        )
        self.db.execute(
            """INSERT INTO index_state(id, checksum, embedding_model) VALUES (1, ?, ?)
               ON CONFLICT(id) DO UPDATE SET checksum=excluded.checksum, embedding_model=excluded.embedding_model""",
            (digest, MODEL_NAME),
        )

    def restore(self):
        if self.published:
            current = _path("index.faiss")
            if self.had_original:
                shutil.copyfile(_path("index.faiss.previous"), current)
            else:
                current.unlink(missing_ok=True)


@contextmanager
def vector_transaction():
    """All vector writers, including document deletion, share one process-safe lock."""
    transaction = None
    try:
        lock_path = _path("index.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_path), timeout=30):
            try:
                with database.connection() as db:
                    db.execute("BEGIN IMMEDIATE")
                    transaction = VectorTransaction(db)
                    yield transaction
                    transaction.publish()
            except Exception as exc:
                if transaction is not None:
                    transaction.restore()
                if isinstance(exc, RuntimeError):
                    raise IndexingError("FAISS could not update the local vector store. The previous index was retained; check storage permissions and retry.") from exc
                raise
            finally:
                if transaction is not None:
                    transaction.temporary.unlink(missing_ok=True)
    except Timeout as exc:
        raise IndexingError("Another session is updating the index. Wait for it to finish and retry.") from exc
