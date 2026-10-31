"""Authorized chunking and embedding workflows; no retrieval or answer generation."""

import hashlib
import sqlite3

from core import database
from core.chunking import split_document
from core.document_management import ManagementError, _actor, _document_paths
from core.embeddings import MODEL_NAME, embed_chunks
from core.index_errors import IndexingError, ModelLoadingError
from core.vector_store import vector_transaction


def _indexer(db, user_id):
    try:
        actor = _actor(db, user_id)
    except ManagementError as exc:
        raise IndexingError(str(exc)) from exc
    if actor["role"] not in ("admin", "editor"):
        raise IndexingError("Only administrators and editors can index documents.")
    return actor


def _indexable(db, user_id, document_id):
    actor = _indexer(db, user_id)
    if type(document_id) is not int or document_id < 1:
        raise IndexingError("Document not found or not available for indexing.")
    row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if (row is None or row["extraction_status"] != "completed"
            or (actor["role"] == "editor" and row["uploaded_by"] != actor["id"])):
        raise IndexingError("Document not found, not successfully extracted, or not permitted for this account.")
    return dict(row)


def indexable_documents(user_id: int) -> list[dict]:
    with database.connection() as db:
        actor = _indexer(db, user_id)
        query = "SELECT * FROM documents WHERE extraction_status = 'completed'"
        parameters = []
        if actor["role"] == "editor":
            query += " AND uploaded_by = ?"
            parameters.append(actor["id"])
        rows = db.execute(query + " ORDER BY uploaded_at DESC, id DESC", parameters).fetchall()
    return [dict(row) for row in rows]


def index_status(user_id: int) -> dict:
    with database.connection() as db:
        try:
            actor = _actor(db, user_id)
        except ManagementError as exc:
            raise IndexingError(str(exc)) from exc
        condition, parameters = "", []
        if actor["role"] == "editor":
            condition = "WHERE uploaded_by = ?"
            parameters.append(actor["id"])
        row = db.execute(
            f"SELECT COUNT(CASE WHEN chunk_count > 0 THEN 1 END), COALESCE(SUM(chunk_count), 0) FROM documents {condition}", parameters,
        ).fetchone()
    return {"indexed_documents": row[0], "total_chunks": row[1], "embedding_model": MODEL_NAME}


def _read_text(document):
    try:
        _, path = _document_paths(document)
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ManagementError) as exc:
        raise IndexingError("The extracted text is missing, unreadable, or has an unsafe path. Check the document's extracted text file.") from exc
    if not text.strip():
        raise IndexingError("The extracted text is empty. No chunks or embeddings were created.")
    return text


def _mark_failed(user_id, document_id):
    with database.connection() as db:
        _indexable(db, user_id, document_id)
        # A failed refresh must not discard an existing usable index.
        db.execute("UPDATE documents SET indexing_status = CASE WHEN chunk_count > 0 THEN 'indexed' ELSE 'failed' END WHERE id = ?", (document_id,))


def index_documents(user_id: int, document_ids: list[int], progress=None) -> dict:
    if not document_ids:
        raise IndexingError("Select at least one successfully extracted document to index.")
    # Validate the entire requested scope before embedding or changing any document.
    with database.connection() as db:
        _indexer(db, user_id)
        documents = []
        seen = set()
        for document_id in document_ids:
            document = _indexable(db, user_id, document_id)
            if document_id not in seen:
                documents.append(document)
                seen.add(document_id)
    results = []
    model_failure = None
    for position, document in enumerate(documents):
        def report(fraction, message):
            if progress:
                progress((position + fraction) / len(documents), f"{document['original_filename']}: {message}")
        try:
            if model_failure is not None:
                raise IndexingError(str(model_failure))
            report(0, "Reading extracted text and splitting into chunks…")
            text = _read_text(document)
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunks = split_document(text, document["original_filename"])
            if not chunks:
                raise IndexingError("The document has no non-empty text chunks.")
            vectors = embed_chunks(chunks, lambda fraction, message: report(0.05 + fraction * 0.85, message))
            report(0.92, "Saving chunks and FAISS vectors…")
            with vector_transaction() as transaction:
                current = _indexable(transaction.db, user_id, document["id"])
                if hashlib.sha256(_read_text(current).encode("utf-8")).hexdigest() != fingerprint:
                    raise IndexingError("The extracted text changed during indexing. Retry to index the current content.")
                # Load/validate the old mapping before inserting new chunks.
                transaction.load()
                transaction.remove_documents([document["id"]])
                chunk_ids = []
                for chunk in chunks:
                    cursor = transaction.db.execute(
                        """INSERT INTO chunks(document_id, chunk_index, chunk_text, chunk_length, page_number, section_title)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (document["id"], chunk["chunk_index"], chunk["chunk_text"], chunk["chunk_length"], chunk["page_number"], chunk["section_title"]),
                    )
                    chunk_ids.append(cursor.lastrowid)
                transaction.add(vectors, chunk_ids)
                transaction.db.execute(
                    """UPDATE documents SET indexing_status = 'indexed', indexed_at = CURRENT_TIMESTAMP,
                       chunk_count = ?, embedding_model = ? WHERE id = ?""",
                    (len(chunks), MODEL_NAME, document["id"]),
                )
            results.append({"document_id": document["id"], "filename": document["original_filename"], "status": "indexed", "chunks": len(chunks), "message": "Index created successfully."})
        except (IndexingError, OSError, sqlite3.Error) as exc:
            if isinstance(exc, ModelLoadingError):
                model_failure = exc
            try:
                _mark_failed(user_id, document["id"])
            except (IndexingError, sqlite3.Error):
                pass  # The account or document may have been removed while embedding.
            message = str(exc) if isinstance(exc, IndexingError) else "The index could not be saved. Previous chunks and vectors were retained; check storage permissions and retry."
            results.append({"document_id": document["id"], "filename": document["original_filename"], "status": "failed", "chunks": 0, "message": message})
        report(1, results[-1]["message"])
    return {
        "documents_indexed": sum(item["status"] == "indexed" for item in results),
        "chunks_created": sum(item["chunks"] for item in results),
        "embedding_model": MODEL_NAME, "results": results,
    }


def rebuild_accessible_index(user_id: int, progress=None) -> dict:
    documents = indexable_documents(user_id)
    if not documents:
        raise IndexingError("There are no successfully extracted documents available for this account to index.")
    # Refresh only the caller's partition; never delete or re-embed other users' vectors.
    return index_documents(user_id, [document["id"] for document in documents], progress)


def _rebuild_all(progress=None) -> dict:
    """Re-chunk and re-embed every completed document into a brand-new FAISS index.

    Documents and files are never deleted. Every document is embedded before the
    write lock is taken; any failure aborts the rebuild and keeps the current index.
    """
    with database.connection() as db:
        documents = [dict(row) for row in db.execute(
            "SELECT * FROM documents WHERE extraction_status = 'completed' ORDER BY id")]
    if not documents:
        raise IndexingError("There are no successfully extracted documents to index.")
    prepared = []
    for position, document in enumerate(documents):
        def report(fraction, message):
            if progress:
                progress((position + fraction) / len(documents) * 0.9, f"{document['original_filename']}: {message}")
        try:
            text = _read_text(document)
            chunks = split_document(text, document["original_filename"])
            if not chunks:
                raise IndexingError("The document has no non-empty text chunks.")
            vectors = embed_chunks(chunks, report)
        except IndexingError as exc:
            raise IndexingError(f"Clean rebuild stopped at {document['original_filename']} (#{document['id']}): {exc} The existing index was kept.") from exc
        prepared.append((document, hashlib.sha256(text.encode("utf-8")).hexdigest(), chunks, vectors))
    if progress:
        progress(0.95, "Writing a fresh FAISS index and chunk mapping…")
    with vector_transaction() as transaction:
        transaction.reset()
        for document, fingerprint, chunks, vectors in prepared:
            if hashlib.sha256(_read_text(document).encode("utf-8")).hexdigest() != fingerprint:
                raise IndexingError("Extracted text changed during the rebuild. Retry; the existing index was kept.")
            chunk_ids = []
            for chunk in chunks:
                cursor = transaction.db.execute(
                    """INSERT INTO chunks(document_id, chunk_index, chunk_text, chunk_length, page_number, section_title)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (document["id"], chunk["chunk_index"], chunk["chunk_text"], chunk["chunk_length"], chunk["page_number"], chunk["section_title"]),
                )
                chunk_ids.append(cursor.lastrowid)
            transaction.add(vectors, chunk_ids)
            transaction.db.execute(
                """UPDATE documents SET indexing_status = 'indexed', indexed_at = CURRENT_TIMESTAMP,
                   chunk_count = ?, embedding_model = ? WHERE id = ?""",
                (len(chunks), MODEL_NAME, document["id"]),
            )
        transaction.db.execute(
            """UPDATE documents SET indexing_status = 'not_indexed', chunk_count = 0
               WHERE extraction_status != 'completed'""")
    return {
        "documents_indexed": len(prepared), "chunks_created": sum(len(item[2]) for item in prepared),
        "embedding_model": MODEL_NAME,
        "results": [{"document_id": document["id"], "filename": document["original_filename"], "status": "indexed",
                     "chunks": len(chunks), "message": "Index created successfully."}
                    for document, _, chunks, _ in prepared],
    }


def rebuild_clean_index(user_id: int, progress=None) -> dict:
    """Admin-only: replace the whole shared index with a clean, consistent rebuild."""
    with database.connection() as db:
        if _indexer(db, user_id)["role"] != "admin":
            raise IndexingError("Only administrators can rebuild the entire workspace index.")
    return _rebuild_all(progress)


def index_health() -> dict:
    """Report FAISS/SQLite consistency without changing anything."""
    from core.vector_store import _faiss
    report = {"consistent": True, "problems": []}
    try:
        with vector_transaction() as transaction:
            db = transaction.db
            index = transaction.load()
            ids = _faiss().vector_to_array(index.id_map).tolist() if index.ntotal else []
            report.update(vectors=index.ntotal, chunks=db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                          mappings=db.execute("SELECT COUNT(*) FROM vector_mapping").fetchone()[0])
            if len(ids) != len(set(ids)):
                report["problems"].append("Duplicate vector IDs in FAISS.")
            for row in db.execute("""
                    SELECT d.id, d.original_filename, d.chunk_count, d.indexing_status, COUNT(c.id) AS actual
                    FROM documents d LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.id"""):
                if row["actual"] != row["chunk_count"]:
                    report["problems"].append(f"Document #{row['id']} records {row['chunk_count']} chunks but has {row['actual']}.")
                if row["indexing_status"] == "indexed" and not row["actual"]:
                    report["problems"].append(f"Document #{row['id']} is marked indexed but has no chunks.")
    except IndexingError as exc:
        report["problems"].append(str(exc))
    report["consistent"] = not report["problems"]
    return report
