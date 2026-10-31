"""Lazy, cached CPU sentence embeddings. Importing this module downloads nothing."""

from functools import lru_cache
import os
from pathlib import Path
from threading import RLock

import numpy as np

from core import database
from core.index_errors import IndexingError, ModelLoadingError


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 16
MODEL_LOCK = RLock()


@lru_cache(maxsize=2)
def load_model(cache_directory: str):
    try:
        # Standard HTTPS avoids requiring the optional native Xet transfer layer.
        # Honor an explicit deployment setting if the operator enables it.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from sentence_transformers import SentenceTransformer
        Path(cache_directory).mkdir(parents=True, exist_ok=True)
        return SentenceTransformer(
            MODEL_NAME, device="cpu", cache_folder=cache_directory,
            trust_remote_code=False, model_kwargs={"use_safetensors": True},
        )
    except Exception as exc:
        raise ModelLoadingError(
            "The embedding model could not be downloaded or loaded. Check your internet connection, "
            "access to huggingface.co, installed requirements, and write permissions for data/models, then retry."
        ) from exc


def embed_chunks(chunks: list[dict], progress=None) -> np.ndarray:
    if not chunks:
        raise IndexingError("No non-empty chunks were available for embedding.")
    def notify(fraction, message):
        if progress:
            progress(fraction, message)
    notify(0, "Loading CPU embedding model (first use may download model files)…")
    with MODEL_LOCK:
        model = load_model(str(database.DB_PATH.parent / "models"))
        vectors = []
        try:
            for start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[start:start + BATCH_SIZE]
                inputs = []
                for chunk in batch:
                    context = [f"Document: {chunk['document_name']}"]
                    if chunk["section_title"]:
                        context.append(f"Section: {chunk['section_title']}")
                    if chunk["page_number"] is not None:
                        context.append(f"Page: {chunk['page_number']}")
                    inputs.append("\n".join(context + [chunk["chunk_text"]]))
                vectors.append(model.encode(
                    inputs, batch_size=BATCH_SIZE, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False, device="cpu",
                ))
                notify(min(start + len(batch), len(chunks)) / len(chunks), f"Embedded {min(start + len(batch), len(chunks))} of {len(chunks)} chunks")
            result = np.ascontiguousarray(np.concatenate(vectors), dtype="float32")
            if result.shape != (len(chunks), EMBEDDING_DIMENSION) or not np.isfinite(result).all():
                raise ValueError("Unexpected embedding shape or values")
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError("Empty embedding")
            return result / norms
        except Exception as exc:
            raise IndexingError("Embedding generation failed. The existing document index was retained; check the model installation and retry.") from exc


def embed_question(question: str) -> np.ndarray:
    """Use the same cached CPU encoder and normalization as document indexing."""
    if not question.strip():
        raise IndexingError("Enter a question before retrieving documents.")
    with MODEL_LOCK:
        model = load_model(str(database.DB_PATH.parent / "models"))
        try:
            vector = np.ascontiguousarray(model.encode(
                [question], convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False, device="cpu",
            ), dtype="float32")
            if vector.shape != (1, EMBEDDING_DIMENSION) or not np.isfinite(vector).all():
                raise ValueError("Invalid question embedding")
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("Empty question embedding")
            return np.ascontiguousarray(vector / norm, dtype="float32")
        except Exception:
            raise IndexingError("The question could not be embedded. Check the local embedding model and retry.") from None
