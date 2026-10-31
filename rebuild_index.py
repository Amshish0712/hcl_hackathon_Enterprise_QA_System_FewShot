"""Check or safely rebuild the FAISS index: `python rebuild_index.py [--check]`.

A rebuild re-chunks and re-embeds every successfully extracted document into a
fresh index. Uploaded files, extracted text, users, and document rows are kept.
Run it while Streamlit is stopped, or at least while nobody is indexing.
"""

import sys

from core.database import initialize_database
from core.index_errors import IndexingError
from core.indexing import _rebuild_all, index_health


def main(arguments: list[str]) -> int:
    initialize_database()
    if "--check" not in arguments:
        try:
            result = _rebuild_all(lambda fraction, message: print(f"[{fraction:4.0%}] {message}"))
        except IndexingError as exc:
            print(f"Rebuild failed: {exc}")
            return 1
        print(f"Rebuilt {result['documents_indexed']} documents into {result['chunks_created']} chunks.")
    health = index_health()
    print(f"Vectors: {health.get('vectors')} · Chunks: {health.get('chunks')} · Mappings: {health.get('mappings')}")
    for problem in health["problems"]:
        print(f"Problem: {problem}")
    print("Index is consistent." if health["consistent"] else "Index is NOT consistent.")
    return 0 if health["consistent"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
