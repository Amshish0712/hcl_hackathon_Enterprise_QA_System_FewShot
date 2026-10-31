"""Role-scoped chunking and embedding controls with per-document progress."""

import streamlit as st

from core.embeddings import MODEL_NAME
from core.index_errors import IndexingError
from core.indexing import index_documents, index_status, indexable_documents, rebuild_accessible_index, rebuild_clean_index
from ui.theme import page_header


def render_index_status(user_id: int) -> None:
    st.markdown("### Index Status")
    try:
        status = index_status(user_id)
    except IndexingError as exc:
        st.error(str(exc))
        return
    count_column, chunks_column = st.columns(2)
    count_column.metric("Indexed documents", status["indexed_documents"])
    chunks_column.metric("Total chunks", status["total_chunks"])
    st.caption(f"Embedding model: {status['embedding_model']}")


def render_index_documents(user: dict) -> None:
    page_header("Index Documents", "Documents / Chunking and CPU embeddings")
    st.caption("Heading-aware chunks · Target 800 characters · 150-character overlap within each section")
    try:
        documents = indexable_documents(user["id"])
    except IndexingError as exc:
        st.error(str(exc))
        return
    if not documents:
        st.error("There are no successfully extracted documents available for this account to index. Upload and extract a document first.")
        return

    st.dataframe([
        {
            "ID": document["id"], "Filename": document["original_filename"],
            "Type": document["extension"].lstrip(".").upper(),
            "Extracted characters": document["extracted_characters"],
            "Indexing status": document["indexing_status"].replace("_", " ").title(),
            "Chunk count": document["chunk_count"], "Last indexed (UTC)": document["indexed_at"],
        }
        for document in documents
    ], hide_index=True, use_container_width=True)
    labels = {document["id"]: f"{document['original_filename']} · #{document['id']}" for document in documents}
    previous_selection = st.session_state.get("index_selection", [])
    if any(document_id not in labels for document_id in previous_selection):
        st.session_state["index_selection"] = [document_id for document_id in previous_selection if document_id in labels]
    chosen = st.multiselect("Select documents to index", list(labels), format_func=labels.get, key="index_selection")
    create = st.button("Create or refresh index", type="primary")
    st.caption("Rebuild refreshes every successfully extracted document you can index. Other users' indexed documents are preserved.")
    rebuild = st.button("Rebuild accessible index")
    clean = False
    if user["role"] == "admin":
        st.caption("Clean rebuild re-embeds every extracted document into a fresh FAISS index. Documents and files are kept; any failure keeps the current index.")
        clean = st.button("Clean rebuild entire index (admin)", key="clean_rebuild")
    st.caption(f"Model: {MODEL_NAME}. First use downloads the model automatically; later runs reuse the local cache.")
    if create or rebuild or clean:
        bar = st.progress(0)
        message = st.empty()
        def progress(fraction, description):
            bar.progress(min(1.0, max(0.0, fraction)))
            message.info(description)
        try:
            with st.spinner("Chunking and embedding documents…"):
                if clean:
                    result = rebuild_clean_index(user["id"], progress)
                elif rebuild:
                    result = rebuild_accessible_index(user["id"], progress)
                else:
                    result = index_documents(user["id"], chosen, progress)
        except IndexingError as exc:
            st.error(str(exc))
        else:
            st.session_state["index_results"] = result
            st.rerun()
        finally:
            bar.empty()
            message.empty()
    result = st.session_state.get("index_results")
    if result:
        # A role/ownership change must also restrict results retained in session state.
        visible_results = [item for item in result["results"] if item["document_id"] in labels]
        if not visible_results:
            return
        st.markdown("### Last indexing results")
        completed = sum(item["status"] == "indexed" for item in visible_results)
        created = sum(item["chunks"] for item in visible_results)
        st.write(f"Documents indexed: {completed} · Chunks created: {created}")
        st.caption(f"Embedding model: {result['embedding_model']}")
        for item in visible_results:
            with st.container(border=True):
                st.text(item["filename"])
                if item["status"] == "indexed":
                    st.success(f"Indexed successfully: {item['chunks']} chunks created.")
                else:
                    st.error(item["message"])
