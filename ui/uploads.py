"""Multi-file ingestion form and persisted document history."""

import streamlit as st

from core.documents import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, DocumentError, ingest_document, list_documents
from core.permissions import can_access
from ui.theme import page_header


def render_upload_documents(user: dict) -> None:
    if not can_access(user["role"], "Upload Documents"):
        st.error("You do not have permission to access this page.")
        st.stop()
    page_header("Upload Documents", "Documents / Multi-format ingestion")
    st.caption("PDF, DOCX, TXT, Markdown, and HTML · Maximum 20 MB per file · Select multiple files in one batch.")
    with st.form("document_upload_form", clear_on_submit=True):
        files = st.file_uploader(
            "Choose documents", type=list(ALLOWED_EXTENSIONS), accept_multiple_files=True,
            help="Files are validated and extracted individually. Password-protected PDFs and image-only documents are not supported.",
        )
        submitted = st.form_submit_button("Upload and extract", type="primary")

    if submitted:
        if not files:
            st.warning("Choose at least one document to upload.")
        else:
            results = []
            progress = st.progress(0, text="Processing documents…")
            for index, uploaded in enumerate(files):
                try:
                    if uploaded.size > MAX_FILE_SIZE:
                        raise DocumentError("The file exceeds the maximum size of 20 MB.")
                    document = ingest_document(uploaded.name, uploaded.getvalue(), uploaded.type, user["id"])
                except DocumentError as exc:
                    results.append({"filename": uploaded.name, "status": "rejected", "message": str(exc)})
                else:
                    results.append({
                        "filename": uploaded.name, "status": "completed",
                        "message": f"Uploaded and extracted successfully ({document['extracted_characters']:,} characters).",
                    })
                progress.progress((index + 1) / len(files), text=f"Processed {index + 1} of {len(files)} documents")
            st.session_state["upload_results"] = results
            progress.empty()

    results = st.session_state.get("upload_results", [])
    if results:
        st.markdown("### Last upload results")
        for result in results:
            # Plain text filename prevents Markdown in an uploaded filename becoming UI markup.
            with st.container(border=True):
                st.text(result["filename"])
                if result["status"] == "completed":
                    st.success(result["message"])
                else:
                    st.error(result["message"])

    st.markdown("### Ingestion history")
    try:
        documents = list_documents(user["id"])
    except DocumentError as exc:
        st.error(str(exc))
        st.stop()
    if not documents:
        st.info("No documents have been ingested yet. Successfully processed documents will appear here.")
        return
    st.dataframe([
        {
            "Filename": document["original_filename"],
            "Type": document["extension"].lstrip(".").upper(),
            "Size (KB)": round(document["file_size"] / 1024, 2),
            "Uploaded by": f"{document['uploader_name']} ({document['uploader_email']})",
            "Date (UTC)": document["uploaded_at"],
            "Status": document["extraction_status"].title(),
            "Extracted characters": document["extracted_characters"],
        }
        for document in documents
    ], hide_index=True, use_container_width=True)
