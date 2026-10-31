"""Role-scoped inventory and document details in the existing workspace style."""

import streamlit as st

from core.document_management import (
    ManagementError, delete_document, download_original, get_document, get_inventory,
    preview_document, summarize_inventory,
)
from ui.theme import page_header


def render_document_management(user: dict) -> None:
    page_header("Document Management", "Documents / Inventory and extracted text")
    notice = st.session_state.pop("management_notice", None)
    if notice:
        st.success("Document deleted. Its metadata and available stored files have been removed.")
        for warning in notice["warnings"]:
            st.warning(warning)
    try:
        accessible = get_inventory(user["id"])
        with st.expander("Search and filters", expanded=True):
            search = st.text_input("Search filename or uploader", key="inventory_search")
            type_column, status_column, uploader_column = st.columns(3)
            extension = type_column.selectbox(
                "File type", [None] + sorted({item["extension"] for item in accessible}),
                format_func=lambda value: "All types" if value is None else value.lstrip(".").upper(),
                key="inventory_type",
            )
            status = status_column.selectbox(
                "Extraction status", [None] + sorted({"completed", "failed"} | {item["extraction_status"] for item in accessible}),
                format_func=lambda value: "All statuses" if value is None else value.title(), key="inventory_status",
            )
            uploaders = {item["uploaded_by"]: f"{item['uploader_name']} ({item['uploader_email']})" for item in accessible}
            uploader_id = uploader_column.selectbox(
                "Uploader", [None] + sorted(uploaders, key=uploaders.get),
                format_func=lambda value: "All uploaders" if value is None else uploaders[value],
                key="inventory_uploader",
            )
            start_column, end_column = st.columns(2)
            date_from = start_column.date_input("Uploaded from (UTC)", value=None, key="inventory_from")
            date_to = end_column.date_input("Uploaded through (UTC)", value=None, key="inventory_to")
        documents = get_inventory(
            user["id"], search=search, extension=extension, status=status,
            uploader_id=uploader_id, date_from=date_from, date_to=date_to,
        )
    except ManagementError as exc:
        st.error(str(exc))
        return

    summary = summarize_inventory(documents)
    columns = st.columns(4)
    for column, (label, key) in zip(columns, (
        ("Total documents", "total_documents"), ("Successful extractions", "successful_extractions"),
        ("Failed extractions", "failed_extractions"), ("Extracted characters", "total_extracted_characters"),
    )):
        column.metric(label, f"{summary[key]:,}")
    st.caption("Metrics reflect the current filters and your document access permissions. Upload dates use UTC.")
    if not documents:
        st.info("No documents match your access permissions and filters.")
        return

    st.dataframe([
        {
            "ID": item["id"], "Filename": item["original_filename"],
            "Type": item["extension"].lstrip(".").upper(), "Size (KB)": round(item["file_size"] / 1024, 2),
            "Uploader": f"{item['uploader_name']} ({item['uploader_email']})",
            "Uploaded (UTC)": item["uploaded_at"], "Extraction status": item["extraction_status"].title(),
            "Extracted characters": item["extracted_characters"],
        }
        for item in documents
    ], hide_index=True, use_container_width=True)
    by_id = {item["id"]: item for item in documents}
    if st.session_state.get("management_document") not in by_id:
        st.session_state["management_document"] = None
    document_id = st.selectbox(
        "Select a document", [None] + list(by_id),
        format_func=lambda value: "Choose a document" if value is None else f"{by_id[value]['original_filename']} · #{value}",
        key="management_document",
    )
    if document_id is not None:
        _render_details(user["id"], document_id)


def _render_details(user_id: int, document_id: int) -> None:
    try:
        document = get_document(user_id, document_id)
    except ManagementError as exc:
        st.error(str(exc))
        return
    st.markdown("### Document details")
    st.text(document["original_filename"])
    st.json({key: value for key, value in document.items() if key not in ("can_download", "can_delete")})
    st.markdown("#### Extracted text preview")
    try:
        preview = preview_document(user_id, document_id)
    except ManagementError as exc:
        st.warning(str(exc))
    else:
        if preview["text"]:
            st.code(preview["text"], language=None)
        else:
            st.info("The extracted text file is empty.")
        if preview["truncated"]:
            st.info("Preview limited to the first 5,000 characters. The extracted text contains more content.")

    if document["can_download"]:
        try:
            download = download_original(user_id, document_id)
        except ManagementError as exc:
            st.warning(str(exc))
        else:
            st.download_button(
                "Download original file", data=download["content"], file_name=download["filename"],
                mime=download["mime_type"], key=f"download_{document_id}",
            )
    if document["can_delete"]:
        st.divider()
        confirmed = st.checkbox(
            "I confirm permanent deletion of this document, its original file, and its extracted text.",
            key=f"confirm_delete_{user_id}_{document_id}",
        )
        if st.button("Delete document", type="primary", disabled=not confirmed, key=f"delete_{document_id}"):
            try:
                result = delete_document(user_id, document_id, confirmed=confirmed)
            except ManagementError as exc:
                st.error(str(exc))
            else:
                st.session_state["management_notice"] = result
                st.rerun()
    else:
        st.caption("Read-only access: you can view metadata and extracted text.")
