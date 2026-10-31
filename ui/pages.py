"""Existing workspace views and document workflow entry points."""

import streamlit as st

from core.permissions import allowed_pages, can_access
from ui.theme import page_header
from ui.uploads import render_upload_documents
from ui.document_management import render_document_management
from ui.indexing import render_index_documents, render_index_status
from ui.chat import render_chat, render_rag_status


PAGE_DETAILS = {
    "User Management": (
        "Administration / User access",
        "Manage workspace users",
        "This area is reserved for administrators. User listing, role changes, and account management will be added in a future phase.",
    ),
}


def render_page(page: str, user: dict) -> None:
    # Enforce authorization here too, even when called without sidebar navigation.
    if not can_access(user["role"], page):
        st.error("You do not have permission to access this page.")
        st.stop()

    if page == "Upload Documents":
        render_upload_documents(user)
        return

    if page == "Document Management":
        render_document_management(user)
        return

    if page == "Index Documents":
        render_index_documents(user)
        return

    if page == "Chat":
        render_chat(user)
        return

    if page == "Dashboard":
        page_header("Dashboard", "Your workspace overview and access permissions")
        st.subheader(f"Welcome, {user['name']}")
        st.caption("Your account is authenticated. Your role determines which workspace pages you can access.")
        role_column, access_column, phase_column = st.columns(3)
        role_column.metric("Workspace role", user["role"].title())
        access_column.metric("Available pages", len(allowed_pages(user["role"])))
        phase_column.metric("Project phase", "06")
        st.markdown("### Your workspace")
        descriptions = {
            "Dashboard": "Workspace overview",
            "Upload Documents": "Multi-format upload and text extraction",
            "Document Management": "Inventory, previews, and permitted document actions",
            "Index Documents": "Document chunking and local embeddings",
            "Chat": "Grounded questions with retrieved document evidence",
        }
        for available_page in allowed_pages(user["role"]):
            with st.container(border=True):
                st.markdown(f"**{available_page}**")
                st.caption(descriptions.get(available_page, "Placeholder · Planned for a future phase"))
        st.info("Phases 1 to 6 are complete: secure role-based access, ingestion, document management, indexing, RAG chat, and citations with safe fallback.")
        render_index_status(user["id"])
        render_rag_status(user["id"])
        return

    subtitle, title, description = PAGE_DETAILS[page]
    page_header(page, subtitle)
    with st.container(border=True):
        st.subheader(title)
        st.write(description)
        st.caption("PLANNED · FUTURE PHASE")
    st.info("This is a placeholder page. No document or AI operations are enabled.")
