"""Streamlit entry point: run with `streamlit run app.py`."""

import sqlite3

import streamlit as st

from core.auth import get_user, login, register, ValidationError
from core.database import initialize_database
from core.permissions import allowed_pages, can_access
from ui.pages import render_page
from ui.theme import apply_theme, page_header, sidebar_identity


def authentication_screen() -> None:
    page_header("Secure Document Assistant", "Enterprise workspace · Phase 1")
    st.markdown("### Welcome to your secure workspace")
    st.caption("Sign in to access your workspace, or create an editor or viewer account.")
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])

    with login_tab:
        with st.form("login_form", clear_on_submit=True):
            email = st.text_input("Email", placeholder="you@enterprise.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            user = login(email, password)
            if user is None:
                st.error("Invalid email or password.")
            else:
                st.session_state["user"] = user
                st.rerun()

    with register_tab:
        with st.form("registration_form", clear_on_submit=True):
            name = st.text_input("Full name", placeholder="Your full name", max_chars=80)
            email = st.text_input("Work email", placeholder="you@enterprise.com")
            password = st.text_input("Create password", type="password", help="At least 8 characters and at most 72 UTF-8 bytes.")
            confirmation = st.text_input("Confirm password", type="password")
            role = st.selectbox("Workspace role", ["viewer", "editor"], format_func=str.title)
            st.caption("Viewer: dashboard and chat. Editor: also upload and document management.")
            submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if submitted:
            if password != confirmation:
                st.error("Passwords do not match.")
            else:
                try:
                    register(name, email, password, role)
                except ValidationError as exc:
                    st.error(str(exc))
                else:
                    st.success("Account created. Open Sign in to access your workspace.")


def workspace() -> None:
    # Reload identity from SQLite so authorization always uses the stored role.
    saved_user = st.session_state.get("user")
    user = get_user(saved_user.get("id")) if isinstance(saved_user, dict) else None
    if user is None:
        st.session_state.pop("user", None)
        authentication_screen()
        return

    st.session_state["user"] = user
    with st.sidebar:
        st.markdown("### SECURE DOC")
        st.caption("Enterprise document workspace")
        st.divider()
        sidebar_identity(user)
        st.divider()
        pages = allowed_pages(user["role"])
        if st.session_state.get("navigation") not in pages:
            st.session_state["navigation"] = "Dashboard"
        selected_page = st.radio("Workspace", pages, key="navigation")
        st.divider()
        st.caption("PHASES 1 TO 6 · ACCESS, INGESTION, MANAGEMENT, INDEXING, RAG AND TRUST")
        if st.button("Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if not can_access(user["role"], selected_page):
        st.error("You do not have permission to access this page.")
        st.stop()
    render_page(selected_page, user)


def main() -> None:
    st.set_page_config(page_title="Secure Document Assistant", page_icon="🔐", layout="wide")
    apply_theme()
    try:
        initialize_database()
        workspace()
    except sqlite3.Error:
        st.error("The database is unavailable. Check that the data directory is writable and restart the app.")
        st.stop()


if __name__ == "__main__":
    main()
