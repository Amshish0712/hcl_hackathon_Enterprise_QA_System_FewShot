"""Streamlit conversation with one visible [S1] citation and collapsed debug evidence."""

import re
import streamlit as st

from core.chat_config import ChatError, public_text, settings
from core.chat_session import clear_conversation, conversation
from core.rag import answer_question, bounded_evidence
from core.retrieval import accessible_chunk_count
from core.trust import CONFLICT_WARNING, finite_score
from ui.theme import page_header


SUGGESTED_QUESTIONS = (
    "What retention period is stated in the documents?",
    "Which approval steps are required?",
    "What access restrictions are described?",
)


def render_rag_status(user_id: int) -> None:
    st.markdown("### RAG Status")
    configuration = settings()
    try:
        count = accessible_chunk_count(user_id)
        messages = conversation(st.session_state, user_id)
    except ChatError as exc:
        st.error(str(exc))
        return
    chunk_column, message_column = st.columns(2)
    chunk_column.metric("Indexed chunks for chat", count)
    message_column.metric("Session messages", len(messages))
    status = "Available" if configuration.configured and count else ("API key required" if not configuration.configured else "Index documents first")
    st.caption(f"{configuration.label} model: {configuration.model} · Chat: {status}")


def _score(value, digits=3):
    return None if value is None else round(value, digits)


def _debug_rows(debug: list[dict], key: str) -> list[dict]:
    rows = []
    for item in debug:
        similarity = finite_score(item.get("similarity_score"))
        rows.append({
            "Rank": item.get("candidate_rank"),
            "Filename": public_text(item["filename"], key) if item.get("passed_access") else "—",
            "Chunk ID": item.get("chunk_id"),
            "Preview": " ".join(public_text(item["preview"], key).split()) if item.get("passed_access") else "—",
            "Raw similarity (inner product)": _score(similarity),
            "Normalized score (0–1)": _score((similarity + 1) / 2) if similarity is not None else None,
            "BM25": _score(item.get("bm25_score")),
            "Keyword coverage": _score(item.get("keyword_coverage"), 2),
            "Semantic rank": item.get("semantic_rank"),
            "BM25 rank": item.get("keyword_rank"),
            "RRF score": _score(item.get("fused_score"), 4),
            "Access filter": "passed" if item.get("passed_access") else "filtered",
            "Sent to Gemini": "yes" if item.get("selected") else ("duplicate" if item.get("duplicate") else "no"),
        })
    return rows


def _evidence_panel(evidence: list[dict], debug: list[dict] | None = None) -> None:
    with st.expander("Retrieved evidence", expanded=False):
        if not evidence:
            st.caption("No accessible indexed passages were retrieved for this answer.")
        scores = [score for item in evidence if (score := finite_score(item.get("similarity_score"))) is not None]
        if scores:
            st.caption(f"Top retrieval score: {max(scores):.3f}")
        for passage in bounded_evidence(evidence, settings().api_key):
            st.text(f"Rank {passage['rank']} · {passage['filename']}")
            st.caption(f"Document ID: {passage['document_id']} · Chunk ID: {passage['chunk_id']}")
            if passage.get("section_title"):
                st.text(f"Section: {passage['section_title']}")
            if passage.get("page_number") is not None:
                st.caption(f"Page: {passage['page_number']}")
            st.code(passage["chunk_text"], language=None)


def _primary_source(message: dict) -> dict | None:
    """The validated [S1] source is always the rank-1 retrieved passage."""
    sources = [source for source in message.get("sources", [])
               if source.get("source_label") == "S1" and source.get("rank") == 1]
    return sources[0] if sources else None


def _trust_panel(message: dict, turn: int) -> None:
    primary = _primary_source(message)
    if primary:
        st.markdown("#### Sources")
        st.caption("Top-ranked supporting source. Other retrieved passages remain in Retrieved evidence.")
        for source in [primary]:
            # The anchor contains server-generated integers only; document text stays plain text.
            anchor = f"source-{turn}-{int(source['chunk_id'])}"
            st.markdown(f'<span id="{anchor}"></span>', unsafe_allow_html=True)
            st.text(public_text(f"[S1] {source['filename']}"))
            st.text(public_text(f"Section: {source.get('section_title') or 'Not available'}"))
            page = source.get("page_number")
            st.caption(f"Page: {page if page is not None else 'Not available'} · Chunk ID: {source['chunk_id']} · Retrieval rank: 1")
            st.code(public_text(source["excerpt"]), language=None)
    if message.get("conflict_warning"):
        st.warning(CONFLICT_WARNING)


def _linked_answer(message: dict, turn: int, key: str) -> str:
    text = public_text(message["content"], key)
    primary = _primary_source(message)
    if primary is None:
        return text
    # Collapse repeated per-sentence [S1] markers into one linked marker at the end.
    text = re.sub(r"[ \t]*\[S[1-4]\]", "", text)
    text = re.sub(r"[ \t]+([.!?,;:])", r"\1", text).rstrip()
    return f"{text} [S1](#source-{turn}-{int(primary['chunk_id'])})"


def render_chat(user: dict) -> None:
    page_header("Chat", "Ask questions grounded in your accessible indexed documents")
    configuration = settings()
    try:
        messages = conversation(st.session_state, user["id"])
        count = accessible_chunk_count(user["id"])
    except ChatError as exc:
        st.error(str(exc))
        return
    if st.button("Clear conversation", key="clear_conversation"):
        clear_conversation(st.session_state)
        st.rerun()

    if not configuration.configured:
        st.error(f"Chat setup is incomplete: configure {configuration.key_variable} in the project's .env file or server environment, then restart Streamlit.")
    if not count:
        st.warning("No indexed documents are available for your account. An administrator or editor must index accessible documents before you can ask questions.")
    ready = configuration.configured and count > 0
    for turn, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(_linked_answer(message, turn, configuration.api_key) if message["role"] == "assistant"
                        else public_text(message["content"], configuration.api_key))
            if message["role"] == "assistant":
                _trust_panel(message, turn)
                _evidence_panel(message.get("evidence", []),
                                message.get("retrieval_debug") if user["role"] == "admin" else None)

    st.caption("Suggested test questions — answers depend on the content of your indexed documents.")
    suggestion = None
    for column, text in zip(st.columns(3), SUGGESTED_QUESTIONS):
        if column.button(text, disabled=not ready, use_container_width=True):
            suggestion = text
    typed = st.chat_input("Ask a question about your indexed documents", disabled=not ready, key="rag_question")
    question = suggestion or typed
    if question and ready:
        question = public_text(question, configuration.api_key)
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.spinner("Retrieving accessible evidence and generating a grounded answer…"):
                result = answer_question(user["id"], question, messages)
        except ChatError as exc:
            st.error(str(exc))
        else:
            st.session_state["chat_messages"] = messages + [
                {"role": "user", "content": question},
                dict(result, role="assistant", content=result["answer"]),
            ]
            st.rerun()
