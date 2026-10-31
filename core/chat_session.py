"""Session-only conversation history, scoped to current identity and live evidence."""

from core import database
from core.retrieval import authorize_evidence, chat_actor
from core.chat_config import settings
from core.trust import review_answer


def authorized_history(user_id: int, messages: list[dict]) -> list[dict]:
    """Drop complete turns when their evidence was deleted, refreshed, or made inaccessible."""
    result = []
    pending = None
    with database.connection() as db:
        is_admin = chat_actor(db, user_id)["role"] == "admin"
    for message in messages:
        if message.get("role") == "user":
            pending = {"role": "user", "content": str(message.get("content", ""))}
        elif message.get("role") == "assistant" and pending is not None:
            stored = message.get("evidence", [])
            current = authorize_evidence(user_id, stored)
            if len(current) == len(stored):
                configuration = settings()
                reviewed = review_answer(str(message.get("content", "")), current, configuration.minimum_relevance,
                                         secret=configuration.api_key)
                assistant = dict(reviewed, role="assistant", content=reviewed["answer"])
                # Retrieval diagnostics are admin-only and are dropped if the role changes.
                if is_admin and isinstance(message.get("retrieval_debug"), list):
                    assistant["retrieval_debug"] = message["retrieval_debug"]
                result.extend([pending, assistant])
            pending = None
    return result


def conversation(state, user_id: int) -> list[dict]:
    with database.connection() as db:
        actor = chat_actor(db, user_id)
    owner = (actor["id"], actor["role"])
    if state.get("chat_owner") != owner:
        state["chat_messages"] = []
        state["chat_owner"] = owner
    messages = authorized_history(user_id, state.get("chat_messages", []))
    state["chat_messages"] = messages
    return messages


def clear_conversation(state) -> None:
    state["chat_messages"] = []
