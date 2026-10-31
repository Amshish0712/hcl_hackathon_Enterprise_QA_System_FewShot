"""Single source of truth for the role-to-page access policy."""

ROLE_PAGES = {
    "admin": ("Dashboard", "User Management", "Upload Documents", "Document Management", "Index Documents", "Chat"),
    "editor": ("Dashboard", "Upload Documents", "Document Management", "Index Documents", "Chat"),
    "viewer": ("Dashboard", "Document Management", "Chat"),
}


def allowed_pages(role: str) -> tuple[str, ...]:
    return ROLE_PAGES.get(role, ())


def can_access(role: str, page: str) -> bool:
    return page in allowed_pages(role)
