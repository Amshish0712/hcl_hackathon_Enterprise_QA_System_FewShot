"""Server-side Gemini configuration and credential-safe public text/logging."""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import traceback

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
# provider: (label, key variable, model variable, default model, placeholder key)
PROVIDERS = {
    "gemini": ("Gemini", "GEMINI_API_KEY", "GEMINI_MODEL", DEFAULT_GEMINI_MODEL, "your_gemini_api_key_here"),
    "groq": ("Groq", "GROQ_API_KEY", "GROQ_MODEL", DEFAULT_GROQ_MODEL, "your_groq_api_key_here"),
}
LOGGER_PREFIXES = ("google.genai", "groq", "httpx", "httpcore")


class ChatError(ValueError):
    """A sanitized error suitable for presentation to the user."""


class ChatSetupError(ChatError):
    """Chat needs local configuration before it can generate an answer."""


@dataclass(frozen=True)
class GeminiSettings:
    model: str
    api_key: str = field(repr=False)
    minimum_relevance: float = 0.30
    provider: str = "gemini"

    @property
    def label(self) -> str:
        return PROVIDERS[self.provider][0]

    @property
    def key_variable(self) -> str:
        return PROVIDERS[self.provider][1]

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_key != PROVIDERS[self.provider][4])


def settings() -> GeminiSettings:
    # Explicit project path, no CWD-dependent lookup or overriding deployment variables.
    load_dotenv(PROJECT_ROOT / ".env", override=False, interpolate=False)
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider not in PROVIDERS:
        provider = "gemini"
    _, key_variable, model_variable, default_model, _ = PROVIDERS[provider]
    key = os.environ.get(key_variable, "").strip()
    model = os.environ.get(model_variable, "").strip() or default_model
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/_.:-]{0,119}", model) or (key and key in model):
        model = default_model
    # Groq "compound" models run built-in web search and code tools; answers must stay document-only.
    if provider == "groq" and "compound" in model.lower():
        model = default_model
    try:
        minimum = float(os.environ.get("RAG_MIN_RELEVANCE_SCORE", "0.30"))
        if not 0 <= minimum <= 1:
            minimum = 0.30
    except (ValueError, TypeError):
        minimum = 0.30
    return GeminiSettings(model=model, api_key=key, minimum_relevance=minimum, provider=provider)


def public_text(value: str, key: str | None = None) -> str:
    secret = key if key is not None else settings().api_key
    return value.replace(secret, "[redacted]") if secret else value


class _CredentialFilter(logging.Filter):
    def __init__(self, key: str):
        super().__init__()
        self._key = key

    def filter(self, record):
        # Suppress SDK/transport logs entirely: even redacted request headers
        # and raw provider error bodies must not reach application logs.
        if record.name.startswith(LOGGER_PREFIXES):
            return False
        record.msg = public_text(record.getMessage(), self._key)
        record.args = ()
        if record.exc_info:
            record.exc_text = public_text("".join(traceback.format_exception(*record.exc_info)), self._key)
            record.exc_info = None
        if record.exc_text:
            record.exc_text = public_text(record.exc_text, self._key)
        if record.stack_info:
            record.stack_info = public_text(record.stack_info, self._key)
        return True


def protect_provider_logs(key: str) -> None:
    """Scrub SDK/transport debug messages and exception bodies before handlers emit them."""
    if not key:
        return
    redactor = _CredentialFilter(key)
    loggers = [logging.getLogger(), logging.getLogger("google.genai._api_client"),
               *(logging.getLogger(prefix) for prefix in LOGGER_PREFIXES)]
    loggers.extend(logger for name, logger in list(logging.Logger.manager.loggerDict.items())
                   if name.startswith(LOGGER_PREFIXES) and isinstance(logger, logging.Logger))
    for logger in loggers:
        # Avoid accumulating duplicate filters for the same configured credential.
        if not any(isinstance(item, _CredentialFilter) and item._key == key for item in logger.filters):
            logger.addFilter(redactor)
        for handler in logger.handlers:
            if not any(isinstance(item, _CredentialFilter) and item._key == key for item in handler.filters):
                handler.addFilter(redactor)
    if logging.lastResort is not None:
        logging.lastResort.addFilter(redactor)
