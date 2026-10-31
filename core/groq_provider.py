"""Groq chat-completions generation and allowlisted, credential-safe errors."""

from core.chat_config import ChatError, ChatSetupError, protect_provider_logs, public_text


SAFE_ERROR_TYPES = {
    "APIError", "APIStatusError", "APIConnectionError", "APITimeoutError", "BadRequestError",
    "AuthenticationError", "PermissionDeniedError", "NotFoundError", "ConflictError",
    "UnprocessableEntityError", "RateLimitError", "InternalServerError", "ConnectionError", "TimeoutError",
}
NETWORK_ERROR_TYPES = {"APIConnectionError", "APITimeoutError"}


def safe_error_message(error: Exception) -> str:
    """Only allowlisted labels and numeric HTTP codes may appear in a public error."""
    raw_code = getattr(error, "status_code", None)
    code = raw_code if type(raw_code) is int and 100 <= raw_code <= 599 else None
    raw_type = type(error).__name__
    error_type = raw_type if raw_type in SAFE_ERROR_TYPES else "GroqError"
    if code in (401, 403):
        explanation = "Groq authentication failed. Check GROQ_API_KEY and its permissions in the Groq console."
    elif code == 429:
        explanation = "Groq rate limit or quota reached. Wait, then retry, or check your Groq plan limits."
    elif code == 404:
        explanation = "The configured Groq model is unavailable. Check GROQ_MODEL and model access in the Groq console."
    elif code is not None and code >= 500:
        explanation = "Groq is temporarily unavailable. Retry shortly or check the configured model's availability."
    elif raw_type in NETWORK_ERROR_TYPES or isinstance(error, (ConnectionError, TimeoutError, OSError)):
        explanation = "Groq network request failed or timed out. Check your internet connection, proxy/firewall settings, and retry."
    elif code in (400, 413, 422):
        explanation = "Groq rejected the request. Check GROQ_MODEL and the Groq API configuration."
    else:
        explanation = "Groq could not complete the request. Check the API configuration, model access, and network connection, then retry."
    labels = [error_type] + ([f"HTTP {code}"] if code is not None else [])
    return explanation + " (" + "; ".join(labels) + ")"


def groq_completion(configuration, messages, no_evidence: str) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ChatSetupError("Groq SDK is unavailable. Install requirements.txt in the active environment and restart Streamlit.") from None
    protect_provider_logs(configuration.api_key)
    try:
        # No tools are defined, so the model can only answer from the supplied passages.
        client = Groq(api_key=configuration.api_key, timeout=30.0, max_retries=1)
        response = client.chat.completions.create(
            model=configuration.model, temperature=0.1, max_tokens=2048,
            messages=[{"role": message["role"], "content": message["content"]} for message in messages],
        )
        message = response.choices[0].message if response.choices else None
        if message is not None and getattr(message, "tool_calls", None):
            return no_evidence
        text = ((message.content if message is not None else None) or "").strip()
        if not text:
            raise ChatError("Groq returned no answer. Please try the question again.")
        return public_text(text, configuration.api_key)
    except ChatError:
        raise
    except Exception as error:
        # No raw exceptions, response bodies, request headers, or chained SDK errors.
        raise ChatError(safe_error_message(error)) from None
