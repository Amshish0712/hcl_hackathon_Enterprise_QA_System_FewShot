"""Gemini Developer API generation and allowlisted, credential-safe errors."""

from core.chat_config import ChatError, ChatSetupError, protect_provider_logs, public_text


SAFE_STATUSES = {
    "INVALID_ARGUMENT", "UNAUTHENTICATED", "PERMISSION_DENIED", "NOT_FOUND",
    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED",
    "FAILED_PRECONDITION", "UNKNOWN",
}
SAFE_ERROR_TYPES = {
    "APIError", "ClientError", "ServerError", "ConnectError", "ReadError", "WriteError",
    "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException",
    "TransportError", "NetworkError", "RemoteProtocolError", "ConnectionError", "TimeoutError",
}


def safe_error_message(error: Exception) -> str:
    """Only allowlisted labels and numeric HTTP codes may appear in a public error."""
    raw_code = getattr(error, "code", None)
    if raw_code is None:
        raw_code = getattr(getattr(error, "response", None), "status_code", None)
    code = int(raw_code) if isinstance(raw_code, (int, str)) and len(str(raw_code)) == 3 and str(raw_code).isdigit() else None
    if code is not None and not 100 <= code <= 599:
        code = None
    raw_status = getattr(error, "status", None)
    status = raw_status if isinstance(raw_status, str) and raw_status in SAFE_STATUSES else None
    raw_type = type(error).__name__
    error_type = raw_type if raw_type in SAFE_ERROR_TYPES else "GeminiError"
    # Inspect provider details only for classification. Never interpolate or log them.
    message = getattr(error, "message", "")
    message = message.lower() if isinstance(message, str) else ""
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        body = details.get("error", details)
        details = body.get("details", []) if isinstance(body, dict) else []
    invalid_key = "api key not valid" in message or "api_key_invalid" in message
    if isinstance(details, list):
        invalid_key = invalid_key or any(
            isinstance(item, dict) and item.get("reason") in {
                "API_KEY_INVALID", "API_KEY_EXPIRED", "API_KEY_NOT_FOUND", "API_KEY_SERVICE_BLOCKED",
            } for item in details
        )
    if invalid_key or code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        explanation = "Gemini authentication failed. Check GEMINI_API_KEY and its API restrictions or permissions in Google AI Studio."
    elif code == 429 or status == "RESOURCE_EXHAUSTED":
        explanation = "Gemini quota or rate limit reached. Check your Google AI Studio project quota and billing, wait, then retry."
    elif code == 404 or status == "NOT_FOUND" or (code == 400 and "model" in message and any(term in message for term in ("not found", "not supported", "unavailable"))):
        explanation = "The configured Gemini model is unavailable. Check GEMINI_MODEL and model access in Google AI Studio."
    elif code in (500, 502, 503, 504) or status in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"):
        explanation = "Gemini is temporarily unavailable. Retry shortly or check the configured model's availability."
    elif isinstance(error, (ConnectionError, TimeoutError, OSError)) or raw_type in {
        "ConnectError", "ReadError", "WriteError", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
        "PoolTimeout", "TimeoutException", "TransportError", "NetworkError", "RemoteProtocolError",
    }:
        explanation = "Gemini network request failed or timed out. Check your internet connection, proxy/firewall settings, and retry."
    elif code == 400 or status == "INVALID_ARGUMENT":
        explanation = "Gemini rejected the request. Check GEMINI_MODEL and the Gemini API configuration."
    else:
        explanation = "Gemini could not complete the request. Check the API configuration, model access, and network connection, then retry."
    labels = [error_type]
    if code is not None:
        labels.append(f"HTTP {code}")
    if status:
        labels.append(status)
    return explanation + " (" + "; ".join(labels) + ")"


def gemini_completion(configuration, messages, no_evidence: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ChatSetupError("Gemini SDK is unavailable. Install requirements.txt in the active environment and restart Streamlit.") from None
    protect_provider_logs(configuration.api_key)
    try:
        with genai.Client(
            api_key=configuration.api_key, vertexai=False,
            http_options=types.HttpOptions(timeout=30000, retry_options=types.HttpRetryOptions(attempts=2)),
        ) as client:
            response = client.models.generate_content(
                model=configuration.model, contents=messages[1]["content"],
                config=types.GenerateContentConfig(
                    system_instruction=messages[0]["content"], temperature=0.1, max_output_tokens=2048,
                    tools=[], automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        if getattr(response, "function_calls", None):
            return no_evidence
        text = (response.text or "").strip()
        if not text:
            raise ChatError("Gemini returned no answer. Please try the question again.")
        return public_text(text, configuration.api_key)
    except ChatError:
        raise
    except Exception as error:
        # No raw exceptions, response bodies, request headers, or chained SDK errors.
        raise ChatError(safe_error_message(error)) from None
