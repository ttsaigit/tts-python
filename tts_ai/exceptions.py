"""Exception classes for the TTS.ai Python SDK."""


class TTSError(Exception):
    """Base exception for all TTS.ai API errors."""

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        self.message = message
        self.status_code = status_code
        self.response = response or {}
        super().__init__(self.message)


class AuthenticationError(TTSError):
    """Raised when the API key is invalid or missing."""
    pass


class RateLimitError(TTSError):
    """Raised when the rate limit is exceeded (HTTP 429)."""

    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, status_code=429, **kwargs)


class InsufficientCreditsError(TTSError):
    """Raised when the account does not have enough credits (HTTP 402)."""

    def __init__(
        self,
        message: str = "Insufficient credits",
        credits_remaining: int = None,
        credits_needed: int = None,
        **kwargs,
    ):
        self.credits_remaining = credits_remaining
        self.credits_needed = credits_needed
        super().__init__(message, status_code=402, **kwargs)


class ModelNotFoundError(TTSError):
    """Raised when the requested model does not exist."""
    pass


class ServerError(TTSError):
    """Raised when the server returns a 5xx error."""
    pass


class TimeoutError(TTSError):
    """Raised when a request or polling operation times out."""
    pass
