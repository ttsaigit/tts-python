from .client import TTSClient
from .async_client import AsyncTTSClient
from .exceptions import (
    TTSError,
    AuthenticationError,
    RateLimitError,
    InsufficientCreditsError,
    ModelNotFoundError,
    ServerError,
    TimeoutError,
)

__version__ = "0.1.0"
__all__ = [
    "TTSClient",
    "AsyncTTSClient",
    "TTSError",
    "AuthenticationError",
    "RateLimitError",
    "InsufficientCreditsError",
    "ModelNotFoundError",
    "ServerError",
    "TimeoutError",
]
