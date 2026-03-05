"""Data models for the TTS.ai Python SDK."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Voice:
    """A voice available for TTS generation."""

    voice_id: str
    name: str
    model_name: str
    language: str = ""
    gender: str = ""
    preview_url: str = ""
    is_premium: bool = False
    tags: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Voice":
        return cls(
            voice_id=data.get("voice_id", ""),
            name=data.get("name", ""),
            model_name=data.get("model_name", ""),
            language=data.get("language", ""),
            gender=data.get("gender", ""),
            preview_url=data.get("preview_url", ""),
            is_premium=data.get("is_premium", False),
            tags=data.get("tags", ""),
        )


@dataclass
class Model:
    """A TTS model available on the platform."""

    name: str
    tier: str = "free"
    credits_per_1k: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        return cls(
            name=data.get("name", ""),
            tier=data.get("tier", "free"),
            credits_per_1k=data.get("credits_per_1k", 0),
        )


@dataclass
class GenerationResult:
    """Result from an async TTS generation request."""

    uuid: str
    status: str = "queued"
    share_uuid: str = ""
    url: str = ""
    cdn_url: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationResult":
        return cls(
            uuid=data.get("uuid", ""),
            status=data.get("status", "queued"),
            share_uuid=data.get("share_uuid", ""),
            url=data.get("url", ""),
            cdn_url=data.get("cdn_url", ""),
            raw=data,
        )


@dataclass
class TranscriptionResult:
    """Result from a speech-to-text transcription."""

    text: str
    language: str = ""
    segments: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptionResult":
        return cls(
            text=data.get("text", ""),
            language=data.get("language", ""),
            segments=data.get("segments", []),
            raw=data,
        )


@dataclass
class BatchItem:
    """A single item within a batch result."""

    index: int
    uuid: str = ""
    status: str = ""
    error: str = ""
    url: str = ""
    credits: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "BatchItem":
        return cls(
            index=data.get("index", 0),
            uuid=data.get("uuid", ""),
            status=data.get("status", ""),
            error=data.get("error", ""),
            url=data.get("url", ""),
            credits=data.get("credits", 0),
        )


@dataclass
class BatchResult:
    """Result from a batch TTS operation."""

    batch_id: str
    status: str = "processing"
    total: int = 0
    completed: int = 0
    credits_charged: int = 0
    items: List[BatchItem] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "BatchResult":
        items = [BatchItem.from_dict(item) for item in data.get("items", [])]
        return cls(
            batch_id=data.get("batch_id", ""),
            status=data.get("status", "processing"),
            total=data.get("total", 0),
            completed=data.get("completed", 0),
            credits_charged=data.get("credits_charged", 0),
            items=items,
            raw=data,
        )


@dataclass
class CloneResult:
    """Result from a voice cloning operation."""

    uuid: str = ""
    status: str = ""
    url: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "CloneResult":
        return cls(
            uuid=data.get("uuid", ""),
            status=data.get("status", ""),
            url=data.get("url", ""),
            raw=data,
        )
