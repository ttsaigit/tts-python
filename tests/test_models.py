"""Tests for tts_ai.models dataclasses."""

from tts_ai.models import (
    BatchItem,
    BatchResult,
    CloneResult,
    GenerationResult,
    Model,
    TranscriptionResult,
    Voice,
)


class TestVoice:
    def test_from_dict_full(self):
        data = {
            "voice_id": "af_bella",
            "name": "Bella",
            "model_name": "kokoro",
            "language": "en",
            "gender": "female",
            "preview_url": "https://cdn.tts.ai/previews/bella.wav",
            "is_premium": False,
            "tags": "warm,friendly",
        }
        voice = Voice.from_dict(data)
        assert voice.voice_id == "af_bella"
        assert voice.name == "Bella"
        assert voice.model_name == "kokoro"
        assert voice.language == "en"
        assert voice.gender == "female"
        assert voice.preview_url == "https://cdn.tts.ai/previews/bella.wav"
        assert voice.is_premium is False
        assert voice.tags == "warm,friendly"

    def test_from_dict_minimal(self):
        voice = Voice.from_dict({})
        assert voice.voice_id == ""
        assert voice.name == ""
        assert voice.model_name == ""
        assert voice.language == ""
        assert voice.gender == ""
        assert voice.is_premium is False

    def test_from_dict_premium(self):
        voice = Voice.from_dict({"voice_id": "v1", "name": "Premium", "is_premium": True})
        assert voice.is_premium is True


class TestModel:
    def test_from_dict_full(self):
        data = {"name": "kokoro", "tier": "free", "credits_per_1k": 0}
        model = Model.from_dict(data)
        assert model.name == "kokoro"
        assert model.tier == "free"
        assert model.credits_per_1k == 0

    def test_from_dict_minimal(self):
        model = Model.from_dict({})
        assert model.name == ""
        assert model.tier == "free"
        assert model.credits_per_1k == 0

    def test_from_dict_premium(self):
        model = Model.from_dict({"name": "chatterbox", "tier": "premium", "credits_per_1k": 10})
        assert model.tier == "premium"
        assert model.credits_per_1k == 10


class TestGenerationResult:
    def test_from_dict(self):
        data = {
            "uuid": "abc-123",
            "status": "queued",
            "share_uuid": "share-456",
            "url": "https://api.tts.ai/static/downloads/abc-123/tts_output.wav",
            "cdn_url": "https://cdn.tts.ai/abc-123.wav",
        }
        result = GenerationResult.from_dict(data)
        assert result.uuid == "abc-123"
        assert result.status == "queued"
        assert result.share_uuid == "share-456"
        assert result.url == "https://api.tts.ai/static/downloads/abc-123/tts_output.wav"
        assert result.cdn_url == "https://cdn.tts.ai/abc-123.wav"
        assert result.raw == data

    def test_from_dict_minimal(self):
        result = GenerationResult.from_dict({})
        assert result.uuid == ""
        assert result.status == "queued"


class TestTranscriptionResult:
    def test_from_dict(self):
        data = {
            "text": "Hello world",
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.5, "text": "Hello world"}],
        }
        result = TranscriptionResult.from_dict(data)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.raw == data

    def test_from_dict_minimal(self):
        result = TranscriptionResult.from_dict({})
        assert result.text == ""
        assert result.language == ""
        assert result.segments == []


class TestBatchItem:
    def test_from_dict(self):
        data = {
            "index": 0,
            "uuid": "item-1",
            "status": "completed",
            "url": "https://cdn.tts.ai/item-1.wav",
            "credits": 5,
        }
        item = BatchItem.from_dict(data)
        assert item.index == 0
        assert item.uuid == "item-1"
        assert item.status == "completed"
        assert item.url == "https://cdn.tts.ai/item-1.wav"
        assert item.credits == 5
        assert item.error == ""


class TestBatchResult:
    def test_from_dict(self):
        data = {
            "batch_id": "batch-abc",
            "status": "completed",
            "total": 2,
            "completed": 2,
            "credits_charged": 10,
            "items": [
                {"index": 0, "uuid": "u1", "status": "completed"},
                {"index": 1, "uuid": "u2", "status": "completed"},
            ],
        }
        result = BatchResult.from_dict(data)
        assert result.batch_id == "batch-abc"
        assert result.status == "completed"
        assert result.total == 2
        assert result.completed == 2
        assert result.credits_charged == 10
        assert len(result.items) == 2
        assert result.items[0].uuid == "u1"
        assert result.items[1].uuid == "u2"

    def test_from_dict_minimal(self):
        result = BatchResult.from_dict({"batch_id": "b1"})
        assert result.batch_id == "b1"
        assert result.status == "processing"
        assert result.items == []


class TestCloneResult:
    def test_from_dict(self):
        data = {"uuid": "clone-1", "status": "queued", "url": ""}
        result = CloneResult.from_dict(data)
        assert result.uuid == "clone-1"
        assert result.status == "queued"
        assert result.raw == data

    def test_from_dict_minimal(self):
        result = CloneResult.from_dict({})
        assert result.uuid == ""
        assert result.status == ""
