"""Tests for the asynchronous AsyncTTSClient."""

import os
from unittest.mock import patch

import pytest
import pytest_asyncio
from aioresponses import aioresponses

from tts_ai import AsyncTTSClient
from tts_ai.exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    ModelNotFoundError,
    RateLimitError,
    ServerError,
    TTSError,
    TimeoutError,
)
from tts_ai.models import BatchResult, CloneResult, GenerationResult, TranscriptionResult


BASE_URL = "https://tts.ai"
GPU_URL = "https://api.tts.ai"
API_KEY = "sk-tts-test-key-123"


@pytest_asyncio.fixture
async def client():
    c = AsyncTTSClient(api_key=API_KEY, base_url=BASE_URL, gpu_url=GPU_URL, max_retries=0)
    yield c
    await c.close()


@pytest_asyncio.fixture
async def client_with_retries():
    c = AsyncTTSClient(api_key=API_KEY, base_url=BASE_URL, gpu_url=GPU_URL, max_retries=2)
    yield c
    await c.close()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestAsyncClientInit:
    def test_init_with_api_key(self):
        c = AsyncTTSClient(api_key=API_KEY)
        assert c.api_key == API_KEY
        assert c.base_url == "https://tts.ai"
        assert c.gpu_url == "https://api.tts.ai"

    def test_init_without_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TTS_API_KEY", None)
            with pytest.raises(AuthenticationError, match="API key is required"):
                AsyncTTSClient()

    def test_init_from_env(self):
        with patch.dict(os.environ, {"TTS_API_KEY": "sk-tts-env-key"}):
            c = AsyncTTSClient()
            assert c.api_key == "sk-tts-env-key"

    def test_custom_urls(self):
        c = AsyncTTSClient(
            api_key=API_KEY,
            base_url="https://custom.example.com/",
            gpu_url="https://gpu.example.com/",
        )
        assert c.base_url == "https://custom.example.com"
        assert c.gpu_url == "https://gpu.example.com"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with AsyncTTSClient(api_key=API_KEY) as c:
            assert c.api_key == API_KEY


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestAsyncGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_audio_bytes(self, client):
        audio_data = b"\x00\x01\x02\x03fake-wav-data"
        with aioresponses() as m:
            m.post(f"{BASE_URL}/v1/audio/speech", body=audio_data, status=200)
            result = await client.generate("Hello world!")
        assert result == audio_data

    @pytest.mark.asyncio
    async def test_generate_default_params(self, client):
        with aioresponses() as m:
            m.post(f"{BASE_URL}/v1/audio/speech", body=b"audio", status=200)
            result = await client.generate("Hello")
            assert result == b"audio"
            assert len(m.requests) == 1


# ---------------------------------------------------------------------------
# generate_async()
# ---------------------------------------------------------------------------


class TestAsyncGenerateAsync:
    @pytest.mark.asyncio
    async def test_generate_async_returns_result(self, client):
        with aioresponses() as m:
            m.post(
                f"{BASE_URL}/api/v1/tts/",
                payload={"uuid": "job-123", "status": "queued"},
                status=200,
            )
            result = await client.generate_async("Hello!")
        assert isinstance(result, GenerationResult)
        assert result.uuid == "job-123"
        assert result.status == "queued"


# ---------------------------------------------------------------------------
# poll_result()
# ---------------------------------------------------------------------------


class TestAsyncPollResult:
    @pytest.mark.asyncio
    async def test_poll_completed(self, client):
        with aioresponses() as m:
            m.get(
                f"{GPU_URL}/v1/speech/results/?uuid=abc",
                payload={"status": "completed", "cdn_url": "https://cdn.tts.ai/abc.wav"},
                status=200,
            )
            m.get("https://cdn.tts.ai/abc.wav", body=b"audio-bytes", status=200)
            result = await client.poll_result("abc")
        assert result == b"audio-bytes"

    @pytest.mark.asyncio
    async def test_poll_failed_raises(self, client):
        with aioresponses() as m:
            m.get(
                f"{GPU_URL}/v1/speech/results/?uuid=fail",
                payload={"status": "failed", "error": "GPU OOM"},
                status=200,
            )
            with pytest.raises(TTSError, match="GPU OOM"):
                await client.poll_result("fail")

    @pytest.mark.asyncio
    async def test_poll_timeout(self, client):
        with aioresponses() as m:
            # Always return queued
            for _ in range(50):
                m.get(
                    f"{GPU_URL}/v1/speech/results/?uuid=slow",
                    payload={"status": "queued"},
                    status=200,
                )
            with pytest.raises(TimeoutError, match="did not complete"):
                await client.poll_result("slow", timeout=0.1, interval=0.01)


# ---------------------------------------------------------------------------
# list_voices() / list_models()
# ---------------------------------------------------------------------------


class TestAsyncListVoicesModels:
    @pytest.mark.asyncio
    async def test_list_voices(self, client):
        with aioresponses() as m:
            m.get(
                f"{BASE_URL}/api/v1/voices/",
                payload={
                    "voices": [
                        {"voice_id": "af_bella", "name": "Bella", "model_name": "kokoro"},
                    ]
                },
                status=200,
            )
            voices = await client.list_voices()
        assert len(voices) == 1
        assert voices[0].voice_id == "af_bella"

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        with aioresponses() as m:
            m.get(
                f"{BASE_URL}/api/v1/models/",
                payload={
                    "models": [
                        {"name": "kokoro", "tier": "free", "credits_per_1k": 0},
                        {"name": "chatterbox", "tier": "premium", "credits_per_1k": 10},
                    ]
                },
                status=200,
            )
            models = await client.list_models()
        assert len(models) == 2
        assert models[0].name == "kokoro"
        assert models[1].tier == "premium"


# ---------------------------------------------------------------------------
# transcribe()
# ---------------------------------------------------------------------------


class TestAsyncTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_with_bytes(self, client):
        with aioresponses() as m:
            m.post(
                f"{BASE_URL}/api/v1/stt/",
                payload={"text": "Hello world", "language": "en", "segments": []},
                status=200,
            )
            result = await client.transcribe(b"fake-audio-bytes")
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"


# ---------------------------------------------------------------------------
# clone_voice()
# ---------------------------------------------------------------------------


class TestAsyncCloneVoice:
    @pytest.mark.asyncio
    async def test_clone_voice(self, client):
        with aioresponses() as m:
            m.post(
                f"{BASE_URL}/api/v1/voice-clone/",
                payload={"uuid": "clone-1", "status": "queued"},
                status=200,
            )
            result = await client.clone_voice("My Voice", b"ref-audio", text="Hello")
        assert isinstance(result, CloneResult)
        assert result.uuid == "clone-1"


# ---------------------------------------------------------------------------
# batch_generate() / batch_result()
# ---------------------------------------------------------------------------


class TestAsyncBatch:
    @pytest.mark.asyncio
    async def test_batch_generate(self, client):
        with aioresponses() as m:
            m.post(
                f"{BASE_URL}/api/v1/tts/batch/",
                payload={
                    "batch_id": "batch-1",
                    "status": "processing",
                    "total": 2,
                    "items": [
                        {"index": 0, "uuid": "u1", "status": "queued"},
                        {"index": 1, "uuid": "u2", "status": "queued"},
                    ],
                },
                status=200,
            )
            result = await client.batch_generate([
                {"text": "Hello", "model": "kokoro", "voice": "af_bella"},
                {"text": "World", "model": "kokoro", "voice": "af_heart"},
            ])
        assert isinstance(result, BatchResult)
        assert result.batch_id == "batch-1"
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_batch_result(self, client):
        with aioresponses() as m:
            m.get(
                f"{BASE_URL}/api/v1/tts/batch/result/?batch_id=batch-1",
                payload={
                    "batch_id": "batch-1",
                    "status": "completed",
                    "total": 1,
                    "completed": 1,
                    "items": [{"index": 0, "uuid": "u1", "status": "completed"}],
                },
                status=200,
            )
            result = await client.batch_result("batch-1")
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestAsyncErrorHandling:
    @pytest.mark.asyncio
    async def test_401_raises_authentication_error(self, client):
        with aioresponses() as m:
            m.get(
                f"{BASE_URL}/api/v1/models/",
                payload={"error": {"message": "Invalid API key"}},
                status=401,
            )
            with pytest.raises(AuthenticationError, match="Invalid API key"):
                await client.list_models()

    @pytest.mark.asyncio
    async def test_402_raises_insufficient_credits(self, client):
        with aioresponses() as m:
            m.post(
                f"{BASE_URL}/v1/audio/speech",
                payload={
                    "error": {"message": "Insufficient credits"},
                    "credits_remaining": 5,
                    "credits_needed": 10,
                },
                status=402,
            )
            with pytest.raises(InsufficientCreditsError) as exc_info:
                await client.generate("Hello")
            assert exc_info.value.credits_remaining == 5
            assert exc_info.value.credits_needed == 10

    @pytest.mark.asyncio
    async def test_429_raises_rate_limit_error(self, client):
        with aioresponses() as m:
            m.get(
                f"{BASE_URL}/api/v1/models/",
                payload={"error": {"message": "Too many requests"}},
                status=429,
            )
            with pytest.raises(RateLimitError):
                await client.list_models()

    @pytest.mark.asyncio
    async def test_500_raises_server_error(self, client):
        with aioresponses() as m:
            m.get(
                f"{BASE_URL}/api/v1/models/",
                payload={"error": {"message": "Internal error"}},
                status=500,
            )
            with pytest.raises(ServerError):
                await client.list_models()

    @pytest.mark.asyncio
    async def test_400_unknown_model(self, client):
        with aioresponses() as m:
            m.post(
                f"{BASE_URL}/v1/audio/speech",
                payload={"error": {"message": "Unknown model: nope"}},
                status=400,
            )
            with pytest.raises(ModelNotFoundError):
                await client.generate("Hello", model="nope")


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestAsyncRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_on_429(self, client_with_retries):
        import asyncio
        from unittest.mock import AsyncMock

        with aioresponses() as m:
            # First two calls return 429, third succeeds
            m.get(
                f"{BASE_URL}/api/v1/models/",
                payload={"error": {"message": "Rate limited"}},
                status=429,
            )
            m.get(
                f"{BASE_URL}/api/v1/models/",
                payload={"error": {"message": "Rate limited"}},
                status=429,
            )
            m.get(
                f"{BASE_URL}/api/v1/models/",
                payload={"models": [{"name": "kokoro"}]},
                status=200,
            )

            with patch("tts_ai.async_client.asyncio.sleep", new_callable=AsyncMock):
                models = await client_with_retries.list_models()
            assert len(models) == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self, client_with_retries):
        call_count = 0

        with aioresponses() as m:
            m.get(
                f"{BASE_URL}/api/v1/models/",
                payload={"error": {"message": "Invalid key"}},
                status=401,
            )

            with pytest.raises(AuthenticationError):
                await client_with_retries.list_models()
