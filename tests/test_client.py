"""Tests for the synchronous TTSClient."""

import os
import tempfile
from unittest.mock import patch

import pytest
import responses

from tts_ai import TTSClient
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


@pytest.fixture
def client():
    c = TTSClient(api_key=API_KEY, base_url=BASE_URL, gpu_url=GPU_URL, max_retries=0)
    yield c
    c.close()


@pytest.fixture
def client_with_retries():
    c = TTSClient(api_key=API_KEY, base_url=BASE_URL, gpu_url=GPU_URL, max_retries=2)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_init_with_api_key(self):
        c = TTSClient(api_key=API_KEY)
        assert c.api_key == API_KEY
        assert c.base_url == "https://tts.ai"
        assert c.gpu_url == "https://api.tts.ai"
        c.close()

    def test_init_without_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TTS_API_KEY", None)
            with pytest.raises(AuthenticationError, match="API key is required"):
                TTSClient()

    def test_init_from_env(self):
        with patch.dict(os.environ, {"TTS_API_KEY": "sk-tts-env-key"}):
            c = TTSClient()
            assert c.api_key == "sk-tts-env-key"
            c.close()

    def test_custom_base_url(self):
        c = TTSClient(api_key=API_KEY, base_url="https://custom.example.com/")
        assert c.base_url == "https://custom.example.com"
        c.close()

    def test_custom_gpu_url(self):
        c = TTSClient(api_key=API_KEY, gpu_url="https://gpu.example.com/")
        assert c.gpu_url == "https://gpu.example.com"
        c.close()

    def test_context_manager(self):
        with TTSClient(api_key=API_KEY) as c:
            assert c.api_key == API_KEY

    def test_default_timeout(self):
        c = TTSClient(api_key=API_KEY)
        assert c.timeout == 120
        c.close()

    def test_custom_timeout(self):
        c = TTSClient(api_key=API_KEY, timeout=60)
        assert c.timeout == 60
        c.close()

    def test_auth_header_set(self):
        c = TTSClient(api_key=API_KEY)
        assert c._session.headers["Authorization"] == f"Bearer {API_KEY}"
        c.close()


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    @responses.activate
    def test_generate_returns_audio_bytes(self, client):
        audio_data = b"\x00\x01\x02\x03fake-wav-data"
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/audio/speech",
            body=audio_data,
            status=200,
            content_type="audio/wav",
        )

        result = client.generate("Hello world!")
        assert result == audio_data

    @responses.activate
    def test_generate_sends_correct_payload(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/audio/speech",
            body=b"audio",
            status=200,
        )

        client.generate("Test text", model="chatterbox", voice="narrator", output_format="mp3", speed=1.5)

        req = responses.calls[0].request
        import json
        body = json.loads(req.body)
        assert body["input"] == "Test text"
        assert body["model"] == "chatterbox"
        assert body["voice"] == "narrator"
        assert body["response_format"] == "mp3"
        assert body["speed"] == 1.5

    @responses.activate
    def test_generate_default_params(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/audio/speech",
            body=b"audio",
            status=200,
        )

        client.generate("Hello")
        import json
        body = json.loads(responses.calls[0].request.body)
        assert body["model"] == "kokoro"
        assert body["voice"] == "af_bella"
        assert body["response_format"] == "wav"
        assert "speed" not in body

    @responses.activate
    def test_generate_extra_kwargs(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/audio/speech",
            body=b"audio",
            status=200,
        )

        client.generate("Hello", language="en")
        import json
        body = json.loads(responses.calls[0].request.body)
        assert body["language"] == "en"


# ---------------------------------------------------------------------------
# generate_async()
# ---------------------------------------------------------------------------


class TestGenerateAsync:
    @responses.activate
    def test_generate_async_returns_generation_result(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/tts/",
            json={"uuid": "job-123", "status": "queued"},
            status=200,
        )

        result = client.generate_async("Hello!")
        assert isinstance(result, GenerationResult)
        assert result.uuid == "job-123"
        assert result.status == "queued"

    @responses.activate
    def test_generate_async_sends_correct_payload(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/tts/",
            json={"uuid": "j1", "status": "queued"},
            status=200,
        )

        client.generate_async("Test", model="bark", voice="v2_en_speaker_0", output_format="mp3")
        import json
        body = json.loads(responses.calls[0].request.body)
        assert body["text"] == "Test"
        assert body["model"] == "bark"
        assert body["voice"] == "v2_en_speaker_0"
        assert body["format"] == "mp3"


# ---------------------------------------------------------------------------
# poll_result()
# ---------------------------------------------------------------------------


class TestPollResult:
    @responses.activate
    def test_poll_completed_with_cdn_url(self, client):
        responses.add(
            responses.GET,
            f"{GPU_URL}/v1/speech/results/",
            json={"status": "completed", "cdn_url": "https://cdn.tts.ai/abc.wav"},
            status=200,
        )
        responses.add(
            responses.GET,
            "https://cdn.tts.ai/abc.wav",
            body=b"audio-bytes",
            status=200,
        )

        result = client.poll_result("abc")
        assert result == b"audio-bytes"

    @responses.activate
    def test_poll_completed_fallback_url(self, client):
        responses.add(
            responses.GET,
            f"{GPU_URL}/v1/speech/results/",
            json={"status": "completed"},
            status=200,
        )
        responses.add(
            responses.GET,
            f"{GPU_URL}/static/downloads/job-1/tts_output.wav",
            body=b"fallback-audio",
            status=200,
        )

        result = client.poll_result("job-1")
        assert result == b"fallback-audio"

    @responses.activate
    def test_poll_queued_then_completed(self, client):
        # First call: queued
        responses.add(
            responses.GET,
            f"{GPU_URL}/v1/speech/results/",
            json={"status": "queued"},
            status=200,
        )
        # Second call: completed
        responses.add(
            responses.GET,
            f"{GPU_URL}/v1/speech/results/",
            json={"status": "completed", "cdn_url": "https://cdn.tts.ai/done.wav"},
            status=200,
        )
        responses.add(
            responses.GET,
            "https://cdn.tts.ai/done.wav",
            body=b"done-audio",
            status=200,
        )

        result = client.poll_result("uuid-1", interval=0.01)
        assert result == b"done-audio"
        # Should have polled at least twice
        assert len(responses.calls) >= 2

    @responses.activate
    def test_poll_failed_raises(self, client):
        responses.add(
            responses.GET,
            f"{GPU_URL}/v1/speech/results/",
            json={"status": "failed", "error": "GPU out of memory"},
            status=200,
        )

        with pytest.raises(TTSError, match="GPU out of memory"):
            client.poll_result("fail-1")

    @responses.activate
    def test_poll_timeout(self, client):
        # Always return queued
        responses.add(
            responses.GET,
            f"{GPU_URL}/v1/speech/results/",
            json={"status": "queued"},
            status=200,
        )

        with pytest.raises(TimeoutError, match="did not complete"):
            client.poll_result("slow-1", timeout=0.05, interval=0.01)


# ---------------------------------------------------------------------------
# list_voices() / list_models()
# ---------------------------------------------------------------------------


class TestListVoicesModels:
    @responses.activate
    def test_list_voices(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/voices/",
            json={
                "voices": [
                    {"voice_id": "af_bella", "name": "Bella", "model_name": "kokoro"},
                    {"voice_id": "af_heart", "name": "Heart", "model_name": "kokoro"},
                ]
            },
            status=200,
        )

        voices = client.list_voices()
        assert len(voices) == 2
        assert voices[0].voice_id == "af_bella"
        assert voices[1].name == "Heart"

    @responses.activate
    def test_list_voices_with_filters(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/voices/",
            json={"voices": [{"voice_id": "v1", "name": "V1", "model_name": "bark"}]},
            status=200,
        )

        voices = client.list_voices(model="bark", language="en")
        req = responses.calls[0].request
        assert "model=bark" in req.url
        assert "language=en" in req.url

    @responses.activate
    def test_list_models(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            json={
                "models": [
                    {"name": "kokoro", "tier": "free", "credits_per_1k": 0},
                    {"name": "chatterbox", "tier": "premium", "credits_per_1k": 10},
                ]
            },
            status=200,
        )

        models = client.list_models()
        assert len(models) == 2
        assert models[0].name == "kokoro"
        assert models[1].tier == "premium"


# ---------------------------------------------------------------------------
# transcribe()
# ---------------------------------------------------------------------------


class TestTranscribe:
    @responses.activate
    def test_transcribe_with_bytes(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/stt/",
            json={"text": "Hello world", "language": "en", "segments": []},
            status=200,
        )

        result = client.transcribe(b"fake-audio-bytes")
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"

    @responses.activate
    def test_transcribe_with_file_path(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/stt/",
            json={"text": "Transcribed text", "language": "en"},
            status=200,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
            tmp_path = f.name

        try:
            result = client.transcribe(tmp_path)
            assert result.text == "Transcribed text"
        finally:
            os.unlink(tmp_path)

    @responses.activate
    def test_transcribe_with_language(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/stt/",
            json={"text": "Hola mundo", "language": "es"},
            status=200,
        )

        result = client.transcribe(b"audio", model="faster-whisper", language="es")
        assert result.text == "Hola mundo"


# ---------------------------------------------------------------------------
# clone_voice()
# ---------------------------------------------------------------------------


class TestCloneVoice:
    @responses.activate
    def test_clone_voice(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/voice-clone/",
            json={"uuid": "clone-1", "status": "queued"},
            status=200,
        )

        result = client.clone_voice("My Voice", b"reference-audio", text="Hello in my voice")
        assert isinstance(result, CloneResult)
        assert result.uuid == "clone-1"
        assert result.status == "queued"

    @responses.activate
    def test_clone_voice_with_model(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/voice-clone/",
            json={"uuid": "c2", "status": "queued"},
            status=200,
        )

        result = client.clone_voice("Voice2", b"audio", model="cosyvoice2")
        assert result.uuid == "c2"


# ---------------------------------------------------------------------------
# batch_generate() / batch_result()
# ---------------------------------------------------------------------------


class TestBatch:
    @responses.activate
    def test_batch_generate(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/tts/batch/",
            json={
                "batch_id": "batch-1",
                "status": "processing",
                "total": 2,
                "completed": 0,
                "items": [
                    {"index": 0, "uuid": "u1", "status": "queued"},
                    {"index": 1, "uuid": "u2", "status": "queued"},
                ],
            },
            status=200,
        )

        items = [
            {"text": "Hello", "model": "kokoro", "voice": "af_bella"},
            {"text": "World", "model": "kokoro", "voice": "af_heart"},
        ]
        result = client.batch_generate(items)
        assert isinstance(result, BatchResult)
        assert result.batch_id == "batch-1"
        assert result.total == 2
        assert len(result.items) == 2

    @responses.activate
    def test_batch_generate_with_webhook(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/api/v1/tts/batch/",
            json={"batch_id": "b2", "status": "processing", "total": 1, "items": []},
            status=200,
        )

        client.batch_generate(
            [{"text": "Hi", "model": "kokoro", "voice": "af_bella"}],
            webhook_url="https://example.com/webhook",
        )
        import json
        body = json.loads(responses.calls[0].request.body)
        assert body["webhook_url"] == "https://example.com/webhook"

    @responses.activate
    def test_batch_result(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/tts/batch/result/",
            json={
                "batch_id": "batch-1",
                "status": "completed",
                "total": 1,
                "completed": 1,
                "items": [{"index": 0, "uuid": "u1", "status": "completed"}],
            },
            status=200,
        )

        result = client.batch_result("batch-1")
        assert result.status == "completed"
        assert result.completed == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @responses.activate
    def test_401_raises_authentication_error(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
            status=401,
        )

        with pytest.raises(AuthenticationError, match="Invalid API key") as exc_info:
            client.list_models()
        assert exc_info.value.status_code == 401

    @responses.activate
    def test_402_raises_insufficient_credits(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/audio/speech",
            json={
                "error": {"message": "Insufficient credits"},
                "credits_remaining": 5,
                "credits_needed": 10,
            },
            status=402,
        )

        with pytest.raises(InsufficientCreditsError) as exc_info:
            client.generate("Hello")
        assert exc_info.value.status_code == 402
        assert exc_info.value.credits_remaining == 5
        assert exc_info.value.credits_needed == 10

    @responses.activate
    def test_429_raises_rate_limit_error(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            json={"error": {"message": "Too many requests"}},
            status=429,
        )

        with pytest.raises(RateLimitError, match="Too many requests"):
            client.list_models()

    @responses.activate
    def test_500_raises_server_error(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            json={"error": {"message": "Internal server error"}},
            status=500,
        )

        with pytest.raises(ServerError, match="Internal server error") as exc_info:
            client.list_models()
        assert exc_info.value.status_code == 500

    @responses.activate
    def test_400_unknown_model_raises_model_not_found(self, client):
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/audio/speech",
            json={"error": {"message": "Unknown model: bad-model"}},
            status=400,
        )

        with pytest.raises(ModelNotFoundError, match="Unknown model"):
            client.generate("Hello", model="bad-model")

    @responses.activate
    def test_generic_error(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            json={"error": {"message": "Something went wrong"}},
            status=418,
        )

        with pytest.raises(TTSError, match="Something went wrong") as exc_info:
            client.list_models()
        assert exc_info.value.status_code == 418

    @responses.activate
    def test_error_response_stored(self, client):
        error_body = {"error": {"message": "Bad request"}, "detail": "extra info"}
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            json=error_body,
            status=400,
        )

        with pytest.raises(TTSError) as exc_info:
            client.list_models()
        assert exc_info.value.response == error_body

    @responses.activate
    def test_non_json_error_response(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            body="Bad Gateway",
            status=502,
            content_type="text/plain",
        )

        with pytest.raises(ServerError):
            client.list_models()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    @responses.activate
    def test_retry_on_429(self, client_with_retries):
        # First two calls return 429, third succeeds
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            json={"error": {"message": "Rate limited"}},
            status=429,
        )
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            json={"error": {"message": "Rate limited"}},
            status=429,
        )
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            json={"models": [{"name": "kokoro"}]},
            status=200,
        )

        with patch("tts_ai.client.time.sleep"):  # Skip actual sleep
            models = client_with_retries.list_models()
        assert len(models) == 1
        assert len(responses.calls) == 3

    @responses.activate
    def test_retry_exhausted_raises(self, client_with_retries):
        for _ in range(3):
            responses.add(
                responses.GET,
                f"{BASE_URL}/api/v1/models/",
                json={"error": {"message": "Rate limited"}},
                status=429,
            )

        with patch("tts_ai.client.time.sleep"):
            with pytest.raises(RateLimitError):
                client_with_retries.list_models()

    @responses.activate
    def test_no_retry_on_auth_error(self, client_with_retries):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            json={"error": {"message": "Invalid key"}},
            status=401,
        )

        with pytest.raises(AuthenticationError):
            client_with_retries.list_models()
        # Should only make one request, no retries
        assert len(responses.calls) == 1

    @responses.activate
    def test_no_retry_on_402(self, client_with_retries):
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/audio/speech",
            json={"error": {"message": "No credits"}},
            status=402,
        )

        with pytest.raises(InsufficientCreditsError):
            client_with_retries.generate("Hello")
        assert len(responses.calls) == 1


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    @responses.activate
    def test_request_timeout_raises(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            body=requests.exceptions.Timeout("Connection timed out"),
        )

        with pytest.raises(TimeoutError):
            client.list_models()

    @responses.activate
    def test_connection_error_raises(self, client):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/models/",
            body=requests.exceptions.ConnectionError("Connection refused"),
        )

        with pytest.raises(ServerError, match="Connection error"):
            client.list_models()


# ---------------------------------------------------------------------------
# save() helper — we test it indirectly via generate + write
# ---------------------------------------------------------------------------


class TestSaveAudio:
    @responses.activate
    def test_write_audio_to_file(self, client):
        audio_data = b"\x00\x01\x02\x03fake-wav"
        responses.add(
            responses.POST,
            f"{BASE_URL}/v1/audio/speech",
            body=audio_data,
            status=200,
        )

        result = client.generate("Hello")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(result)
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                assert f.read() == audio_data
        finally:
            os.unlink(tmp_path)


# Need this import for Timeout exception
import requests
