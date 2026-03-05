"""Synchronous client for the TTS.ai API."""

import io
import os
import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import requests

from .exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    ModelNotFoundError,
    RateLimitError,
    ServerError,
    TTSError,
    TimeoutError,
)
from .models import (
    BatchResult,
    CloneResult,
    GenerationResult,
    Model,
    TranscriptionResult,
    Voice,
)


class TTSClient:
    """Synchronous client for the TTS.ai API.

    Args:
        api_key: Your TTS.ai API key (sk-tts-...). Can also be set via
            the ``TTS_API_KEY`` environment variable.
        base_url: Base URL for the TTS.ai web server.
            Defaults to ``https://tts.ai``.
        gpu_url: Base URL for the GPU server (used for polling results).
            Defaults to ``https://api.tts.ai``.
        timeout: Default request timeout in seconds.
        max_retries: Maximum number of retries for transient errors.

    Example::

        from tts_ai import TTSClient

        client = TTSClient(api_key="sk-tts-...")
        audio = client.generate("Hello world!")
        with open("output.wav", "wb") as f:
            f.write(audio)
    """

    DEFAULT_BASE_URL = "https://tts.ai"
    DEFAULT_GPU_URL = "https://api.tts.ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        gpu_url: str = DEFAULT_GPU_URL,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("TTS_API_KEY", "")
        if not self.api_key:
            raise AuthenticationError(
                "API key is required. Pass api_key or set TTS_API_KEY environment variable."
            )

        self.base_url = base_url.rstrip("/")
        self.gpu_url = gpu_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": f"tts-ai-python/0.1.0",
            }
        )

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        *,
        json: dict = None,
        data: dict = None,
        files: dict = None,
        params: dict = None,
        timeout: int = None,
        raw_response: bool = False,
    ) -> Union[dict, requests.Response]:
        """Make an HTTP request with retry logic and error handling."""
        last_exc = None
        timeout = timeout or self.timeout

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.request(
                    method,
                    url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    timeout=timeout,
                )

                if raw_response and resp.status_code == 200:
                    return resp

                if resp.status_code == 200:
                    return resp.json()

                # Handle error responses
                self._handle_error_response(resp)

            except (TTSError, AuthenticationError, InsufficientCreditsError,
                    ModelNotFoundError) as e:
                raise e

            except RateLimitError as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 60)
                    time.sleep(wait)
                    continue
                raise e

            except requests.exceptions.Timeout:
                last_exc = TimeoutError(
                    f"Request timed out after {timeout}s", status_code=None
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue

            except requests.exceptions.ConnectionError as e:
                last_exc = ServerError(
                    f"Connection error: {e}", status_code=None
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue

            except requests.exceptions.RequestException as e:
                last_exc = TTSError(f"Request failed: {e}")
                break

        if last_exc:
            raise last_exc
        raise TTSError("Request failed after retries")

    def _handle_error_response(self, resp: requests.Response) -> None:
        """Parse an error response and raise the appropriate exception."""
        try:
            body = resp.json()
        except ValueError:
            body = {}

        # OpenAI-compatible error format: {"error": {"message": "...", "type": "..."}}
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            message = error_obj.get("message", "")
            error_type = error_obj.get("type", "")
        else:
            message = body.get("message", "") or str(error_obj)
            error_type = ""

        if not message:
            message = body.get("error", f"HTTP {resp.status_code}")

        if resp.status_code == 401:
            raise AuthenticationError(message, status_code=401, response=body)
        elif resp.status_code == 402:
            raise InsufficientCreditsError(
                message,
                credits_remaining=body.get("credits_remaining"),
                credits_needed=body.get("credits_needed"),
                response=body,
            )
        elif resp.status_code == 429:
            raise RateLimitError(message, response=body)
        elif resp.status_code == 400 and "unknown model" in message.lower():
            raise ModelNotFoundError(message, status_code=400, response=body)
        elif resp.status_code >= 500:
            raise ServerError(message, status_code=resp.status_code, response=body)
        else:
            raise TTSError(message, status_code=resp.status_code, response=body)

    @staticmethod
    def _prepare_file(
        file: Union[str, Path, bytes, BinaryIO],
        field_name: str = "file",
    ) -> dict:
        """Prepare a file for multipart upload.

        Accepts a file path (str/Path), raw bytes, or a file-like object.
        """
        if isinstance(file, (str, Path)):
            path = Path(file)
            return {field_name: (path.name, open(path, "rb"), "application/octet-stream")}
        elif isinstance(file, bytes):
            return {field_name: ("audio.wav", io.BytesIO(file), "application/octet-stream")}
        else:
            # File-like object
            name = getattr(file, "name", "audio.wav")
            if hasattr(name, "split"):
                name = name.rsplit("/", 1)[-1]
            return {field_name: (name, file, "application/octet-stream")}

    # -------------------------------------------------------------------------
    # Public API: TTS
    # -------------------------------------------------------------------------

    def generate(
        self,
        text: str,
        model: str = "kokoro",
        voice: str = "af_bella",
        output_format: str = "wav",
        speed: float = None,
        **kwargs,
    ) -> bytes:
        """Generate speech audio from text (OpenAI-compatible endpoint).

        This uses the synchronous ``/v1/audio/speech`` endpoint which returns
        audio bytes directly. The server polls the GPU internally.

        Args:
            text: The text to convert to speech (max 4096 characters).
            model: TTS model name. Use ``"tts-1"`` for OpenAI compatibility
                or a native model name like ``"kokoro"``, ``"chatterbox"``.
            voice: Voice identifier. OpenAI names (``"alloy"``, ``"nova"``, etc.)
                or native voice IDs (``"af_bella"``).
            output_format: Audio format: ``"wav"``, ``"mp3"``, ``"opus"``,
                ``"flac"``, ``"aac"``.
            speed: Playback speed multiplier (0.25 - 4.0). Defaults to 1.0.
            **kwargs: Additional parameters passed to the API.

        Returns:
            Raw audio bytes in the requested format.

        Raises:
            AuthenticationError: Invalid API key.
            InsufficientCreditsError: Not enough credits.
            ModelNotFoundError: Unknown model name.
            TimeoutError: Generation timed out.
            TTSError: Other API errors.

        Example::

            audio = client.generate("Hello world!", model="kokoro", voice="af_bella")
            with open("speech.wav", "wb") as f:
                f.write(audio)
        """
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": output_format,
        }
        if speed is not None:
            payload["speed"] = speed
        payload.update(kwargs)

        resp = self._request(
            "POST",
            f"{self.base_url}/v1/audio/speech",
            json=payload,
            timeout=180,
            raw_response=True,
        )

        if isinstance(resp, requests.Response):
            return resp.content
        # Should not happen for 200, but handle gracefully
        raise TTSError("Unexpected response format from generate endpoint")

    def generate_async(
        self,
        text: str,
        model: str = "kokoro",
        voice: str = "af_bella",
        output_format: str = "wav",
        **kwargs,
    ) -> GenerationResult:
        """Start an async TTS generation job.

        Returns immediately with a job UUID that can be polled with
        :meth:`poll_result`.

        Args:
            text: Text to convert to speech.
            model: TTS model name.
            voice: Voice identifier.
            output_format: Audio format (``"wav"``, ``"mp3"``, etc.).
            **kwargs: Additional parameters (language, speed, pitch, etc.).

        Returns:
            A :class:`GenerationResult` with the job UUID and status.

        Example::

            result = client.generate_async("Hello!", model="chatterbox")
            audio = client.poll_result(result.uuid)
        """
        payload = {
            "text": text,
            "model": model,
            "voice": voice,
            "format": output_format,
        }
        payload.update(kwargs)

        data = self._request(
            "POST",
            f"{self.base_url}/api/v1/tts/",
            json=payload,
        )
        return GenerationResult.from_dict(data)

    def poll_result(
        self,
        uuid: str,
        timeout: int = 300,
        interval: float = 1.0,
    ) -> bytes:
        """Poll for a TTS job result and return audio bytes.

        Args:
            uuid: The job UUID from :meth:`generate_async`.
            timeout: Maximum time to wait in seconds.
            interval: Polling interval in seconds.

        Returns:
            Raw audio bytes.

        Raises:
            TimeoutError: If the job does not complete within the timeout.
            TTSError: If the job fails.
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self._session.get(
                    f"{self.gpu_url}/v1/speech/results/",
                    params={"uuid": uuid},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", "")

                    if status == "completed":
                        audio_url = data.get("cdn_url") or data.get("url", "")
                        if not audio_url:
                            audio_url = (
                                f"{self.gpu_url}/static/downloads/{uuid}/tts_output.wav"
                            )
                        audio_resp = self._session.get(audio_url, timeout=60)
                        if audio_resp.status_code == 200:
                            return audio_resp.content
                        raise TTSError(
                            f"Failed to download audio: HTTP {audio_resp.status_code}"
                        )

                    if status == "failed":
                        error_msg = data.get("error", "Generation failed")
                        raise TTSError(error_msg, response=data)

            except TTSError:
                raise
            except requests.exceptions.RequestException:
                pass  # Transient error, keep polling

            time.sleep(interval)

        raise TimeoutError(
            f"Job {uuid} did not complete within {timeout}s", status_code=None
        )

    # -------------------------------------------------------------------------
    # Public API: Speech-to-Text
    # -------------------------------------------------------------------------

    def transcribe(
        self,
        file: Union[str, Path, bytes, BinaryIO],
        model: str = "faster-whisper",
        language: str = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            file: Audio file as a path, bytes, or file-like object.
            model: STT model (``"faster-whisper"``, ``"sensevoice"``).
            language: Optional language code hint.
            **kwargs: Additional parameters passed to the API.

        Returns:
            A :class:`TranscriptionResult` with the transcribed text.

        Example::

            result = client.transcribe("recording.wav")
            print(result.text)
        """
        files = self._prepare_file(file, "file")
        form_data = {"model": model}
        if language:
            form_data["language"] = language
        form_data.update(kwargs)

        data = self._request(
            "POST",
            f"{self.base_url}/api/v1/stt/",
            data=form_data,
            files=files,
            timeout=300,
        )
        return TranscriptionResult.from_dict(data)

    # -------------------------------------------------------------------------
    # Public API: Voice Cloning
    # -------------------------------------------------------------------------

    def clone_voice(
        self,
        name: str,
        file: Union[str, Path, bytes, BinaryIO],
        model: str = "chatterbox",
        text: str = "",
        **kwargs,
    ) -> CloneResult:
        """Clone a voice from a reference audio file.

        Args:
            name: Name for the cloned voice.
            file: Reference audio file (path, bytes, or file-like object).
                For best results, use 10-30 seconds of clear speech.
            model: Voice cloning model. Options: ``"chatterbox"``,
                ``"cosyvoice2"``, ``"openvoice"``, ``"spark"``,
                ``"gpt-sovits"``, ``"tortoise"``, ``"indextts2"``,
                ``"glm-tts"``, ``"qwen3-tts"``.
            text: Text to synthesize with the cloned voice.
            **kwargs: Additional parameters passed to the API.

        Returns:
            A :class:`CloneResult` with the job details.

        Example::

            result = client.clone_voice("My Voice", "reference.wav",
                                        text="Hello in my cloned voice!")
        """
        files = self._prepare_file(file, "reference_audio")
        form_data = {"name": name, "model": model}
        if text:
            form_data["text"] = text
        form_data.update(kwargs)

        data = self._request(
            "POST",
            f"{self.base_url}/api/v1/voice-clone/",
            data=form_data,
            files=files,
            timeout=300,
        )
        return CloneResult.from_dict(data)

    # -------------------------------------------------------------------------
    # Public API: Voices and Models
    # -------------------------------------------------------------------------

    def list_voices(self, model: str = None, language: str = None) -> List[Voice]:
        """List available voices.

        Args:
            model: Filter by model name.
            language: Filter by language code.

        Returns:
            List of :class:`Voice` objects.

        Example::

            voices = client.list_voices(model="kokoro")
            for v in voices:
                print(f"{v.voice_id}: {v.name} ({v.language})")
        """
        params = {}
        if model:
            params["model"] = model
        if language:
            params["language"] = language

        data = self._request(
            "GET",
            f"{self.base_url}/api/v1/voices/",
            params=params,
        )
        return [Voice.from_dict(v) for v in data.get("voices", [])]

    def list_models(self) -> List[Model]:
        """List available TTS models.

        Returns:
            List of :class:`Model` objects.

        Example::

            models = client.list_models()
            for m in models:
                print(f"{m.name} ({m.tier}): {m.credits_per_1k} credits/1k chars")
        """
        data = self._request(
            "GET",
            f"{self.base_url}/api/v1/models/",
        )
        return [Model.from_dict(m) for m in data.get("models", [])]

    # -------------------------------------------------------------------------
    # Public API: Batch
    # -------------------------------------------------------------------------

    def batch_generate(
        self,
        items: List[Dict[str, Any]],
        webhook_url: str = None,
    ) -> BatchResult:
        """Start a batch TTS generation job (up to 50 items).

        Args:
            items: List of dicts, each with ``text``, ``model``, and
                ``voice`` keys. Optional keys: ``language``, ``format``,
                ``speed``.
            webhook_url: Optional URL to receive a webhook when the batch
                completes.

        Returns:
            A :class:`BatchResult` with the batch ID and item statuses.

        Example::

            result = client.batch_generate([
                {"text": "Hello!", "model": "kokoro", "voice": "af_bella"},
                {"text": "Goodbye!", "model": "kokoro", "voice": "af_heart"},
            ])
            print(f"Batch {result.batch_id}: {result.total} items")
        """
        payload = {"texts": items}
        if webhook_url:
            payload["webhook_url"] = webhook_url

        data = self._request(
            "POST",
            f"{self.base_url}/api/v1/tts/batch/",
            json=payload,
        )
        return BatchResult.from_dict(data)

    def batch_result(self, batch_id: str) -> BatchResult:
        """Check the status of a batch TTS job.

        Args:
            batch_id: The batch ID from :meth:`batch_generate`.

        Returns:
            A :class:`BatchResult` with current progress.
        """
        data = self._request(
            "GET",
            f"{self.base_url}/api/v1/tts/batch/result/",
            params={"batch_id": batch_id},
        )
        return BatchResult.from_dict(data)

    def batch_generate_and_wait(
        self,
        items: List[Dict[str, Any]],
        timeout: int = 600,
        interval: float = 3.0,
    ) -> BatchResult:
        """Start a batch job and wait for all items to complete.

        Args:
            items: List of generation items (see :meth:`batch_generate`).
            timeout: Maximum time to wait in seconds.
            interval: Polling interval in seconds.

        Returns:
            The final :class:`BatchResult`.

        Raises:
            TimeoutError: If the batch does not complete within the timeout.
        """
        result = self.batch_generate(items)
        start = time.time()

        while time.time() - start < timeout:
            time.sleep(interval)
            result = self.batch_result(result.batch_id)
            if result.status == "completed":
                return result

        raise TimeoutError(
            f"Batch {result.batch_id} did not complete within {timeout}s"
        )
