"""Asynchronous client for the TTS.ai API."""

import io
import os
import asyncio
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]

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


def _require_aiohttp():
    if aiohttp is None:
        raise ImportError(
            "aiohttp is required for AsyncTTSClient. "
            "Install it with: pip install tts-ai[async]"
        )


class AsyncTTSClient:
    """Asynchronous client for the TTS.ai API using aiohttp.

    Args:
        api_key: Your TTS.ai API key (sk-tts-...). Can also be set via
            the ``TTS_API_KEY`` environment variable.
        base_url: Base URL for the TTS.ai web server.
        gpu_url: Base URL for the GPU server (used for polling results).
        timeout: Default request timeout in seconds.
        max_retries: Maximum number of retries for transient errors.

    Example::

        import asyncio
        from tts_ai import AsyncTTSClient

        async def main():
            async with AsyncTTSClient(api_key="sk-tts-...") as client:
                audio = await client.generate("Hello world!")
                with open("output.wav", "wb") as f:
                    f.write(audio)

        asyncio.run(main())
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
        _require_aiohttp()

        self.api_key = api_key or os.environ.get("TTS_API_KEY", "")
        if not self.api_key:
            raise AuthenticationError(
                "API key is required. Pass api_key or set TTS_API_KEY environment variable."
            )

        self.base_url = base_url.rstrip("/")
        self.gpu_url = gpu_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> "aiohttp.ClientSession":
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "tts-ai-python/0.1.0",
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json: dict = None,
        data: Any = None,
        params: dict = None,
        timeout: int = None,
        raw_response: bool = False,
    ) -> Union[dict, bytes]:
        """Make an HTTP request with retry logic and error handling."""
        session = await self._get_session()
        last_exc = None
        req_timeout = aiohttp.ClientTimeout(total=timeout or self.timeout)

        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(
                    method, url, json=json, data=data, params=params,
                    timeout=req_timeout,
                ) as resp:
                    if resp.status == 200:
                        if raw_response:
                            return await resp.read()
                        return await resp.json()

                    await self._handle_error_response(resp)

            except RateLimitError as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 60)
                    await asyncio.sleep(wait)
                    continue
                raise e

            except (AuthenticationError, InsufficientCreditsError,
                    ModelNotFoundError) as e:
                raise e

            except TTSError as e:
                raise e

            except asyncio.TimeoutError:
                last_exc = TimeoutError(
                    f"Request timed out after {timeout or self.timeout}s"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue

            except aiohttp.ClientError as e:
                last_exc = ServerError(f"Connection error: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue

        if last_exc:
            raise last_exc
        raise TTSError("Request failed after retries")

    async def _handle_error_response(self, resp: "aiohttp.ClientResponse") -> None:
        """Parse an error response and raise the appropriate exception."""
        try:
            body = await resp.json()
        except Exception:
            body = {}

        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            message = error_obj.get("message", "")
        else:
            message = body.get("message", "") or str(error_obj)

        if not message:
            message = body.get("error", f"HTTP {resp.status}")

        if resp.status == 401:
            raise AuthenticationError(message, status_code=401, response=body)
        elif resp.status == 402:
            raise InsufficientCreditsError(
                message,
                credits_remaining=body.get("credits_remaining"),
                credits_needed=body.get("credits_needed"),
                response=body,
            )
        elif resp.status == 429:
            raise RateLimitError(message, response=body)
        elif resp.status == 400 and "unknown model" in message.lower():
            raise ModelNotFoundError(message, status_code=400, response=body)
        elif resp.status >= 500:
            raise ServerError(message, status_code=resp.status, response=body)
        else:
            raise TTSError(message, status_code=resp.status, response=body)

    @staticmethod
    def _prepare_file(
        file: Union[str, Path, bytes, BinaryIO],
        field_name: str = "file",
    ) -> "aiohttp.FormData":
        """Prepare a file for multipart upload."""
        form = aiohttp.FormData()
        if isinstance(file, (str, Path)):
            path = Path(file)
            form.add_field(
                field_name,
                open(path, "rb"),
                filename=path.name,
                content_type="application/octet-stream",
            )
        elif isinstance(file, bytes):
            form.add_field(
                field_name,
                io.BytesIO(file),
                filename="audio.wav",
                content_type="application/octet-stream",
            )
        else:
            name = getattr(file, "name", "audio.wav")
            if hasattr(name, "rsplit"):
                name = name.rsplit("/", 1)[-1]
            form.add_field(
                field_name,
                file,
                filename=name,
                content_type="application/octet-stream",
            )
        return form

    @staticmethod
    def _prepare_multipart(
        file: Union[str, Path, bytes, BinaryIO],
        field_name: str,
        form_fields: Dict[str, str],
    ) -> "aiohttp.FormData":
        """Prepare a multipart form with a file and additional fields."""
        form = aiohttp.FormData()
        for key, val in form_fields.items():
            form.add_field(key, str(val))

        if isinstance(file, (str, Path)):
            path = Path(file)
            form.add_field(
                field_name,
                open(path, "rb"),
                filename=path.name,
                content_type="application/octet-stream",
            )
        elif isinstance(file, bytes):
            form.add_field(
                field_name,
                io.BytesIO(file),
                filename="audio.wav",
                content_type="application/octet-stream",
            )
        else:
            name = getattr(file, "name", "audio.wav")
            if hasattr(name, "rsplit"):
                name = name.rsplit("/", 1)[-1]
            form.add_field(
                field_name,
                file,
                filename=name,
                content_type="application/octet-stream",
            )
        return form

    # -------------------------------------------------------------------------
    # Public API: TTS
    # -------------------------------------------------------------------------

    async def generate(
        self,
        text: str,
        model: str = "kokoro",
        voice: str = "af_bella",
        output_format: str = "wav",
        speed: float = None,
        **kwargs,
    ) -> bytes:
        """Generate speech audio from text (OpenAI-compatible endpoint).

        Args:
            text: The text to convert to speech (max 4096 characters).
            model: TTS model name.
            voice: Voice identifier.
            output_format: Audio format (wav, mp3, opus, flac, aac).
            speed: Playback speed multiplier (0.25 - 4.0).

        Returns:
            Raw audio bytes.

        Example::

            audio = await client.generate("Hello world!")
            with open("output.wav", "wb") as f:
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

        result = await self._request(
            "POST",
            f"{self.base_url}/v1/audio/speech",
            json=payload,
            timeout=180,
            raw_response=True,
        )

        if isinstance(result, bytes):
            return result
        raise TTSError("Unexpected response format from generate endpoint")

    async def generate_async(
        self,
        text: str,
        model: str = "kokoro",
        voice: str = "af_bella",
        output_format: str = "wav",
        **kwargs,
    ) -> GenerationResult:
        """Start an async TTS generation job.

        Returns immediately with a job UUID for polling.

        Args:
            text: Text to convert to speech.
            model: TTS model name.
            voice: Voice identifier.
            output_format: Audio format.

        Returns:
            A :class:`GenerationResult` with the job UUID.
        """
        payload = {
            "text": text,
            "model": model,
            "voice": voice,
            "format": output_format,
        }
        payload.update(kwargs)

        data = await self._request(
            "POST",
            f"{self.base_url}/api/v1/tts/",
            json=payload,
        )
        return GenerationResult.from_dict(data)

    async def poll_result(
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
        """
        session = await self._get_session()
        start = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start < timeout:
            try:
                async with session.get(
                    f"{self.gpu_url}/v1/speech/results/",
                    params={"uuid": uuid},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        status = data.get("status", "")

                        if status == "completed":
                            audio_url = data.get("cdn_url") or data.get("url", "")
                            if not audio_url:
                                audio_url = (
                                    f"{self.gpu_url}/static/downloads/{uuid}/tts_output.wav"
                                )
                            async with session.get(
                                audio_url,
                                timeout=aiohttp.ClientTimeout(total=60),
                            ) as audio_resp:
                                if audio_resp.status == 200:
                                    return await audio_resp.read()
                                raise TTSError(
                                    f"Failed to download audio: HTTP {audio_resp.status}"
                                )

                        if status == "failed":
                            error_msg = data.get("error", "Generation failed")
                            raise TTSError(error_msg, response=data)

            except TTSError:
                raise
            except Exception:
                pass  # Transient error, keep polling

            await asyncio.sleep(interval)

        raise TimeoutError(
            f"Job {uuid} did not complete within {timeout}s"
        )

    # -------------------------------------------------------------------------
    # Public API: Speech-to-Text
    # -------------------------------------------------------------------------

    async def transcribe(
        self,
        file: Union[str, Path, bytes, BinaryIO],
        model: str = "faster-whisper",
        language: str = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            file: Audio file as a path, bytes, or file-like object.
            model: STT model.
            language: Optional language code hint.

        Returns:
            A :class:`TranscriptionResult` with the transcribed text.
        """
        form_fields = {"model": model}
        if language:
            form_fields["language"] = language
        form_fields.update({k: str(v) for k, v in kwargs.items()})

        form_data = self._prepare_multipart(file, "file", form_fields)

        data = await self._request(
            "POST",
            f"{self.base_url}/api/v1/stt/",
            data=form_data,
            timeout=300,
        )
        return TranscriptionResult.from_dict(data)

    # -------------------------------------------------------------------------
    # Public API: Voice Cloning
    # -------------------------------------------------------------------------

    async def clone_voice(
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
            file: Reference audio file.
            model: Voice cloning model.
            text: Text to synthesize with the cloned voice.

        Returns:
            A :class:`CloneResult` with the job details.
        """
        form_fields = {"name": name, "model": model}
        if text:
            form_fields["text"] = text
        form_fields.update({k: str(v) for k, v in kwargs.items()})

        form_data = self._prepare_multipart(file, "reference_audio", form_fields)

        data = await self._request(
            "POST",
            f"{self.base_url}/api/v1/voice-clone/",
            data=form_data,
            timeout=300,
        )
        return CloneResult.from_dict(data)

    # -------------------------------------------------------------------------
    # Public API: Voices and Models
    # -------------------------------------------------------------------------

    async def list_voices(
        self, model: str = None, language: str = None
    ) -> List[Voice]:
        """List available voices.

        Args:
            model: Filter by model name.
            language: Filter by language code.

        Returns:
            List of :class:`Voice` objects.
        """
        params = {}
        if model:
            params["model"] = model
        if language:
            params["language"] = language

        data = await self._request(
            "GET",
            f"{self.base_url}/api/v1/voices/",
            params=params,
        )
        return [Voice.from_dict(v) for v in data.get("voices", [])]

    async def list_models(self) -> List[Model]:
        """List available TTS models.

        Returns:
            List of :class:`Model` objects.
        """
        data = await self._request(
            "GET",
            f"{self.base_url}/api/v1/models/",
        )
        return [Model.from_dict(m) for m in data.get("models", [])]

    # -------------------------------------------------------------------------
    # Public API: Batch
    # -------------------------------------------------------------------------

    async def batch_generate(
        self,
        items: List[Dict[str, Any]],
        webhook_url: str = None,
    ) -> BatchResult:
        """Start a batch TTS generation job (up to 50 items).

        Args:
            items: List of dicts with text, model, and voice keys.
            webhook_url: Optional webhook URL for completion notification.

        Returns:
            A :class:`BatchResult` with the batch ID and item statuses.
        """
        payload = {"texts": items}
        if webhook_url:
            payload["webhook_url"] = webhook_url

        data = await self._request(
            "POST",
            f"{self.base_url}/api/v1/tts/batch/",
            json=payload,
        )
        return BatchResult.from_dict(data)

    async def batch_result(self, batch_id: str) -> BatchResult:
        """Check the status of a batch TTS job.

        Args:
            batch_id: The batch ID from :meth:`batch_generate`.

        Returns:
            A :class:`BatchResult` with current progress.
        """
        data = await self._request(
            "GET",
            f"{self.base_url}/api/v1/tts/batch/result/",
            params={"batch_id": batch_id},
        )
        return BatchResult.from_dict(data)

    async def batch_generate_and_wait(
        self,
        items: List[Dict[str, Any]],
        timeout: int = 600,
        interval: float = 3.0,
    ) -> BatchResult:
        """Start a batch job and wait for all items to complete.

        Args:
            items: List of generation items.
            timeout: Maximum time to wait in seconds.
            interval: Polling interval in seconds.

        Returns:
            The final :class:`BatchResult`.
        """
        result = await self.batch_generate(items)
        start = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start < timeout:
            await asyncio.sleep(interval)
            result = await self.batch_result(result.batch_id)
            if result.status == "completed":
                return result

        raise TimeoutError(
            f"Batch {result.batch_id} did not complete within {timeout}s"
        )
