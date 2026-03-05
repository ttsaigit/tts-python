# TTS.ai Python SDK

Official Python SDK for the [TTS.ai](https://tts.ai) text-to-speech API. Generate high-quality speech from text using 20+ AI models, clone voices, transcribe audio, and more.

## Installation

```bash
pip install tts-ai
```

For async support:

```bash
pip install tts-ai[async]
```

## Quick Start

```python
from tts_ai import TTSClient

client = TTSClient(api_key="sk-tts-...")

# Generate speech (returns audio bytes)
audio = client.generate("Hello world!", model="kokoro", voice="af_bella")
with open("output.wav", "wb") as f:
    f.write(audio)
```

Or set the `TTS_API_KEY` environment variable:

```bash
export TTS_API_KEY=sk-tts-...
```

```python
client = TTSClient()  # Uses TTS_API_KEY from environment
```

## API Key

Get your API key at [tts.ai/account](https://tts.ai/account) after creating an account.

## Usage

### Text-to-Speech

The simplest way to generate speech. Uses the OpenAI-compatible `/v1/audio/speech` endpoint:

```python
from tts_ai import TTSClient

client = TTSClient(api_key="sk-tts-...")

# Basic generation
audio = client.generate("Hello world!")
with open("output.wav", "wb") as f:
    f.write(audio)

# With options
audio = client.generate(
    "Welcome to TTS.ai!",
    model="chatterbox",       # Any supported model
    voice="af_bella",
    output_format="mp3",
    speed=1.2,
)
```

#### OpenAI Drop-In Replacement

TTS.ai is compatible with the OpenAI TTS API format:

```python
audio = client.generate(
    "Hello from TTS.ai!",
    model="tts-1",       # Maps to kokoro
    voice="alloy",       # Maps to af_bella
    output_format="mp3",
)
```

### Async TTS (Non-Blocking)

For long-running jobs or when you need the job UUID:

```python
# Start generation (returns immediately)
result = client.generate_async("Long text here...", model="tortoise")
print(f"Job UUID: {result.uuid}")

# Poll for result
audio = client.poll_result(result.uuid, timeout=300)
with open("output.wav", "wb") as f:
    f.write(audio)
```

### Speech-to-Text

```python
# Transcribe from file path
result = client.transcribe("recording.wav")
print(result.text)
print(result.language)

# Transcribe from bytes
with open("recording.wav", "rb") as f:
    audio_bytes = f.read()
result = client.transcribe(audio_bytes, model="faster-whisper")
print(result.text)
```

### Voice Cloning

Clone a voice from a reference audio file (10-30 seconds of clear speech):

```python
result = client.clone_voice(
    name="My Voice",
    file="reference.wav",
    model="chatterbox",    # or cosyvoice2, openvoice, spark, etc.
    text="Hello in my cloned voice!",
)
print(f"Clone job: {result.uuid}")
```

Supported cloning models: `chatterbox`, `cosyvoice2`, `glm-tts`, `gpt-sovits`, `indextts2`, `openvoice`, `spark`, `tortoise`, `qwen3-tts`.

### List Voices and Models

```python
# List all voices
voices = client.list_voices()
for v in voices:
    print(f"{v.voice_id}: {v.name} ({v.language}, {v.gender})")

# Filter by model
kokoro_voices = client.list_voices(model="kokoro")

# List available models
models = client.list_models()
for m in models:
    print(f"{m.name} ({m.tier}): {m.credits_per_1k} credits per 1k chars")
```

### Batch Generation

Process up to 50 texts in a single request:

```python
items = [
    {"text": "First sentence.", "model": "kokoro", "voice": "af_bella"},
    {"text": "Second sentence.", "model": "kokoro", "voice": "af_heart"},
    {"text": "Third sentence.", "model": "chatterbox", "voice": "af_bella"},
]

# Start batch
result = client.batch_generate(items)
print(f"Batch {result.batch_id}: {result.total} items")

# Check progress
status = client.batch_result(result.batch_id)
print(f"Completed: {status.completed}/{status.total}")

# Or generate and wait for completion
result = client.batch_generate_and_wait(items, timeout=300)
```

With webhook notifications:

```python
result = client.batch_generate(
    items,
    webhook_url="https://yoursite.com/webhook/tts-complete",
)
```

### Context Manager

Use as a context manager to automatically close the HTTP session:

```python
with TTSClient(api_key="sk-tts-...") as client:
    audio = client.generate("Hello!")
```

## Async Client

For asyncio-based applications:

```python
import asyncio
from tts_ai import AsyncTTSClient

async def main():
    async with AsyncTTSClient(api_key="sk-tts-...") as client:
        # All methods are async
        audio = await client.generate("Hello world!")
        with open("output.wav", "wb") as f:
            f.write(audio)

        # Parallel generation
        tasks = [
            client.generate("First sentence.", voice="af_bella"),
            client.generate("Second sentence.", voice="af_heart"),
        ]
        results = await asyncio.gather(*tasks)

asyncio.run(main())
```

## Error Handling

```python
from tts_ai import TTSClient
from tts_ai.exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    ModelNotFoundError,
    TimeoutError,
    TTSError,
)

client = TTSClient(api_key="sk-tts-...")

try:
    audio = client.generate("Hello!", model="kokoro")
except AuthenticationError:
    print("Invalid API key")
except InsufficientCreditsError as e:
    print(f"Not enough credits (have {e.credits_remaining}, need {e.credits_needed})")
except RateLimitError:
    print("Rate limit exceeded, try again later")
except ModelNotFoundError:
    print("Model not found")
except TimeoutError:
    print("Request timed out")
except TTSError as e:
    print(f"API error ({e.status_code}): {e.message}")
```

## Available Models

| Model | Tier | Cloning |
|-------|------|---------|
| kokoro | Free | No |
| piper | Free | No |
| vits | Free | No |
| melotts | Free | No |
| chatterbox | Premium | Yes |
| cosyvoice2 | Standard | Yes |
| bark | Standard | No |
| dia | Standard | No |
| glm-tts | Standard | Yes |
| gpt-sovits | Standard | Yes |
| indextts2 | Standard | Yes |
| openvoice | Premium | Yes |
| orpheus | Standard | No |
| parler | Standard | No |
| qwen3-tts | Standard | Yes |
| sesame-csm | Premium | No |
| spark | Standard | Yes |
| styletts2 | Premium | No |
| tortoise | Premium | Yes |

Use `client.list_models()` for the latest list with credit costs.

## Configuration

| Parameter | Environment Variable | Default |
|-----------|---------------------|---------|
| `api_key` | `TTS_API_KEY` | (required) |
| `base_url` | - | `https://tts.ai` |
| `gpu_url` | - | `https://api.tts.ai` |
| `timeout` | - | `120` seconds |
| `max_retries` | - | `3` |

## License

MIT
