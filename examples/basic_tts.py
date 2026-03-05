"""Basic text-to-speech example using the TTS.ai Python SDK."""

from tts_ai import TTSClient

# Initialize the client
client = TTSClient(api_key="sk-tts-your-key-here")

# --- Simple generation (OpenAI-compatible endpoint) ---
# This blocks until the audio is ready and returns bytes directly.
audio = client.generate("Hello! Welcome to TTS.ai.")
with open("hello.wav", "wb") as f:
    f.write(audio)
print(f"Saved hello.wav ({len(audio)} bytes)")

# --- Generate with a different model and format ---
audio_mp3 = client.generate(
    "This is generated with the Chatterbox model.",
    model="chatterbox",
    voice="af_bella",
    output_format="mp3",
)
with open("chatterbox.mp3", "wb") as f:
    f.write(audio_mp3)
print(f"Saved chatterbox.mp3 ({len(audio_mp3)} bytes)")

# --- OpenAI drop-in replacement ---
audio_oai = client.generate(
    "TTS.ai works as a drop-in replacement for the OpenAI TTS API.",
    model="tts-1",
    voice="alloy",
    output_format="mp3",
)
with open("openai_compat.mp3", "wb") as f:
    f.write(audio_oai)
print(f"Saved openai_compat.mp3 ({len(audio_oai)} bytes)")

# --- Async generation with polling ---
# For long texts or slower models, use generate_async + poll_result.
result = client.generate_async(
    "This is a longer passage that demonstrates async generation. "
    "The job is submitted to the GPU server and we poll for the result.",
    model="kokoro",
    voice="af_heart",
)
print(f"Job submitted: {result.uuid} (status: {result.status})")

audio = client.poll_result(result.uuid, timeout=120)
with open("async_result.wav", "wb") as f:
    f.write(audio)
print(f"Saved async_result.wav ({len(audio)} bytes)")

# --- List available models ---
models = client.list_models()
print(f"\nAvailable models ({len(models)}):")
for m in models:
    print(f"  {m.name} ({m.tier}) - {m.credits_per_1k} credits/1k chars")

# --- List voices for a model ---
voices = client.list_voices(model="kokoro")
print(f"\nKokoro voices ({len(voices)}):")
for v in voices[:10]:
    print(f"  {v.voice_id}: {v.name} ({v.language}, {v.gender})")

client.close()
