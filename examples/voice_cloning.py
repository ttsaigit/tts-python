"""Voice cloning example using the TTS.ai Python SDK."""

from tts_ai import TTSClient

client = TTSClient(api_key="sk-tts-your-key-here")

# --- Clone a voice from a reference audio file ---
# For best results, use 10-30 seconds of clear speech with minimal background noise.
result = client.clone_voice(
    name="My Custom Voice",
    file="reference_audio.wav",
    model="chatterbox",
    text="Hello! This is my cloned voice speaking.",
)
print(f"Voice clone job: {result.uuid}")
print(f"Status: {result.status}")

# --- Clone with different models ---
# Each cloning model has different characteristics:

# CosyVoice2 - Good for multilingual cloning
result_cosy = client.clone_voice(
    name="Multilingual Voice",
    file="reference_audio.wav",
    model="cosyvoice2",
    text="This voice can speak in multiple languages.",
)
print(f"CosyVoice2 clone: {result_cosy.uuid}")

# OpenVoice - Fast voice conversion
result_ov = client.clone_voice(
    name="Quick Clone",
    file="reference_audio.wav",
    model="openvoice",
    text="OpenVoice provides fast voice cloning.",
)
print(f"OpenVoice clone: {result_ov.uuid}")

# --- Clone from bytes ---
with open("reference_audio.wav", "rb") as f:
    audio_bytes = f.read()

result_bytes = client.clone_voice(
    name="Voice From Bytes",
    file=audio_bytes,
    model="chatterbox",
    text="Cloned from raw audio bytes.",
)
print(f"Clone from bytes: {result_bytes.uuid}")

# --- List available cloning models ---
# Cloning models: chatterbox, cosyvoice2, glm-tts, gpt-sovits,
# indextts2, openvoice, spark, tortoise, qwen3-tts
models = client.list_models()
cloning_models = ["chatterbox", "cosyvoice2", "glm-tts", "gpt-sovits",
                  "indextts2", "openvoice", "spark", "tortoise", "qwen3-tts"]
print("\nCloning-capable models:")
for m in models:
    if m.name in cloning_models:
        print(f"  {m.name} ({m.tier})")

client.close()
