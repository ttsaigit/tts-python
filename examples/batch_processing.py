"""Batch TTS processing example using the TTS.ai Python SDK."""

import time
from tts_ai import TTSClient

client = TTSClient(api_key="sk-tts-your-key-here")

# --- Batch generation (up to 50 items) ---
items = [
    {"text": "Welcome to our application.", "model": "kokoro", "voice": "af_bella"},
    {"text": "Please hold while we connect you.", "model": "kokoro", "voice": "af_heart"},
    {"text": "Your order has been confirmed.", "model": "kokoro", "voice": "af_bella"},
    {"text": "Thank you for your patience.", "model": "kokoro", "voice": "am_adam"},
    {"text": "Goodbye and have a great day!", "model": "kokoro", "voice": "af_bella"},
]

# Submit batch
result = client.batch_generate(items)
print(f"Batch {result.batch_id} submitted: {result.total} items")
print(f"Credits charged: {result.credits_charged}")

# Poll for progress
for _ in range(60):
    time.sleep(3)
    status = client.batch_result(result.batch_id)
    print(f"  Progress: {status.completed}/{status.total} ({status.status})")
    if status.status == "completed":
        print("\nAll items completed!")
        for item in status.items:
            print(f"  Item {item.index}: {item.status} - {item.url}")
        break

# --- Or use batch_generate_and_wait for convenience ---
print("\n--- Using batch_generate_and_wait ---")
items_v2 = [
    {"text": "First notification sound.", "model": "kokoro", "voice": "af_bella"},
    {"text": "Second notification sound.", "model": "kokoro", "voice": "af_heart"},
    {"text": "Third notification sound.", "model": "kokoro", "voice": "am_adam"},
]

result = client.batch_generate_and_wait(items_v2, timeout=120, interval=2.0)
print(f"Batch complete: {result.completed}/{result.total}")
for item in result.items:
    print(f"  Item {item.index}: {item.url}")

# --- Batch with webhook ---
print("\n--- Batch with webhook ---")
result = client.batch_generate(
    items=[
        {"text": "Webhook notification test.", "model": "kokoro", "voice": "af_bella"},
    ],
    webhook_url="https://yoursite.com/api/webhook/tts-batch",
)
print(f"Batch {result.batch_id} submitted with webhook")
print("You will receive a POST to your webhook URL when complete.")

client.close()
