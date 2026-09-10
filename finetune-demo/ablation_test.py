import json
import time
from transformers import pipeline

TEST_MESSAGES = [
    "let's grab coffee tues at 3, maybe zoom?",
    "Reminder: dentist appointment next Monday 9am at Dr. Patel's clinic",
    "team sync moved to thursday 2:30pm, conf room B, invite raj and priya",
    "quick call with the vendor tomorrow morning around 10, should take 30 min",
    "Can we do the product review on the 15th? afternoon works, maybe 4ish, online",
]

SYSTEM_PROMPT = (
    "Today's date is 2026-09-06 (Sunday). "
    "Extract the event details from the message into JSON with exactly these "
    "fields: title, date (as YYYY-MM-DD, resolved relative to today), time (24h HH:MM), "
    "location, attendees (list). Respond with JSON only, no extra text."
)

MODELS = {
    "baseline+date-in-prompt": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "finetuned+date-in-prompt": "pramodkoujalagi/SmolLM2-360M-Instruct-Text-2-JSON",
}

if __name__ == "__main__":
    for name, model_id in MODELS.items():
        print(f"\n{'='*70}\n{name}  ({model_id})\n{'='*70}")
        pipe = pipeline("text-generation", model=model_id, device="cpu")
        outputs = []
        for msg in TEST_MESSAGES:
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg},
            ]
            out = pipe(chat, max_new_tokens=150, do_sample=False)
            generated = out[0]["generated_text"][-1]["content"]
            outputs.append(generated)
            print(f"--- {msg}\n    {generated!r}")
        print(f"\nRAW_OUTPUTS_{name} = {outputs!r}")
