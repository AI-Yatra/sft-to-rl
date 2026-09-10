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
    "Extract the event details from the message into JSON with exactly these "
    "fields: title, date, time, location, attendees (list). "
    "Respond with JSON only, no extra text."
)

MODELS = {
    "baseline (SmolLM2-360M-Instruct)": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "fine-tuned (Text-2-JSON)": "pramodkoujalagi/SmolLM2-360M-Instruct-Text-2-JSON",
}


def is_valid_json(text):
    try:
        json.loads(text.strip())
        return True
    except Exception:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            json.loads(text[start:end])
            return True
        except Exception:
            return False


def run(model_name, model_id):
    print(f"\n{'='*70}\n{model_name}  ({model_id})\n{'='*70}")
    t0 = time.time()
    pipe = pipeline("text-generation", model=model_id, device="cpu")
    print(f"[loaded in {time.time()-t0:.1f}s]")

    valid_count = 0
    for msg in TEST_MESSAGES:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg},
        ]
        t0 = time.time()
        out = pipe(chat, max_new_tokens=150, do_sample=False)
        dt = time.time() - t0
        generated = out[0]["generated_text"][-1]["content"]
        ok = is_valid_json(generated)
        valid_count += int(ok)
        print(f"\n--- input: {msg}")
        print(f"    valid_json={ok}  ({dt:.1f}s)")
        print(f"    output: {generated!r}")

    print(f"\n>>> {model_name}: {valid_count}/{len(TEST_MESSAGES)} valid JSON outputs")
    return valid_count


if __name__ == "__main__":
    results = {}
    for name, model_id in MODELS.items():
        try:
            results[name] = run(name, model_id)
        except Exception as e:
            print(f"\n!!! {name} failed to load/run: {e}")
            results[name] = None

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for name, score in results.items():
        print(f"{name}: {score}/{len(TEST_MESSAGES)} valid JSON")
