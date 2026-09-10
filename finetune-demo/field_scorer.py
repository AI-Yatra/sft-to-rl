"""
Field-level scorer for the text-to-calendar-JSON task.

Why this exists: json.loads() succeeding tells you nothing here — both the
base and fine-tuned SmolLM2-360M already produce syntactically valid JSON
on this task. The real differentiator is whether the *fields* are right:
correct schema keys, correct date/time, no hallucinated attendees. This
scorer measures that, so LoRA/GRPO improvements later actually show up
as a number instead of being invisible behind a 100%/100% JSON-validity tie.
"""
import json
import re
from datetime import datetime, timedelta

REFERENCE_DATE = datetime(2026, 9, 6)  # the "today" all relative dates below are computed against

def next_weekday(ref, weekday_target):
    days_ahead = (weekday_target - ref.weekday()) % 7
    days_ahead = days_ahead or 7
    return ref + timedelta(days=days_ahead)

WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}

GOLD = [
    {
        "input": "let's grab coffee tues at 3, maybe zoom?",
        "title_keywords": ["coffee"],
        "date": next_weekday(REFERENCE_DATE, WEEKDAYS["tuesday"]).strftime("%Y-%m-%d"),
        "time": "15:00",
        "location_keywords": ["zoom"],
        "attendees": [],
    },
    {
        "input": "Reminder: dentist appointment next Monday 9am at Dr. Patel's clinic",
        "title_keywords": ["dentist"],
        "date": next_weekday(REFERENCE_DATE, WEEKDAYS["monday"]).strftime("%Y-%m-%d"),
        "time": "09:00",
        "location_keywords": ["patel", "clinic"],
        "attendees": ["dr. patel", "patel"],
    },
    {
        "input": "team sync moved to thursday 2:30pm, conf room B, invite raj and priya",
        "title_keywords": ["team", "sync"],
        "date": next_weekday(REFERENCE_DATE, WEEKDAYS["thursday"]).strftime("%Y-%m-%d"),
        "time": "14:30",
        "location_keywords": ["conf room b", "room b"],
        "attendees": ["raj", "priya"],
    },
    {
        "input": "quick call with the vendor tomorrow morning around 10, should take 30 min",
        "title_keywords": ["call", "vendor"],
        "date": (REFERENCE_DATE + timedelta(days=1)).strftime("%Y-%m-%d"),
        "time": "10:00",
        "location_keywords": [],
        "attendees": ["vendor"],
    },
    {
        "input": "Can we do the product review on the 15th? afternoon works, maybe 4ish, online",
        "title_keywords": ["product", "review"],
        "date": REFERENCE_DATE.replace(day=15).strftime("%Y-%m-%d"),
        "time": "16:00",
        "location_keywords": ["online"],
        "attendees": [],
    },
]

CANONICAL_KEYS = ["title", "date", "time", "location", "attendees"]


def flatten_fields(obj, found=None):
    """Recursively hunt for canonical keys anywhere in the (possibly mis-nested) JSON."""
    if found is None:
        found = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            for canon in CANONICAL_KEYS:
                if canon in kl and canon not in found:
                    found[canon] = v
            if isinstance(v, (dict, list)):
                flatten_fields(v, found)
    elif isinstance(obj, list):
        for item in obj:
            flatten_fields(item, found)
    return found


def parse_json_loose(text):
    try:
        return json.loads(text.strip())
    except Exception:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return None


def normalize_time(val):
    if not val:
        return None
    val = str(val).strip().lower()
    m = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", val)
    if not m:
        return None
    hour, minute, ampm = m.groups()
    hour = int(hour)
    minute = int(minute) if minute else 0
    if ampm == "pm" and hour != 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def normalize_date(val):
    if not val:
        return None
    val = str(val).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(val, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def score_example(gold, raw_output, source_text):
    parsed = parse_json_loose(raw_output)
    result = {"json_valid": parsed is not None, "fields": {}, "hallucinated_attendees": []}
    if parsed is None:
        for k in CANONICAL_KEYS:
            result["fields"][k] = False
        return result

    fields = flatten_fields(parsed)

    title_val = str(fields.get("title", "")).lower()
    result["fields"]["title"] = any(kw in title_val for kw in gold["title_keywords"]) if gold["title_keywords"] else True

    pred_date = normalize_date(fields.get("date"))
    result["fields"]["date"] = pred_date == gold["date"]

    pred_time = normalize_time(fields.get("time"))
    result["fields"]["time"] = pred_time == gold["time"]

    loc_val = str(fields.get("location", "")).lower()
    result["fields"]["location"] = (
        any(kw in loc_val for kw in gold["location_keywords"]) if gold["location_keywords"] else True
    )

    pred_attendees = fields.get("attendees", [])
    if isinstance(pred_attendees, str):
        pred_attendees = [pred_attendees]
    pred_attendees = [str(a).lower() for a in (pred_attendees or [])]

    gold_attendees = gold["attendees"]
    if gold_attendees:
        matched = sum(1 for g in gold_attendees if any(g in p or p in g for p in pred_attendees))
        result["fields"]["attendees"] = matched >= 1
    else:
        result["fields"]["attendees"] = len(pred_attendees) == 0

    source_lower = source_text.lower()
    for name in pred_attendees:
        if name and name not in source_lower and name not in ("team", "client", "attendees"):
            result["hallucinated_attendees"].append(name)

    return result


def score_against_exact_gold(gold, raw_output, source_text):
    """
    Scorer for the generate_dataset.py gold format (exact values, not keyword
    lists) — used to evaluate the synthetic train/eval JSONL files, as opposed
    to score_example() above which handles the 5 hand-written manual examples.
    """
    parsed = parse_json_loose(raw_output)
    result = {"json_valid": parsed is not None, "fields": {}, "hallucinated_attendees": []}
    if parsed is None:
        for k in CANONICAL_KEYS:
            result["fields"][k] = False
        return result

    fields = flatten_fields(parsed)

    title_val = str(fields.get("title", "")).lower().strip()
    gold_title = str(gold["title"]).lower().strip()
    result["fields"]["title"] = title_val == gold_title or gold_title in title_val or title_val in gold_title

    pred_date = normalize_date(fields.get("date"))
    result["fields"]["date"] = pred_date == gold["date"]

    pred_time = normalize_time(fields.get("time"))
    result["fields"]["time"] = pred_time == gold["time"]

    loc_val = fields.get("location")
    gold_loc = gold["location"]
    if gold_loc is None:
        result["fields"]["location"] = loc_val in (None, "null", "", "none")
    else:
        loc_str = str(loc_val).lower().strip()
        result["fields"]["location"] = gold_loc.lower() in loc_str or loc_str in gold_loc.lower()

    pred_attendees = fields.get("attendees", [])
    if isinstance(pred_attendees, str):
        pred_attendees = [pred_attendees]
    pred_attendees = set(str(a).lower().strip() for a in (pred_attendees or []))
    gold_attendees = set(a.lower().strip() for a in gold["attendees"])
    result["fields"]["attendees"] = pred_attendees == gold_attendees

    source_lower = source_text.lower()
    for name in pred_attendees:
        if name and name not in source_lower:
            result["hallucinated_attendees"].append(name)

    return result


def evaluate_jsonl(pipe, jsonl_path, verbose=False, limit=None):
    """Run a transformers pipeline over a synthetic eval JSONL file and score it."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]

    field_totals = {k: 0 for k in CANONICAL_KEYS}
    total_hallucinations = 0
    json_valid_count = 0

    for r in records:
        system_msg, user_msg = r["messages"][0], r["messages"][1]
        out = pipe([system_msg, user_msg], max_new_tokens=150, do_sample=False)
        generated = out[0]["generated_text"][-1]["content"]
        res = score_against_exact_gold(r["gold"], generated, user_msg["content"])
        json_valid_count += int(res["json_valid"])
        for k in CANONICAL_KEYS:
            field_totals[k] += int(res["fields"][k])
        total_hallucinations += len(res["hallucinated_attendees"])
        if verbose:
            flags = " ".join(f"{k}={'OK' if res['fields'][k] else 'X'}" for k in CANONICAL_KEYS)
            print(f"  {user_msg['content'][:50]:50s} {flags}")

    n = len(records)
    overall = sum(field_totals.values()) / (n * len(CANONICAL_KEYS))
    return {
        "n": n,
        "json_valid_rate": json_valid_count / n,
        "field_totals": field_totals,
        "overall_field_accuracy": overall,
        "hallucinated_attendees": total_hallucinations,
    }


def print_eval_summary(summary, label):
    print(f"\n{'='*70}\n{label}  (n={summary['n']})\n{'='*70}")
    print(f"  JSON valid rate: {summary['json_valid_rate']*100:.1f}%")
    for k in CANONICAL_KEYS:
        print(f"  {k:10s} {summary['field_totals'][k]}/{summary['n']}")
    print(f"  Overall field accuracy: {summary['overall_field_accuracy']*100:.1f}%")
    print(f"  Hallucinated attendee mentions: {summary['hallucinated_attendees']}")


def run_scorer(model_outputs, model_name):
    """model_outputs: list of raw generated text strings, same order as GOLD"""
    print(f"\n{'='*70}\nFIELD-LEVEL SCORE: {model_name}\n{'='*70}")
    field_totals = {k: 0 for k in CANONICAL_KEYS}
    total_hallucinations = 0
    for gold, raw in zip(GOLD, model_outputs):
        res = score_example(gold, raw, gold["input"])
        for k in CANONICAL_KEYS:
            field_totals[k] += int(res["fields"][k])
        total_hallucinations += len(res["hallucinated_attendees"])
        flags = " ".join(f"{k}={'OK' if res['fields'][k] else 'X'}" for k in CANONICAL_KEYS)
        halluc = f"  hallucinated={res['hallucinated_attendees']}" if res["hallucinated_attendees"] else ""
        print(f"  {gold['input'][:45]:45s} {flags}{halluc}")

    n = len(GOLD)
    print("\n  Field accuracy:")
    for k in CANONICAL_KEYS:
        print(f"    {k:10s} {field_totals[k]}/{n}")
    overall = sum(field_totals.values()) / (n * len(CANONICAL_KEYS))
    print(f"  Overall field accuracy: {overall*100:.1f}%")
    print(f"  Hallucinated attendee mentions: {total_hallucinations}")
    return overall, total_hallucinations


if __name__ == "__main__":
    # Outputs captured verbatim from the previous baseline_test.py run.
    baseline_outputs = [
        '{\n  "title": "Coffee at 3",\n  "date": "Tuesday at 3",\n  "time": "Maybe Zoom?",\n  "location": "Tues at 3",\n  "attendees": ["Alex", "Bob", "Charlie"]\n}',
        '{\n  "event_details": {\n    "title": "Dentist Appointment",\n    "date": "9am",\n    "time": "next Monday",\n    "location": "Dr. Patel\'s Clinic",\n    "attendees": ["Dr. Patel"]\n  }\n}',
        '{\n  "team_sync": {\n    "date": "2:30 PM",\n    "time": "conf_room B",\n    "location": "conf_room B",\n    "attendees": ["raj", "priya"]\n  }\n}',
        '{\n  "event_details": {\n    "title": "Vendor\'s Event",\n    "date": "2023-01-01",\n    "time": "10:00 AM",\n    "location": "Vendor\'s Office",\n    "attendees": ["Vendor\'s Attendees"]\n  }\n}',
        '{\n  "title": "Product Review",\n  "date": "15th",\n  "time": "afternoon",\n  "location": "online",\n  "attendees": [\n    {\n      "name": "John Doe",\n      "email": "john.doe@example.com"\n    }\n  ]\n}',
    ]
    finetuned_outputs = [
        '{"title": "Coffee", "date": "20/02/2024", "time": "2:00 PM", "location": "Zoom", "attendees": ["team"], "notes": "for planning"}',
        '{"title": "dentist appointment", "date": "09/01/2024", "time": "9:00 AM", "location": "Dr. Patel\'s clinic", "attendees": ["Dr. Patel", "client"], "notes": "for appointment"}',
        '{"title": "team sync", "date": "2/2/2024", "time": "10:30 AM", "location": "conf room B", "attendees": ["raj", "priya"], "action": "invite"}',
        '{"title": "Quick call", "date": "11/04/2024", "time": "10:00 AM", "location": "room 12", "attendees": ["vendor"], "duration": "30 minutes"}',
        '{"title": "Product review", "date": "15/12/2023", "time": "4:00 PM", "location": "online", "attendees": ["team"]}',
    ]

    run_scorer(baseline_outputs, "baseline (SmolLM2-360M-Instruct)")
    run_scorer(finetuned_outputs, "fine-tuned (Text-2-JSON)")
