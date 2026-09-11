"""
Synthetic dataset generator for the text-to-calendar-JSON task.

Every example is built from known ground-truth values (event type, target date,
time, location, attendees) and then rendered into casual English *and* into the
gold JSON label from the same values — so labels are correct by construction,
no manual annotation needed, and the eval set can hold out phrasing patterns
the train set never uses (checks generalization, not memorization).

The date ablation showed both models fail relative-date arithmetic even when
told "today's date" — so every example's reference date is randomized and the
date phrasing intentionally spans the hard cases (weekday math, "tomorrow",
day-of-month) rather than just easy literal dates.
"""
import json
import random
from datetime import datetime, timedelta

random.seed(42)

WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

EVENTS = [
    ("coffee", "Coffee"),
    ("dentist appointment", "Dentist appointment"),
    ("team sync", "Team sync"),
    ("call with the vendor", "Call with vendor"),
    ("product review", "Product review"),
    ("1:1 with my manager", "1:1 with manager"),
    ("lunch", "Lunch"),
    ("interview", "Interview"),
    ("yoga class", "Yoga class"),
    ("project kickoff", "Project kickoff"),
    ("standup", "Standup"),
    ("client demo", "Client demo"),
]

LOCATIONS = [
    ("zoom", "Zoom"), ("google meet", "Google Meet"), ("the office", "the office"),
    ("conf room A", "Conf Room A"), ("conf room B", "Conf Room B"),
    ("the cafe downtown", "the cafe downtown"), ("online", "online"),
    ("teams", "Teams"), (None, None), (None, None),  # weighted toward "no location mentioned"
]

PEOPLE = ["raj", "priya", "alex", "maria", "dev", "sara", "tom", "lena", "omar", "nina"]

TRAIN_DATE_PHRASINGS = ["tomorrow", "next_weekday", "this_weekday", "day_after_tomorrow"]
EVAL_ONLY_DATE_PHRASINGS = ["day_of_month", "in_n_days"]  # held out from train, tests generalization


def resolve_weekday_date(ref, weekday_idx, strictly_next_week):
    days_ahead = (weekday_idx - ref.weekday()) % 7
    if strictly_next_week:
        days_ahead = days_ahead + 7 if days_ahead == 0 else days_ahead + 7 if days_ahead < 7 and False else days_ahead
        days_ahead = days_ahead or 7
        if days_ahead < 7:
            pass
    else:
        days_ahead = days_ahead or 0
    return ref + timedelta(days=days_ahead)


def make_date(ref, kind):
    if kind == "tomorrow":
        d = ref + timedelta(days=1)
        return d, "tomorrow"
    if kind == "day_after_tomorrow":
        d = ref + timedelta(days=2)
        return d, "the day after tomorrow"
    if kind == "next_weekday":
        wd = random.randrange(7)
        days_ahead = (wd - ref.weekday()) % 7
        days_ahead = days_ahead + 7 if days_ahead < 7 else days_ahead  # always push to "next" occurrence, min 1 day
        days_ahead = days_ahead or 7
        d = ref + timedelta(days=days_ahead)
        return d, f"next {WEEKDAY_NAMES[wd]}"
    if kind == "this_weekday":
        wd = random.randrange(7)
        days_ahead = (wd - ref.weekday()) % 7
        days_ahead = days_ahead or 7
        d = ref + timedelta(days=days_ahead)
        return d, WEEKDAY_NAMES[wd]
    if kind == "day_of_month":
        day = random.randint(1, 28)
        month = ref.month if day >= ref.day else (ref.month % 12) + 1
        year = ref.year if month >= ref.month else ref.year + 1
        d = datetime(year, month, day)
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return d, f"the {day}{suffix}"
    if kind == "in_n_days":
        n = random.randint(2, 6)
        d = ref + timedelta(days=n)
        return d, f"in {n} days"
    raise ValueError(kind)


def make_time():
    hour24 = random.choice([9, 10, 11, 13, 14, 15, 16, 17, 18])
    minute = random.choice([0, 0, 0, 15, 30, 45])
    if hour24 == 12:
        h12, ampm = 12, "pm"
    elif hour24 > 12:
        h12, ampm = hour24 - 12, "pm"
    else:
        h12, ampm = hour24, "am"
    gold_time = f"{hour24:02d}:{minute:02d}"
    if minute == 0:
        phrase = f"{h12}{ampm}"
    else:
        phrase = f"{h12}:{minute:02d}{ampm}"
    return gold_time, phrase


def make_attendees():
    n = random.choices([0, 1, 2], weights=[0.4, 0.35, 0.25])[0]
    chosen = random.sample(PEOPLE, n)
    if n == 0:
        return [], ""
    if n == 1:
        return chosen, f", with {chosen[0]}"
    return chosen, f", invite {chosen[0]} and {chosen[1]}"


def make_example(ref, date_kind):
    event_phrase, title = random.choice(EVENTS)
    loc_phrase, loc_gold = random.choice(LOCATIONS)
    date_obj, date_phrase = make_date(ref, date_kind)
    gold_time, time_phrase = make_time()
    attendees, attendee_phrase = make_attendees()

    loc_clause = f" at {loc_phrase}" if loc_phrase else ""
    text = f"{event_phrase} {date_phrase} at {time_phrase}{loc_clause}{attendee_phrase}"
    text = text[0].upper() + text[1:] + "?"

    gold = {
        "title": title,
        "date": date_obj.strftime("%Y-%m-%d"),
        "time": gold_time,
        "location": loc_gold,
        "attendees": attendees,
    }
    return ref, text, gold


def random_reference_date():
    start = datetime(2026, 1, 1)
    return start + timedelta(days=random.randint(0, 364))


SYSTEM_TEMPLATE = (
    "Today's date is {today} ({weekday}). "
    "Extract the event details from the message into JSON with exactly these "
    "fields: title, date (YYYY-MM-DD, resolved relative to today), time (24h HH:MM), "
    "location (or null if not mentioned), attendees (list, empty if none named). "
    "Respond with JSON only, no extra text."
)


def build_records(n, date_kinds):
    records = []
    for _ in range(n):
        ref = random_reference_date()
        kind = random.choice(date_kinds)
        ref, text, gold = make_example(ref, kind)
        system = SYSTEM_TEMPLATE.format(today=ref.strftime("%Y-%m-%d"), weekday=WEEKDAY_NAMES[ref.weekday()].capitalize())
        assistant = json.dumps(gold, ensure_ascii=False)
        records.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
                {"role": "assistant", "content": assistant},
            ],
            "gold": gold,
            "date_kind": kind,
        })
    return records


if __name__ == "__main__":
    train = build_records(240, TRAIN_DATE_PHRASINGS)
    eval_seen_phrasing = build_records(30, TRAIN_DATE_PHRASINGS)
    eval_holdout_phrasing = build_records(30, EVAL_ONLY_DATE_PHRASINGS)

    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{out_dir}/eval_seen_phrasing.jsonl", "w") as f:
        for r in eval_seen_phrasing:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{out_dir}/eval_holdout_phrasing.jsonl", "w") as f:
        for r in eval_holdout_phrasing:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train: {len(train)} examples -> train.jsonl")
    print(f"eval (seen phrasing types): {len(eval_seen_phrasing)} -> eval_seen_phrasing.jsonl")
    print(f"eval (held-out phrasing types: {EVAL_ONLY_DATE_PHRASINGS}): {len(eval_holdout_phrasing)} -> eval_holdout_phrasing.jsonl")

    print("\n--- sample train record ---")
    print(json.dumps(train[0], indent=2))
    print("\n--- sample eval (holdout phrasing) record ---")
    print(json.dumps(eval_holdout_phrasing[0], indent=2))

    from collections import Counter
    print("\ndate_kind distribution (train):", Counter(r["date_kind"] for r in train))
    print("attendee count distribution (train):", Counter(len(r["gold"]["attendees"]) for r in train))
    print("location-null rate (train):", sum(1 for r in train if r["gold"]["location"] is None) / len(train))
