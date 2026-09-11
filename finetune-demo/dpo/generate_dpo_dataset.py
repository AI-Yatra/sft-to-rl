"""
Synthetic DPO preference-pair dataset for the text-to-calendar-JSON task.

Separate from generate_dataset.py (SFT) on purpose — different schema
("prompt"/"chosen"/"rejected" for DPOTrainer vs "messages" for SFT) and a
different job: SFT teaches the correct output, DPO teaches "this is better
than that". Reuses the underlying event/date/time generators from
generate_dataset.py so gold values stay consistent between the two stages.

"chosen" is always the exact gold JSON. "rejected" is the same JSON with one
targeted corruption applied — chosen to mirror the failure modes the SFT run
actually produced (RESULTS.md): wrong date, hallucinated attendee, dropped
attendee, wrong location/time, or a mis-nested schema (the baseline model's
signature failure).
"""
import copy
import json
import os
import random
import sys
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sft"))
from generate_dataset import (
    EVENTS, LOCATIONS, PEOPLE, TRAIN_DATE_PHRASINGS, SYSTEM_TEMPLATE, WEEKDAY_NAMES,
    make_example, random_reference_date,
)

random.seed(7)

CORRUPTIONS = ["wrong_date", "wrong_time", "hallucinate_attendee", "drop_attendee", "wrong_location", "mis_nested"]


def corrupt(gold):
    rejected = copy.deepcopy(gold)
    kind = random.choice(CORRUPTIONS)

    if kind == "wrong_date":
        from datetime import datetime
        d = datetime.strptime(gold["date"], "%Y-%m-%d")
        shift = random.choice([-3, -2, -1, 1, 2, 3, 7])
        rejected["date"] = (d + timedelta(days=shift)).strftime("%Y-%m-%d")

    elif kind == "wrong_time":
        hh, mm = map(int, gold["time"].split(":"))
        shift = random.choice([-2, -1, 1, 2])
        new_hh = (hh + shift) % 24
        rejected["time"] = f"{new_hh:02d}:{mm:02d}"

    elif kind == "hallucinate_attendee":
        candidates = [p for p in PEOPLE if p not in gold["attendees"]]
        rejected["attendees"] = gold["attendees"] + [random.choice(candidates)]

    elif kind == "drop_attendee":
        if gold["attendees"]:
            rejected["attendees"] = gold["attendees"][:-1]
        else:
            rejected["attendees"] = [random.choice(PEOPLE)]  # invents one where none existed

    elif kind == "wrong_location":
        candidates = [loc for _, loc in LOCATIONS if loc and loc != gold["location"]]
        rejected["location"] = random.choice(candidates)

    elif kind == "mis_nested":
        # mirrors the baseline model's actual failure mode from baseline_test.py:
        # nesting the real fields under an arbitrary wrapper key instead of top-level.
        wrapper_key = gold["title"].lower().replace(" ", "_")
        return kind, {wrapper_key: rejected}

    return kind, rejected


def build_dpo_records(n, date_kinds=TRAIN_DATE_PHRASINGS):
    records = []
    for _ in range(n):
        ref = random_reference_date()
        kind_date = random.choice(date_kinds)
        ref, text, gold = make_example(ref, kind_date)
        system = SYSTEM_TEMPLATE.format(today=ref.strftime("%Y-%m-%d"), weekday=WEEKDAY_NAMES[ref.weekday()].capitalize())

        corruption_kind, rejected_obj = corrupt(gold)

        chosen_str = json.dumps(gold, ensure_ascii=False)
        rejected_str = json.dumps(rejected_obj, ensure_ascii=False)

        records.append({
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "chosen": [{"role": "assistant", "content": chosen_str}],
            "rejected": [{"role": "assistant", "content": rejected_str}],
            "corruption_kind": corruption_kind,
            "gold": gold,
        })
    return records


if __name__ == "__main__":
    import os
    from collections import Counter

    train = build_dpo_records(150)
    eval_set = build_dpo_records(30)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    with open(f"{out_dir}/dpo_train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{out_dir}/dpo_eval.jsonl", "w") as f:
        for r in eval_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"dpo_train.jsonl: {len(train)} preference pairs")
    print(f"dpo_eval.jsonl: {len(eval_set)} preference pairs")
    print("\ncorruption_kind distribution (train):", Counter(r["corruption_kind"] for r in train))
    print("\n--- sample record ---")
    print(json.dumps(train[0], indent=2))
