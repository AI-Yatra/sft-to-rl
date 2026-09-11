"""
Verifiable reward function for GRPO on the text-to-calendar-JSON task.

"Verifiable" because the reward is computed directly from gold JSON we
generated the prompt from (generate_grpo_dataset.py) — no separate reward
model needed, same pattern as the Hugging Face IFStruct/GRPO blog post this
whole exercise was modeled on.

Reward = fraction of the 5 fields (title, date, time, location, attendees)
that match gold, using the same matching logic as field_scorer.py's
score_against_exact_gold — so "reward went up" and "field accuracy went up"
are the same measurement, not two different metrics that could diverge.

Reward is 0.0 outright for invalid JSON (can't game the reward with garbage
that happens to please a fuzzy string match), and hallucinated attendees
(names in the output not present in the source text) subtract a small
penalty on top of the field score — this is deliberately checking the
"reward hacking" scenario: a model that always says `"attendees": []` would
score fine on JSON validity but should not be able to farm reward by never
naming anyone when the source text does.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from field_scorer import CANONICAL_KEYS, flatten_fields, normalize_date, normalize_time, parse_json_loose


def _score_one(gold, raw_output, source_text):
    parsed = parse_json_loose(raw_output)
    if parsed is None:
        return 0.0

    fields = flatten_fields(parsed)
    hits = 0

    title_val = str(fields.get("title", "")).lower().strip()
    gold_title = str(gold["title"]).lower().strip()
    if title_val == gold_title or gold_title in title_val or title_val in gold_title:
        hits += 1

    if normalize_date(fields.get("date")) == gold["date"]:
        hits += 1

    if normalize_time(fields.get("time")) == gold["time"]:
        hits += 1

    loc_val = fields.get("location")
    gold_loc = gold["location"]
    if gold_loc is None:
        if loc_val in (None, "null", "", "none"):
            hits += 1
    else:
        loc_str = str(loc_val).lower().strip()
        if gold_loc.lower() in loc_str or loc_str in gold_loc.lower():
            hits += 1

    pred_attendees = fields.get("attendees", [])
    if isinstance(pred_attendees, str):
        pred_attendees = [pred_attendees]
    pred_attendees = set(str(a).lower().strip() for a in (pred_attendees or []))
    gold_attendees = set(a.lower().strip() for a in gold["attendees"])
    if pred_attendees == gold_attendees:
        hits += 1

    field_score = hits / len(CANONICAL_KEYS)

    source_lower = source_text.lower()
    hallucinated = sum(1 for name in pred_attendees if name and name not in source_lower)
    penalty = 0.1 * hallucinated

    return max(0.0, field_score - penalty)


def _extract_text(completion):
    """completions from GRPOTrainer are either raw strings or [{"role": "assistant", "content": ...}]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return str(completion)


def calendar_json_reward(completions, gold_json=None, source_text=None, **kwargs):
    """
    GRPOTrainer reward function signature: receives `completions` plus any
    extra dataset columns as keyword lists (here: gold_json, source_text from
    generate_grpo_dataset.py). Must return a list[float], one per completion.
    """
    rewards = []
    for i, completion in enumerate(completions):
        gold = json.loads(gold_json[i])
        text = _extract_text(completion)
        src = source_text[i]
        rewards.append(_score_one(gold, text, src))
    return rewards


if __name__ == "__main__":
    # quick self-test
    gold = {"title": "Coffee", "date": "2026-09-08", "time": "15:00", "location": "Zoom", "attendees": []}
    good = json.dumps(gold)
    bad_hallucinated = json.dumps({**gold, "attendees": ["Bob", "Alice"]})
    bad_json = "not json at all"

    r = calendar_json_reward(
        completions=[good, bad_hallucinated, bad_json],
        gold_json=[json.dumps(gold)] * 3,
        source_text=["let's grab coffee tues at 3, zoom?"] * 3,
    )
    print("rewards:", r)
    assert r[0] == 1.0
    assert 0.0 < r[1] < 1.0
    assert r[2] == 0.0
    print("self-test passed")
