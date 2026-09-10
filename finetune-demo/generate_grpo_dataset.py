"""
Synthetic GRPO dataset for the text-to-calendar-JSON task.

Unlike SFT (data/train.jsonl) and DPO (data/dpo_train.jsonl), GRPO doesn't
need a gold *completion* in the dataset at all — it needs a prompt and a way
to score whatever the model generates. Each record here carries the prompt
plus the gold JSON (as a string, for reward scoring) and the raw source text
(for hallucination checks) — reward_fn.py turns those into a 0-1 reward.

Reuses the same underlying event/date/time generators as generate_dataset.py
so all three stages (SFT/DPO/GRPO) are grounded in the same gold logic.
"""
import json
import os
import random

from generate_dataset import SYSTEM_TEMPLATE, TRAIN_DATE_PHRASINGS, WEEKDAY_NAMES, make_example, random_reference_date

random.seed(99)


def build_grpo_records(n, date_kinds=TRAIN_DATE_PHRASINGS):
    records = []
    for _ in range(n):
        ref = random_reference_date()
        kind = random.choice(date_kinds)
        ref, text, gold = make_example(ref, kind)
        system = SYSTEM_TEMPLATE.format(today=ref.strftime("%Y-%m-%d"), weekday=WEEKDAY_NAMES[ref.weekday()].capitalize())
        records.append({
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "gold_json": json.dumps(gold, ensure_ascii=False),
            "source_text": text,
        })
    return records


if __name__ == "__main__":
    train = build_grpo_records(200)
    eval_set = build_grpo_records(30)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    with open(f"{out_dir}/grpo_train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{out_dir}/grpo_eval.jsonl", "w") as f:
        for r in eval_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"grpo_train.jsonl: {len(train)} prompts")
    print(f"grpo_eval.jsonl: {len(eval_set)} prompts")
    print("\n--- sample record ---")
    print(json.dumps(train[0], indent=2))
