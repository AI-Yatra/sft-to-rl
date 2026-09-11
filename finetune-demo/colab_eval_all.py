"""
Four-way field-accuracy comparison, for Colab (GPU, no memory constraints —
unlike the laptop, this can run all four stages in one process).

Stages: base -> +SFT -> +SFT+DPO -> +SFT+DPO+GRPO

Usage (after train_lora.py/colab_train_dpo.py/colab_train_grpo.py have all
produced their adapters, all present in this folder):
    !python colab_eval_all.py
"""
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from field_scorer import evaluate_jsonl, print_eval_summary

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
SFT_ADAPTER = os.path.join(HERE, "sft", "lora-adapter")
DPO_ADAPTER = os.path.join(HERE, "dpo", "dpo-adapter")
GRPO_ADAPTER = os.path.join(HERE, "grpo", "grpo-adapter")
EVAL_SEEN = os.path.join(HERE, "sft", "data", "eval_seen_phrasing.jsonl")
EVAL_HOLDOUT = os.path.join(HERE, "sft", "data", "eval_holdout_phrasing.jsonl")

STAGES = [
    ("base", "BASE", []),
    ("sft", "+SFT", [SFT_ADAPTER]),
    ("sft_dpo", "+SFT+DPO", [SFT_ADAPTER, DPO_ADAPTER]),
    ("sft_dpo_grpo", "+SFT+DPO+GRPO", [SFT_ADAPTER, DPO_ADAPTER, GRPO_ADAPTER]),
]


def make_pipe(adapters, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
    for adapter_dir in adapters:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    results = {}
    for stage, label, adapters in STAGES:
        print(f"\nLoading stage: {label} ({len(adapters)} adapter(s))...")
        pipe = make_pipe(adapters, device, dtype)
        results[(stage, "seen")] = evaluate_jsonl(pipe, EVAL_SEEN)
        results[(stage, "holdout")] = evaluate_jsonl(pipe, EVAL_HOLDOUT)
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()

    for stage, label, _ in STAGES:
        print_eval_summary(results[(stage, "seen")], f"{label} — seen phrasing")
        print_eval_summary(results[(stage, "holdout")], f"{label} — held-out phrasing")

    print(f"\n{'='*70}\nFOUR-WAY SUMMARY\n{'='*70}")
    print(f"{'stage':16s} {'seen acc':>10s} {'holdout acc':>12s} {'halluc(seen)':>13s} {'halluc(holdout)':>16s}")
    for stage, label, _ in STAGES:
        s, h = results[(stage, "seen")], results[(stage, "holdout")]
        print(f"{label:16s} {s['overall_field_accuracy']*100:9.1f}% {h['overall_field_accuracy']*100:11.1f}% "
              f"{s['hallucinated_attendees']:13d} {h['hallucinated_attendees']:16d}")

    out_dir = os.path.join(HERE, "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "four_way_results.json"), "w") as f:
        json.dump({f"{stage}_{split}": results[(stage, split)] for stage, _, _ in STAGES for split in ("seen", "holdout")}, f, indent=2)
    print(f"\nSaved eval_results/four_way_results.json — download this before the session ends.")
