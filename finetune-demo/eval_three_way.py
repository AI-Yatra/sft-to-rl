"""
Three-way comparison on the field scorer: base model vs. +SFT vs. +SFT+DPO.

This evaluates on the ORIGINAL task eval sets (data/eval_seen_phrasing.jsonl,
data/eval_holdout_phrasing.jsonl) — not the DPO preference pairs — because
what we actually care about is whether DPO improved real task field accuracy,
not just preference win-rate on its own training format.

Runs each stage as a SEPARATE process (via --stage), not in one loop — this
laptop is memory-constrained enough (other apps + 3 models sequentially) that
an in-process loop got OOM-killed even with del/gc between stages. A fresh
process per stage guarantees the OS fully reclaims memory in between.

Usage:
    .venv/bin/python eval_three_way.py --stage base
    .venv/bin/python eval_three_way.py --stage sft
    .venv/bin/python eval_three_way.py --stage sft_dpo
    .venv/bin/python eval_three_way.py --combine   # after all three have run
"""
import argparse
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
EVAL_SEEN = os.path.join(HERE, "sft", "data", "eval_seen_phrasing.jsonl")
EVAL_HOLDOUT = os.path.join(HERE, "sft", "data", "eval_holdout_phrasing.jsonl")


def make_pipe(stage):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    if stage in ("sft", "sft_dpo"):
        model = PeftModel.from_pretrained(model, SFT_ADAPTER)
        model = model.merge_and_unload()
    if stage == "sft_dpo":
        model = PeftModel.from_pretrained(model, DPO_ADAPTER)
        model = model.merge_and_unload()
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")


RESULTS_DIR = os.path.join(HERE, "eval_results")
STAGES = [("base", "BASE"), ("sft", "+SFT"), ("sft_dpo", "+SFT+DPO")]


def run_single_stage(stage):
    print(f"\nLoading stage: {stage}...")
    pipe = make_pipe(stage)
    seen = evaluate_jsonl(pipe, EVAL_SEEN)
    holdout = evaluate_jsonl(pipe, EVAL_HOLDOUT)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{stage}.json"), "w") as f:
        json.dump({"seen": seen, "holdout": holdout}, f, indent=2)
    print(f"Saved eval_results/{stage}.json")


def combine():
    results = {}
    for stage, _ in STAGES:
        path = os.path.join(RESULTS_DIR, f"{stage}.json")
        with open(path) as f:
            data = json.load(f)
        results[(stage, "seen")] = data["seen"]
        results[(stage, "holdout")] = data["holdout"]

    for stage, label in STAGES:
        print_eval_summary(results[(stage, "seen")], f"{label} — seen phrasing")
        print_eval_summary(results[(stage, "holdout")], f"{label} — held-out phrasing")

    print(f"\n{'='*70}\nTHREE-WAY SUMMARY\n{'='*70}")
    print(f"{'stage':10s} {'seen acc':>10s} {'holdout acc':>12s} {'halluc(seen)':>13s} {'halluc(holdout)':>16s}")
    for stage, label in STAGES:
        s, h = results[(stage, "seen")], results[(stage, "holdout")]
        print(f"{label:10s} {s['overall_field_accuracy']*100:9.1f}% {h['overall_field_accuracy']*100:11.1f}% "
              f"{s['hallucinated_attendees']:13d} {h['hallucinated_attendees']:16d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["base", "sft", "sft_dpo"])
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()

    if args.combine:
        combine()
    elif args.stage:
        run_single_stage(args.stage)
    else:
        parser.error("pass --stage base|sft|sft_dpo, or --combine once all three are done")
