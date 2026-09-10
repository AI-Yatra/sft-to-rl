"""
Evaluate the base model and the LoRA-adapted model on both eval sets, and
print a before/after comparison. Run this AFTER train_lora.py has produced
./lora-adapter.

Usage:
    .venv/bin/python eval_before_after.py
"""
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from field_scorer import evaluate_jsonl, print_eval_summary

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(HERE, "lora-adapter")
EVAL_SEEN = os.path.join(HERE, "data", "eval_seen_phrasing.jsonl")
EVAL_HOLDOUT = os.path.join(HERE, "data", "eval_holdout_phrasing.jsonl")


def make_pipe(with_adapter):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    if with_adapter:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        model = model.merge_and_unload()
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")


if __name__ == "__main__":
    print("Loading BASE model...")
    base_pipe = make_pipe(with_adapter=False)
    before_seen = evaluate_jsonl(base_pipe, EVAL_SEEN)
    before_holdout = evaluate_jsonl(base_pipe, EVAL_HOLDOUT)
    del base_pipe

    print("\nLoading LoRA-adapted model...")
    tuned_pipe = make_pipe(with_adapter=True)
    after_seen = evaluate_jsonl(tuned_pipe, EVAL_SEEN)
    after_holdout = evaluate_jsonl(tuned_pipe, EVAL_HOLDOUT)

    print_eval_summary(before_seen, "BEFORE (base) — seen phrasing")
    print_eval_summary(after_seen, "AFTER (LoRA)  — seen phrasing")
    print_eval_summary(before_holdout, "BEFORE (base) — HELD-OUT phrasing (day_of_month, in_n_days)")
    print_eval_summary(after_holdout, "AFTER (LoRA)  — HELD-OUT phrasing (day_of_month, in_n_days)")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"Seen phrasing:     {before_seen['overall_field_accuracy']*100:.1f}% -> {after_seen['overall_field_accuracy']*100:.1f}%")
    print(f"Held-out phrasing: {before_holdout['overall_field_accuracy']*100:.1f}% -> {after_holdout['overall_field_accuracy']*100:.1f}%")
    print(f"Hallucinations (seen):     {before_seen['hallucinated_attendees']} -> {after_seen['hallucinated_attendees']}")
    print(f"Hallucinations (held-out): {before_holdout['hallucinated_attendees']} -> {after_holdout['hallucinated_attendees']}")
