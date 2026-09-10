"""
Terminal showcase demo: base Qwen2.5-0.5B vs. base+GRPO adapter on the
Countdown task, side by side. No SFT was ever applied to this model --
whatever it does well here, it learned purely from a verifiable reward
during GRPO training (see RESULTS.md for the training run this used).

Usage (from this folder -- run `uv sync` once in ../ first, see ../README.md):
    uv run --project .. python showcase_demo.py                  # a few built-in examples
    uv run --project .. python showcase_demo.py --interactive     # type your own numbers/target
"""
import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from colab_train_countdown_grpo import BASE_MODEL, build_prompt
from reward_fn import score_completion

DEMO_EXAMPLES = [
    {"nums": [58, 80, 36], "target": 14},   # trained model genuinely SOLVES this
    {"nums": [3, 4, 2], "target": 14},      # trained model: right format, wrong result
    {"nums": [1, 2, 6, 8], "target": 24},   # trained model: wrong numbers used
    {"nums": [10, 3, 5], "target": 25},     # honest failure: both models loop/degenerate
]

VERDICT = {
    1.0: ("SOLVED", "\033[92m"),      # green
    0.2: ("wrong result", "\033[93m"),  # yellow
    0.1: ("wrong numbers", "\033[91m"),  # red
    0.0: ("invalid", "\033[91m"),        # red
}
RESET = "\033[0m"
BOLD = "\033[1m"


def load(adapter_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, nums, target, max_new_tokens=300):
    # Plain greedy decoding, matching eval_countdown.py exactly (the numbers
    # in RESULTS.md were produced this way) -- repetition_penalty/
    # no_repeat_ngram_size were tried here and made output WORSE (hallucinated
    # x/y/z variables never seen in training), so left out. stop_strings just
    # ends generation early once a well-formed answer closes; it doesn't
    # change what's generated before that point.
    prompt = build_prompt(nums, target)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            stop_strings=["</answer>"],
        )
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    full_text = "<think>" + completion
    score = score_completion(full_text, target, nums)
    return full_text.strip(), score


def show_result(label, text, score):
    verdict, color = VERDICT[score]
    print(f"  {BOLD}{label}{RESET} [{color}{verdict}{RESET}, reward={score}]")
    for line in text.splitlines():
        print(f"    {line}")
    print()


def run_example(base_model, base_tok, trained_model, trained_tok, nums, target):
    print(f"{BOLD}Numbers: {nums}   Target: {target}{RESET}")
    base_text, base_score = generate(base_model, base_tok, nums, target)
    show_result("BASE (no training)", base_text, base_score)
    trained_text, trained_score = generate(trained_model, trained_tok, nums, target)
    show_result("BASE + GRPO (150 steps, no SFT)", trained_text, trained_score)
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default="./countdown-grpo-adapter")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print("Loading base model...")
    base_model, base_tok = load(adapter_dir=None)
    print("Loading base + GRPO adapter...")
    trained_model, trained_tok = load(adapter_dir=args.adapter)
    print()

    if args.interactive:
        print("Enter numbers (comma-separated) and a target, or 'q' to quit.")
        while True:
            raw = input("\nnums (e.g. 3,4,2): ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                break
            try:
                nums = [int(x) for x in raw.split(",")]
                target = int(input("target: ").strip())
            except ValueError:
                print("Couldn't parse that, try again.")
                continue
            run_example(base_model, base_tok, trained_model, trained_tok, nums, target)
    else:
        for ex in DEMO_EXAMPLES:
            run_example(base_model, base_tok, trained_model, trained_tok, ex["nums"], ex["target"])


if __name__ == "__main__":
    main()
