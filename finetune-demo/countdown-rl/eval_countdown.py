"""
Base vs. base+GRPO-adapter accuracy comparison on a held-out slice of
Countdown-Tasks-3to4 (indices NOT used in training).

Usage (Colab cell):
    !python eval_countdown.py --adapter ./countdown-grpo-adapter --n 100
"""
import argparse
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from colab_train_countdown_grpo import BASE_MODEL, build_prompt
from reward_fn import score_completion

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTER = os.path.join(HERE, "countdown-grpo-adapter")


def load_holdout(n, train_examples=3000, seed=42):
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    ds = ds.shuffle(seed=seed)
    # training used indices [0, train_examples); eval uses the next n,
    # which were never seen during training
    return ds.select(range(train_examples, train_examples + n))


def evaluate(model, tokenizer, examples, device, label, max_new_tokens=300):
    model.eval()
    scores = []
    format_ok = 0
    numbers_ok = 0
    solved = 0
    for ex in examples:
        prompt = build_prompt(ex["nums"], ex["target"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # prompt already ends with "<think>", prepend it back for scoring since
        # the regex looks for <answer>...</answer> which the model must close itself
        full_text = "<think>" + completion
        score = score_completion(full_text, ex["target"], ex["nums"])
        scores.append(score)
        if score >= 0.1:
            format_ok += 1
        if score >= 0.2:
            numbers_ok += 1
        if score >= 1.0:
            solved += 1

    n = len(examples)
    avg = sum(scores) / n
    print(f"\n=== {label} (n={n}) ===")
    print(f"Solved (reward=1.0):        {solved}/{n} ({100*solved/n:.1f}%)")
    print(f"Right numbers, wrong target: {numbers_ok}/{n} ({100*numbers_ok/n:.1f}%) [includes solved]")
    print(f"Valid <answer> format:       {format_ok}/{n} ({100*format_ok/n:.1f}%) [includes above]")
    print(f"Mean reward:                 {avg:.4f}")
    return {"solved": solved, "numbers_ok": numbers_ok, "format_ok": format_ok, "n": n, "mean_reward": avg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    examples = load_holdout(args.n)
    print(f"Loaded {len(examples)} held-out examples (not used in training).")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (untrained)...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
    base_results = evaluate(base_model, tokenizer, examples, device, "BASE (no training)")
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("Loading base + GRPO adapter...")
    trained_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
    trained_model = PeftModel.from_pretrained(trained_model, args.adapter)
    trained_model = trained_model.merge_and_unload()
    trained_results = evaluate(trained_model, tokenizer, examples, device, "BASE + GRPO (150 steps)")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'':25s} {'BASE':>10s} {'+GRPO':>10s}")
    print(f"{'Solved':25s} {base_results['solved']:>9d}/{base_results['n']} {trained_results['solved']:>9d}/{trained_results['n']}")
    print(f"{'Mean reward':25s} {base_results['mean_reward']:>10.4f} {trained_results['mean_reward']:>10.4f}")


if __name__ == "__main__":
    main()
