"""
Pure-RL Countdown demo (TinyZero / Jiayi-Pan style) -- GRPO from a BASE
model, no SFT at all. Run in Colab (T4/L4 GPU).

Model:   Qwen/Qwen2.5-0.5B
         (the post named Qwen3.5-0.8B-Base, but its config shows
         architectures=["Qwen3_5ForConditionalGeneration"] -- a multimodal,
         hybrid linear-attention model, not a plain causal LM. Qwen2.5-0.5B
         is a standard decoder-only base model (Qwen2.5's un-suffixed
         checkpoints ARE the base/pretrain checkpoints -- "-Base" is Qwen3-era
         naming and doesn't exist for Qwen2.5), well-supported by peft/trl,
         and the actual model family used in the original TinyZero/Countdown
         reproductions this idea is based on -- swapped for reliability.)
Dataset: Jiayi-Pan/Countdown-Tasks-3to4 (target int, nums list)
Reward:  reward_fn.countdown_reward -- did the completion produce a valid
         equation using each given number exactly once that evaluates to
         the target? (see reward_fn.py for the exact scoring breakdown)

Usage (Colab cell):
    !pip install -q transformers accelerate peft trl datasets
    !python colab_train_countdown_grpo.py --steps 300 --output ./countdown-grpo-adapter
"""
import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward_fn import countdown_reward

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# Plain-string prompt, not chat-format: Qwen2.5-0.5B-Base is a base model
# and may not ship a usable chat template. This is also the standard
# TinyZero-style prompt shape (ends mid-<think> to bias the base model
# straight into reasoning, no instruction-following needed).
PROMPT_TEMPLATE = (
    "Using the numbers {nums_str}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number "
    "can only be used once. Show your work in <think> </think> tags, then "
    "return the final equation in <answer> </answer> tags, e.g. "
    "<answer>(1 + 2) * 3</answer>.\n<think>"
)


def build_prompt(nums, target):
    nums_str = ", ".join(str(n) for n in nums)
    return PROMPT_TEMPLATE.format(nums_str=nums_str, target=target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-examples", type=int, default=2000)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--output", type=str, default="./countdown-grpo-adapter")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected -- this will be extremely slow.")
    print(f"Device: {device}")

    print("Loading dataset...")
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    ds = ds.shuffle(seed=42).select(range(args.num_examples))

    def to_prompt(ex):
        return {
            "prompt": build_prompt(ex["nums"], ex["target"]),
            "target": ex["target"],
            "nums": ex["nums"],
        }

    ds = ds.map(to_prompt)

    print(f"Loading base model: {BASE_MODEL} (no SFT -- straight to RL)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    config = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.steps,
        learning_rate=args.lr,
        bf16=(device == "cuda"),
        logging_steps=5,
        save_steps=max(50, args.steps // 4),
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=countdown_reward,
        args=config,
        train_dataset=ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"Starting GRPO training: {args.steps} steps, "
          f"{args.num_generations} generations/prompt, no SFT warmup...")
    trainer.train()

    os.makedirs(args.output, exist_ok=True)
    trainer.save_model(args.output)
    print(f"Saved adapter to {args.output}")


if __name__ == "__main__":
    main()
