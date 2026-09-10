"""
GRPO training for the text-to-calendar-JSON task, sized for a free Colab GPU
(T4/L4) — this is the piece that never made sense to attempt on the laptop's
CPU (see rfi/lora-rl-learning-guide.md): GRPO samples several completions per
prompt every step, which needs real parallel generation throughput.

Reward is `reward_fn.calendar_json_reward` — a verifiable reward computed
directly from the gold JSON each prompt was generated from (no separate
reward model), same pattern as the Hugging Face IFStruct/GRPO blog post this
was modeled on: https://huggingface.co/blog/grpo-with-trl-ifstruct

=== Colab setup (run in a cell first) ===

    !pip install -q transformers accelerate peft trl datasets

    !git clone <your-repo-url> repo
    %cd repo/rfi/finetune-demo

    # (Optional) start from the SFT+DPO result instead of the base model —
    # upload lora-adapter/ and dpo-adapter/ alongside this script if so.

Then run:
    !python colab_train_grpo.py --epochs 1 --output ./grpo-adapter

Note on cost: with num_generations=8 and ~200 training prompts, this is
~1600 generations per epoch. Budget real GPU time for this even on Colab —
it will not be a 2-minute run.
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward_fn import calendar_json_reward

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
SFT_ADAPTER_DIR = os.path.join(HERE, "lora-adapter")
DPO_ADAPTER_DIR = os.path.join(HERE, "dpo-adapter")
GRPO_TRAIN_PATH = os.path.join(HERE, "data", "grpo_train.jsonl")


def load_records(path, limit=None):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=os.path.join(HERE, "grpo-adapter"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--start-from", choices=["base", "sft", "sft_dpo"], default="base",
                         help="Which checkpoint to start GRPO from")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    if device == "cpu":
        print("WARNING: no GPU detected. GRPO's per-step generation cost makes this "
              "impractical on CPU — check Runtime > Change runtime type > GPU.")

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)

    if args.start_from in ("sft", "sft_dpo"):
        print(f"Merging SFT adapter from: {SFT_ADAPTER_DIR}")
        model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR)
        model = model.merge_and_unload()
    if args.start_from == "sft_dpo":
        print(f"Merging DPO adapter from: {DPO_ADAPTER_DIR}")
        model = PeftModel.from_pretrained(model, DPO_ADAPTER_DIR)
        model = model.merge_and_unload()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    records = load_records(GRPO_TRAIN_PATH, limit=args.limit)
    print(f"GRPO training prompts: {len(records)} (x{args.num_generations} generations/prompt/step)")
    dataset = Dataset.from_list(records)  # columns: prompt, gold_json, source_text

    grpo_config = GRPOConfig(
        output_dir=os.path.join(HERE, "grpo-checkpoints"),
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        temperature=args.temperature,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=(device == "cuda"),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=calendar_json_reward,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    os.makedirs(args.output, exist_ok=True)
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nGRPO adapter saved to: {args.output}")
    print("Watch the logged `reward` metric climb during training — that IS the field-accuracy proxy here, "
          "since the reward function is the field scorer. Download this folder before the session ends.")


if __name__ == "__main__":
    main()
