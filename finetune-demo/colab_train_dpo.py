"""
Colab version of train_dpo.py — same DPO-on-top-of-SFT approach, sized for a
free-tier GPU (T4/L4) instead of this laptop's CPU. Key differences from the
local version:
  - bf16 + GPU device instead of CPU fp32
  - full dataset (no --limit needed — GPU makes this fast)
  - larger batch size since VRAM allows it

=== Colab setup (run these in a cell first) ===

    !pip install -q transformers accelerate peft trl datasets

    # Get the finetune-demo/ folder onto the Colab VM. Easiest: push this
    # project to GitHub and clone it, e.g.:
    !git clone <your-repo-url> repo
    %cd repo/rfi/finetune-demo

    # Or upload the finetune-demo/ folder directly via the Colab file browser
    # (Files pane -> upload), then %cd into it.

    # Make sure the SFT adapter (lora-adapter/) is present — either trained
    # earlier on the laptop and uploaded alongside, or retrained here first
    # with train_lora.py (same script works on Colab's GPU too, just faster).

Then run this file:
    !python colab_train_dpo.py --epochs 3 --output ./dpo-adapter
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
SFT_ADAPTER_DIR = os.path.join(HERE, "lora-adapter")
DPO_TRAIN_PATH = os.path.join(HERE, "data", "dpo_train.jsonl")


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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=os.path.join(HERE, "dpo-adapter"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--skip-sft-merge", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    if device == "cpu":
        print("WARNING: no GPU detected — check Runtime > Change runtime type > GPU in Colab.")

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)

    if not args.skip_sft_merge:
        print(f"Merging SFT adapter from: {SFT_ADAPTER_DIR}")
        model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR)
        model = model.merge_and_unload()
    else:
        print("Skipping SFT merge — training DPO straight from base model")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    records = load_records(DPO_TRAIN_PATH, limit=args.limit)
    print(f"DPO training pairs: {len(records)}")
    dataset = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in records
    ])

    dpo_config = DPOConfig(
        output_dir=os.path.join(HERE, "dpo-checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=384,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=(device == "cuda"),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    os.makedirs(args.output, exist_ok=True)
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nDPO adapter saved to: {args.output}")
    print("Download this folder (or push to Drive/HF Hub) before the Colab session ends —")
    print("the VM's disk does not persist between sessions.")


if __name__ == "__main__":
    main()
