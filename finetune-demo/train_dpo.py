"""
DPO training on top of the SFT LoRA adapter, on CPU.

Stage order: base model -> +SFT LoRA (merged in) -> +DPO LoRA (new adapter,
trained here). DPOTrainer needs a plain (non-Peft) model + a peft_config to
attach its own adapter and auto-derive the reference model by disabling that
adapter — passing an already-loaded PeftModel alongside peft_config is
rejected by trl, hence the merge step below.

Usage:
    .venv/bin/python train_dpo.py [--epochs N] [--limit N] [--output DIR]
"""
import argparse
import json
import os

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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=os.path.join(HERE, "dpo-adapter"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--skip-sft-merge", action="store_true",
                         help="Train DPO straight from the base model instead of on top of the SFT adapter")
    args = parser.parse_args()

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

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
        fp16=False,
        bf16=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # auto-derived by disabling the new adapter (see module docstring)
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
    print("Note: this adapter was trained on top of the SFT-merged model — "
          "load it the same way at eval time (merge SFT adapter into base, then attach this one).")


if __name__ == "__main__":
    main()
