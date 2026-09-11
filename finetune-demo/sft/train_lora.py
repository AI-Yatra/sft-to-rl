"""
LoRA SFT training for the text-to-calendar-JSON task, on CPU.

Base model: HuggingFaceTB/SmolLM2-360M-Instruct
Data: data/train.jsonl (built by generate_dataset.py — gold labels are exact
by construction, since they're generated from the same values as the text).

Usage:
    .venv/bin/python train_lora.py [--epochs N] [--limit N] [--output DIR]
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, "data", "train.jsonl")


def load_records(path, limit=None):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]
    return records


def build_tokenized_dataset(records, tokenizer, max_length=256):
    input_ids_list, labels_list, attn_list = [], [], []

    for r in records:
        messages = r["messages"]
        prompt_messages = messages[:2]  # system + user
        full_messages = messages  # system + user + assistant

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]

        if len(full_ids) <= len(prompt_ids):
            continue  # degenerate, skip

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attn_list.append([1] * len(full_ids))

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attn_list,
    })


class PadCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        batch_input_ids, batch_labels, batch_attn = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)
            batch_attn.append(f["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=os.path.join(HERE, "lora-adapter"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    records = load_records(TRAIN_PATH, limit=args.limit)
    print(f"Training examples: {len(records)}")
    dataset = build_tokenized_dataset(records, tokenizer)
    print(f"Tokenized examples (after skipping degenerate): {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=os.path.join(HERE, "train-checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer),
    )

    trainer.train()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nAdapter saved to: {args.output}")


if __name__ == "__main__":
    main()
