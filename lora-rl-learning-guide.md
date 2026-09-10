# Understanding LoRA & Lightweight RL for Small LLMs (Laptop Edition)

Goal: build a mental model good enough to (a) explain it to your community, and
(b) actually run a fine-tune on this laptop (16-core Intel, no NVIDIA GPU, ~15GB RAM).

---

## 1. The three things people lump together as "training an LLM"

| Stage | What changes | Data needed | Cost |
|---|---|---|---|
| **Pretraining** | All weights, from scratch | Trillions of tokens | Out of reach on a laptop — skip entirely |
| **Fine-tuning (SFT)** | All or some weights, starting from a pretrained model | Thousands–millions of examples | Full fine-tune of even a 1B model is heavy; **LoRA makes this laptop-feasible** |
| **Preference/RL tuning (RLHF/DPO/GRPO)** | Nudges the fine-tuned model toward preferred outputs | Pairs/rankings of good vs bad outputs | PPO/RLHF = heavy (needs reward model + policy in memory + rollout generation). **DPO = light**, doable on a laptop at small scale |

Your session idea ("take a small LLM, do RL or pretraining, see if it helps on our data") really has two separate, very different-cost experiments hiding in it:
- **SFT with LoRA** — teach the model your data's patterns. This is the one to actually build hands-on.
- **DPO** — teach the model to prefer one style of answer over another. Doable as a lighter second exercise, not full RLHF.

---

## 2. What LoRA actually does (the intuition)

A full fine-tune updates every weight matrix `W` in the model. For a 1B+ parameter
model, that's a lot of gradients and optimizer state to hold in memory (Adam keeps
2 extra copies of every parameter — that's why full fine-tuning needs ~4x the
model's raw size in RAM/VRAM just for optimizer state).

**LoRA's trick:** freeze the original `W` completely. Instead of updating `W`,
learn a *small* correction `ΔW = A·B`, where `A` and `B` are much smaller
matrices (rank `r`, typically 4–64) than `W` itself.

```
Original:  h = W·x                      (W is, say, 4096x4096 = ~16M params, frozen)
LoRA:      h = W·x + (B·A)·x            (A: 4096xr, B: rxr4096, r=8 → ~65K params, trainable)
```

- Only `A` and `B` get gradients and optimizer state → training memory drops by 10-100x depending on `r`.
- The frozen base model can even be loaded once and reused across many different LoRA adapters (this is why people share tiny `.safetensors` LoRA files instead of whole models).
- **QLoRA** = same idea, but the frozen base model is loaded in 4-bit quantized form to shrink RAM further. On CPU, 4-bit quantization support is patchier (bitsandbytes is CUDA-focused) — on this laptop, plain LoRA on an fp32/bf16 *small* model (0.5B–1B params) is the more reliable path, not QLoRA.

**Why this matters for you specifically:** with ~15GB RAM and no GPU, full fine-tuning of even a 1B model is borderline (model + gradients + optimizer state + activations). LoRA on a 0.5B–1B model keeps memory low enough to have headroom.

---

## 3. What DPO does (the intuition, skipping PPO complexity)

Classic RLHF (PPO) needs: a policy model + a frozen reference model + a separate reward model, and it generates rollouts (samples outputs, scores them, then updates) — 3+ models in memory and a generate-then-train loop. That's heavy and finicky even on GPU.

**DPO (Direct Preference Optimization)** skips the reward model and the RL loop
entirely. You just need **pairs**: `(prompt, chosen_response, rejected_response)`.
The loss directly pushes the model's probability of `chosen` up and `rejected`
down relative to a frozen reference copy of the same model. No sampling loop, no
separate reward model — much closer in cost to ordinary supervised fine-tuning.

This is the "RL-flavored" thing that's actually realistic to demo on a laptop.
**GRPO** (used in DeepSeek-R1-style small reasoning models) is another lighter-weight
RL variant worth mentioning conceptually, but it still requires sampling multiple
generations per prompt during training — slower on CPU than DPO. Good to *explain*,
harder to *run live*.

---

## 4. The actual laptop-feasible pipeline

```
1. Pick a tiny base model      → Qwen2.5-0.5B-Instruct or SmolLM2-360M/1.7B
2. Prepare your data           → JSONL of {prompt, response} (for SFT)
                                  or {prompt, chosen, rejected} (for DPO)
3. Load base model in bf16/fp32 on CPU
4. Wrap it with a LoRA adapter (peft)
5. Train with a small batch size + gradient accumulation (transformers Trainer or trl's SFTTrainer)
6. Save just the adapter (a few MB, not the whole model)
7. Eval: compare base model vs base+adapter on held-out prompts
8. (optional) DPO round on top of the SFT adapter using trl's DPOTrainer
```

### Minimal code skeleton (SFT + LoRA, CPU)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)  # CPU by default, no GPU here

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # attention projections are the usual target
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # sanity check: should be << total params

dataset = load_dataset("json", data_files="my_data.jsonl")["train"]

args = TrainingArguments(
    output_dir="./lora-out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # simulate a bigger batch without more memory
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False, bf16=False,          # CPU: stick to fp32 unless you verify bf16 CPU support
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
model.save_pretrained("./my-lora-adapter")   # tiny — just the adapter weights
```

### Minimal code skeleton (DPO, on top of the SFT adapter)

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# dataset rows: {"prompt": ..., "chosen": ..., "rejected": ...}
pref_data = load_dataset("json", data_files="my_preferences.jsonl")["train"]

dpo_config = DPOConfig(
    output_dir="./dpo-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-6,
    beta=0.1,   # controls how hard it pushes away from the reference model
)

dpo_trainer = DPOTrainer(
    model=model,              # your SFT+LoRA model from step above
    ref_model=None,           # None lets trl auto-create a frozen reference copy
    args=dpo_config,
    train_dataset=pref_data,
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

Expect these to run — not fast. On CPU, a few hundred SFT steps on a 0.5B model
with short sequences might take anywhere from ~15 minutes to a couple hours
depending on dataset/sequence length. Budget for that in a session (pre-run it,
show the result, let people tweak and re-run a *tiny* slice live).

---

## 5. What to install (CPU-only, replacing the current CUDA torch build)

```bash
# swap the cu130 torch wheel for CPU-only
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers peft trl datasets accelerate
```

Skip: `bitsandbytes`, `unsloth` (CUDA-only, won't help here).

---

## 6. Measuring "did it help"

Before/after comparison on a held-out slice of your own data:
- Same prompts, base model vs LoRA-tuned model outputs, side by side.
- If you have any labeled correctness (e.g. right/wrong answers, or an internal
  rubric), score both and compare accuracy — that's your "did fine-tuning help" number.
- For the DPO round, check whether generations shift toward the "chosen" style
  on prompts *not* in the training pairs — that's the generalization check.

---

## 7. Good primary sources (concepts, not just APIs)

- LoRA paper (Hu et al., 2021) — the original, short and readable: "LoRA: Low-Rank Adaptation of Large Language Models"
- Hugging Face PEFT docs — practical LoRA config reference
- Hugging Face TRL docs — SFTTrainer, DPOTrainer, GRPOTrainer usage
- DPO paper (Rafailov et al., 2023) — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- DeepSeek-R1 paper — for the GRPO story, if you want to gesture at "how modern reasoning RL works" conceptually without running it live

---

## 8. Suggested framing for the session (once you're comfortable with the above)

1. 10 min: pretraining vs SFT vs RLHF/DPO — the table from section 1
2. 15 min: LoRA intuition (section 2) with a live `print_trainable_parameters()` demo — dramatic to see "0.3% of params trainable"
3. 30–40 min: hands-on — everyone fine-tunes the same tiny model on a shared small dataset, compares outputs
4. 15 min: DPO as a concept walkthrough + a pre-run before/after result (don't make everyone train DPO live — too slow)
5. Wrap: what would change if you *did* have a GPU (bigger models, QLoRA, real PPO/GRPO)
