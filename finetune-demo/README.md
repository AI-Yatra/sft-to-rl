# Text-to-Calendar-JSON + Countdown-RL

Code behind the AI Yatra talk **"Supervised Fine-Tuning to Reinforcement
Learning: Training Models to Think Better"** (Sept 12, LSEG Hyderabad). Two
real, run demos, matching the two parts of the talk (see `../slides/`):

- **Part 1** (this folder, root level): one task — text → calendar JSON —
  taken through LoRA SFT → DPO → GRPO, on `HuggingFaceTB/SmolLM2-360M-Instruct`.
  Ends on an honest negative result: the `date` field never gets fixed, and
  why (documented in `EXPERIMENTS.md` / `RESULTS.md`).
- **Part 2** (`countdown-rl/`): skip SFT entirely — pure GRPO reinforcement
  learning from a base model (`Qwen/Qwen2.5-0.5B`) on a verifiable
  arithmetic reward (the Countdown task). Real trained adapter weights
  included; runs live in the terminal, no GPU needed.

Every number in `RESULTS.md`, `EXPERIMENTS.md`, and the slide deck came from
an actual run recorded here — nothing is illustrative or hypothetical.

## Files (Part 1 — calendar-JSON pipeline)

| File | What it is |
|---|---|
| `generate_dataset.py` | Builds the synthetic dataset — gold labels are exact by construction (generated from the same values as the text, not hand-annotated) |
| `data/train.jsonl` | 150 training examples used for this run (240 generated total, see below) |
| `data/eval_seen_phrasing.jsonl` | 30 held-out examples, same phrasing style as train |
| `data/eval_holdout_phrasing.jsonl` | 30 examples using phrasing (`day_of_month`, `in_n_days`) **never seen in training** — tests generalization |
| `field_scorer.py` | Scores model output against gold JSON field-by-field (title/date/time/location/attendees), not just "is it valid JSON" |
| `baseline_test.py` | The original 5-example manual smoke test (base vs. an existing public fine-tune) |
| `ablation_test.py` | Tests whether putting "today's date" in the prompt alone fixes date resolution (it doesn't) |
| `train_lora.py` | LoRA SFT training script (peft + transformers `Trainer`, CPU) |
| `eval_before_after.py` | Loads base model and base+adapter, runs both eval sets, prints the comparison |
| `lora-adapter/` | **Trained weights** — the SFT adapter (a few MB, not a full model) |
| `train.log` | Full training log from the run that produced `lora-adapter/` |
| `generate_dpo_dataset.py` | Builds DPO preference pairs — `chosen` is gold JSON, `rejected` has one targeted corruption (wrong date/time, hallucinated/dropped attendee, wrong location, or mis-nested schema) |
| `train_dpo.py` | DPO training on top of the SFT adapter, CPU (local) |
| `dpo-adapter/` | **Trained weights** — the DPO adapter |
| `eval_three_way.py` | Base vs. +SFT vs. +SFT+DPO field-accuracy comparison, run per-stage (`--stage base\|sft\|sft_dpo` then `--combine`) to avoid loading 3 models in one process |
| `generate_grpo_dataset.py` | Builds GRPO prompts — no gold completion needed, just prompt + gold JSON (for reward scoring) + source text |
| `reward_fn.py` | The verifiable GRPO reward function — literally the field scorer, so "reward went up" and "field accuracy went up" are the same measurement |
| `colab_train_dpo.py` | **Run this in Colab** — same DPO approach as `train_dpo.py`, sized for GPU (bf16, bigger batches, full dataset) |
| `colab_train_grpo.py` | **Run this in Colab** — GRPO training; impractical on this laptop's CPU, needs a real GPU for the per-step generation cost |
| `grpo-adapter/` | **Trained weights** — the GRPO adapter (final stage) |
| `EXPERIMENTS.md` | Full experiment log: what was tried, what worked, what didn't, and why |
| `RESULTS.md` | The numbers — every stage's before/after, referenced directly in the slide deck |

See [`countdown-rl/`](./countdown-rl/) for Part 2 — its own README covers
that demo in full.

## How to rerun this reliably, end to end

This folder is a **self-contained `uv` project** — its own `pyproject.toml`,
independent of anything else in the parent repo. Clone the repo, `cd` into
this folder, and `uv` handles the rest (Python version, venv, all packages).

```bash
cd finetune-demo

# 1. one-time environment setup — creates finetune-demo/.venv and installs
#    torch, transformers, accelerate, peft, trl, datasets, python-dateutil
uv sync

# 2. regenerate the dataset (deterministic — same seed, same output every time)
uv run python generate_dataset.py

# 3. train the LoRA adapter (CPU; ~150 examples / 2 epochs took ~10-15 min on this rig)
uv run python train_lora.py --limit 150 --epochs 2 --output ./lora-adapter

# 4. compare base model vs. base+adapter on both eval sets
uv run python eval_before_after.py
```

Everything is seeded (`random.seed(42)` in `generate_dataset.py`) and
deterministic decoding (`do_sample=False`) is used throughout, so reruns
should reproduce the same numbers.

`countdown-rl/` reuses this same `.venv` — run its scripts from within
`finetune-demo/` (`uv run python countdown-rl/showcase_demo.py`) or from
inside the subfolder itself (`cd countdown-rl && uv run --project .. python
showcase_demo.py`).

## What "the weights" actually means here

LoRA doesn't produce a new full model — `lora-adapter/` (and `dpo-adapter/`,
`grpo-adapter/`) contain only the adapter matrices (a few MB each). To use one:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
model = PeftModel.from_pretrained(model, "./lora-adapter")
model = model.merge_and_unload()  # optional: bakes the adapter into a standalone model
```

"Before" is always the same public base model
(`HuggingFaceTB/SmolLM2-360M-Instruct`, downloaded fresh from Hugging Face —
nothing local to keep track of). "After" is that same base model + the
relevant adapter. Chain adapters in stage order (SFT, then DPO, then GRPO)
to reproduce the full pipeline result in `RESULTS.md`.

## Running DPO and GRPO on Colab

Both need more than this laptop's CPU comfortably gives (DPO's double
forward pass, GRPO's 8-generations-per-prompt sampling loop) — `colab_train_dpo.py`
and `colab_train_grpo.py` are the GPU-sized versions of the same approach.

```bash
# In a Colab cell:
!pip install -q transformers accelerate peft trl datasets

# Get this folder onto the Colab VM — push to GitHub and clone, or upload
# rfi/finetune-demo/ directly via the Colab file browser, then:
%cd finetune-demo

# regenerate datasets (or upload the ones already generated here)
!python generate_dataset.py
!python generate_dpo_dataset.py
!python generate_grpo_dataset.py

# if training DPO/GRPO on top of the SFT result, also upload lora-adapter/
# (and dpo-adapter/, if chaining GRPO after DPO) alongside this script.

!python colab_train_dpo.py --epochs 3 --output ./dpo-adapter
!python colab_train_grpo.py --epochs 1 --start-from sft_dpo --output ./grpo-adapter
```

Both scripts auto-detect the GPU (`torch.cuda.is_available()`) and warn if
none is found — check **Runtime > Change runtime type > GPU** first.
`colab_train_grpo.py`'s reward function (`reward_fn.calendar_json_reward`) is
literally the field scorer, so the `reward` metric logged during training
tracks field accuracy directly — no separate eval needed to see whether it's
working, watch it climb (or not) live.

**Colab VMs don't persist storage between sessions** — download or push
`dpo-adapter/` and `grpo-adapter/` (to Drive, or the Hugging Face Hub) before
the session ends, then bring them back to this repo the same way `lora-adapter/`
was: drop the folder in and it's ready for `eval_three_way.py`.
