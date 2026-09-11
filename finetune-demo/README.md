# Text-to-Calendar-JSON + Countdown-RL

Code behind the AI Yatra talk **"Supervised Fine-Tuning to Reinforcement
Learning: Training Models to Think Better"** (Sept 12, LSEG Hyderabad). Two
real, run demos, matching the two parts of the talk (see `../slides/`):

- **Part 1** (this folder): one task — text → calendar JSON — taken through
  LoRA SFT → DPO → GRPO, on `HuggingFaceTB/SmolLM2-360M-Instruct`. Organized
  by method — [`sft/`](./sft/), [`dpo/`](./dpo/), [`grpo/`](./grpo/) — each
  self-contained with its own dataset generator, training script(s), trained
  adapter weights, and train/eval data. Ends on an honest negative result:
  the `date` field never gets fixed, and why (`EXPERIMENTS.md` / `RESULTS.md`).
- **Part 2** (`countdown-rl/`): skip SFT entirely — pure GRPO reinforcement
  learning from a base model (`Qwen/Qwen2.5-0.5B`) on a verifiable
  arithmetic reward (the Countdown task). Real trained adapter weights
  included; runs live in the terminal, no GPU needed.

Every number in `RESULTS.md`, `EXPERIMENTS.md`, and the slide deck came from
an actual run recorded here — nothing is illustrative or hypothetical.

## Live walkthrough (VS Code + CodeTour)

For presenting this on stage: install the
[CodeTour](https://marketplace.visualstudio.com/items?itemName=vsls-contrib.codetour)
extension, open this repo in VS Code, and run **CodeTour: Start Tour** from
the command palette (or the CodeTour panel in the sidebar). `../.tours/grpo-reward-function.tour`
walks through `grpo/reward_fn.py` line by line — what "verifiable reward"
means, the hallucination-penalty reward-hacking guard, and how it connects
to GRPO's group-relative advantage (including why the `date` field couldn't
be fixed by reward reweighting alone).

## Layout

```
finetune-demo/
├── sft/            LoRA SFT — dataset generator, train_lora.py, lora-adapter/, data/
├── dpo/            DPO — preference-pair generator, train scripts, dpo-adapter/, data/
├── grpo/           GRPO — prompt generator, reward_fn.py, train scripts, grpo-adapter/, data/
│                   + reward function walkthrough (see .tours/ above)
├── countdown-rl/   Part 2 — pure-RL Countdown demo (own README)
├── field_scorer.py           shared field-accuracy scorer, used by every stage's eval
├── eval_before_after.py      base vs. +SFT comparison
├── eval_three_way.py         base vs. +SFT vs. +SFT+DPO comparison
├── colab_eval_all.py         all four stages in one process (Colab, GPU)
├── eval_results/             saved four-way eval JSON
├── EXPERIMENTS.md            full experiment log — what was tried, what worked, why
└── RESULTS.md                the numbers, referenced directly in the slide deck
```

## How to rerun this reliably, end to end

This folder is a **self-contained `uv` project** — its own `pyproject.toml`,
independent of anything else in the parent repo. Clone the repo, `cd` into
this folder, and `uv` handles the rest (Python version, venv, all packages).

```bash
cd finetune-demo

# 1. one-time environment setup — creates finetune-demo/.venv and installs
#    torch, transformers, accelerate, peft, trl, datasets, python-dateutil
uv sync

# 2. regenerate the SFT dataset (deterministic — same seed, same output every time)
uv run python sft/generate_dataset.py

# 3. train the LoRA adapter (CPU; ~150 examples / 2 epochs took ~10-15 min on this rig)
uv run python sft/train_lora.py --limit 150 --epochs 2 --output ./sft/lora-adapter

# 4. compare base model vs. base+adapter on both eval sets
uv run python eval_before_after.py
```

Everything is seeded (`random.seed(42)` in `sft/generate_dataset.py`) and
deterministic decoding (`do_sample=False`) is used throughout, so reruns
should reproduce the same numbers.

`countdown-rl/` reuses this same `.venv` — run its scripts from within
`finetune-demo/` (`uv run python countdown-rl/showcase_demo.py`) or from
inside the subfolder itself (`cd countdown-rl && uv run --project .. python
showcase_demo.py`).

## What "the weights" actually means here

LoRA doesn't produce a new full model — `sft/lora-adapter/`, `dpo/dpo-adapter/`,
and `grpo/grpo-adapter/` each contain only adapter matrices (a few MB each).
To use one:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
model = PeftModel.from_pretrained(model, "./sft/lora-adapter")
model = model.merge_and_unload()  # optional: bakes the adapter into a standalone model
```

"Before" is always the same public base model
(`HuggingFaceTB/SmolLM2-360M-Instruct`, downloaded fresh from Hugging Face —
nothing local to keep track of). "After" is that same base model + the
relevant adapter. Chain adapters in stage order (`sft/lora-adapter` →
`dpo/dpo-adapter` → `grpo/grpo-adapter`) to reproduce the full pipeline
result in `RESULTS.md`.

## Running DPO and GRPO on Colab

Both need more than this laptop's CPU comfortably gives (DPO's double
forward pass, GRPO's 8-generations-per-prompt sampling loop) —
`dpo/colab_train_dpo.py` and `grpo/colab_train_grpo.py` are the GPU-sized
versions of the same approach.

```bash
# In a Colab cell:
!pip install -q transformers accelerate peft trl datasets

# Get this folder onto the Colab VM — push to GitHub and clone, or upload
# rfi/finetune-demo/ directly via the Colab file browser, then:
%cd finetune-demo

# regenerate datasets (or upload the ones already generated here)
!python sft/generate_dataset.py
!python dpo/generate_dpo_dataset.py
!python grpo/generate_grpo_dataset.py

# if training DPO/GRPO on top of the SFT result, also upload sft/lora-adapter/
# (and dpo/dpo-adapter/, if chaining GRPO after DPO) alongside these scripts.

%cd dpo && !python colab_train_dpo.py --epochs 3 --output ./dpo-adapter && %cd ..
%cd grpo && !python colab_train_grpo.py --epochs 1 --start-from sft_dpo --output ./grpo-adapter && %cd ..
```

Both scripts auto-detect the GPU (`torch.cuda.is_available()`) and warn if
none is found — check **Runtime > Change runtime type > GPU** first.
`grpo/colab_train_grpo.py`'s reward function (`reward_fn.calendar_json_reward`)
is literally the field scorer, so the `reward` metric logged during training
tracks field accuracy directly — no separate eval needed to see whether it's
working, watch it climb (or not) live.

**Colab VMs don't persist storage between sessions** — download or push
`dpo/dpo-adapter/` and `grpo/grpo-adapter/` (to Drive, or the Hugging Face
Hub) before the session ends, then bring them back to this repo the same way
`sft/lora-adapter/` was: drop the folder in and it's ready for `eval_three_way.py`.
