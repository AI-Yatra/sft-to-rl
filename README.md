# SFT to RL: Training Models to Think Better

Code and slides for the AI Yatra hands-on session **"Supervised Fine-Tuning
to Reinforcement Learning: Training Models to Think Better"** (Sept 12,
LSEG Hyderabad).

Every result in the slides and docs here came from an actual run — a
laptop CPU and free-tier Colab GPUs, nothing simulated or hypothetical,
including the results where the experiment didn't work.

## What's in here

| | |
|---|---|
| [`slides/`](./slides/) | The talk deck (Slidev) — run `npm install && npm run dev` inside it |
| [`finetune-demo/`](./finetune-demo/) | **Part 1**: one task (text → calendar JSON) taken through LoRA SFT → DPO → GRPO on a small model, ending on an honest negative result and why |
| [`finetune-demo/countdown-rl/`](./finetune-demo/countdown-rl/) | **Part 2**: skip SFT entirely — pure GRPO reinforcement learning from a base model on a verifiable arithmetic reward |

## The two demos, in short

**Part 1 — SFT → DPO → GRPO on calendar-JSON.** LoRA SFT (150 examples, 17
min, CPU) gets field accuracy from ~40% to ~57%. DPO (preference pairs on
top of SFT) pushes it to ~64% and drives hallucinated attendees to exactly
zero. GRPO (reward-based RL on top of both) nudges it a little further. One
field — relative-date arithmetic — never gets fixed across all three
methods, and the root cause (an RL exploration problem: GRPO can't
reinforce an outcome the model almost never samples correctly to begin
with) is the most useful finding of the whole pipeline. Full numbers and
narrative in [`finetune-demo/RESULTS.md`](./finetune-demo/RESULTS.md) and
[`finetune-demo/EXPERIMENTS.md`](./finetune-demo/EXPERIMENTS.md).

**Part 2 — pure RL, no SFT at all.** A raw base model (`Qwen2.5-0.5B`, zero
instruction tuning) trained directly with GRPO on the Countdown arithmetic
task, using a fully verifiable reward (does the equation use the given
numbers once each and equal the target?). 150 GRPO steps (~47 min, one
Colab T4) took it from 0/100 solved to 5/100 solved and from 5% to 67%
valid-format compliance — real capability bootstrapped from reward alone.
Runs live on this laptop's CPU, no GPU needed to demo it:

```bash
cd finetune-demo && uv sync
cd countdown-rl && uv run --project .. python showcase_demo.py
```

Full numbers in [`finetune-demo/countdown-rl/RESULTS.md`](./finetune-demo/countdown-rl/RESULTS.md).

## Reproducing everything

Both demos are self-contained `uv` projects (see
[`finetune-demo/README.md`](./finetune-demo/README.md) for the full
step-by-step, local and Colab). The trained adapter weights for every
stage are committed directly (a few MB each — LoRA adapters, not full
model checkpoints), so nothing needs retraining just to see the results.
