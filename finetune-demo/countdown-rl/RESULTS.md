# Run Record: Pure-RL Countdown (GRPO, no SFT) on Qwen2.5-0.5B

Based on the idea from [this X post](https://x.com/TheGlobalMinima/status/2096532361844609320)
(TinyZero/Jiayi-Pan style): skip SFT entirely, train a base model directly
with GRPO on a verifiable reward, and watch it learn to reason from scratch.

## Setup

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` (base, not instruct — no SFT applied at any point) |
| Dataset | `Jiayi-Pan/Countdown-Tasks-3to4` (490K examples; used first 3000, shuffled seed=42) |
| Task | Given N numbers, produce an equation using each exactly once that equals a target |
| Reward | `reward_fn.countdown_reward` — 0.0 (no valid `<answer>` block) / 0.1 (valid format, wrong numbers) / 0.2 (right numbers, wrong result) / 1.0 (solved) — fully verifiable, no reward model |
| Training | GRPO via TRL, LoRA r=16/alpha=32 on q/k/v/o_proj, 150 steps, batch=8, 8 generations/prompt, max_completion_length=300, lr=1e-5, bf16 |
| Hardware | Colab T4 (15.6GB VRAM) |
| Runtime | 2849s (~47.5 min) for 150 steps |

**Note on the base model**: the X post named `Qwen3.5-0.8B-Base`, but that
checkpoint's config shows `architectures: ["Qwen3_5ForConditionalGeneration"]`
— a multimodal, hybrid linear-attention architecture, not a plain causal LM,
and support in the current peft/trl/transformers stack was unproven. Swapped
to `Qwen2.5-0.5B` — a standard decoder-only base model and the actual model
family used in the original TinyZero/Countdown reproductions this idea
descends from.

## Headline result

| | BASE (untrained) | +GRPO (150 steps) |
|---|---|---|
| Solved (exact answer) | 0/100 (0.0%) | **5/100 (5.0%)** |
| Right numbers, wrong result | 1/100 (1.0%) | 14/100 (14.0%) |
| Valid `<answer>` format at all | 5/100 (5.0%) | **67/100 (67.0%)** |
| Mean reward | 0.0060 | **0.1210** |

Evaluated greedy (no sampling) on 100 held-out examples — indices 3000-3100
of the shuffled dataset, never seen during the 150-step training run (which
only used indices 0-3000, and at 8 prompts/step × 150 steps = 1200 prompt
draws, didn't even exhaust that subset once).

**This is a real, positive result**, and the shape of the improvement is the
textbook RL pattern: format compliance moved the most (5% → 67% — the model
went from mostly ignoring the requested `<answer>` tag structure to reliably
producing it), correct-numbers-but-wrong-arithmetic moved next (1% → 14%),
and actually-solved moved least but nonzero (0% → 5%). That ordering —
*structure learned before correctness* — is exactly what GRPO is expected to
do: the reward can't teach anything about arithmetic correctness until the
model is reliably emitting a parseable answer in the first place, so format
compliance is the "low-hanging fruit" the policy gradient grabs first.

## Training dynamics (from the live log, 150 steps)

- **Reward**: mean drifted from ~0.025 (early steps) to ~0.05 (late steps) —
  a real but noisy upward trend, not a clean climb. `frac_reward_zero_std`
  (fraction of prompts where all 8 generations got the identical reward,
  i.e. no learning signal for that prompt) fluctuated between 0 and 0.2 —
  meaning most steps *did* have some reward variance for GRPO to use, unlike
  the calendar-JSON `date` field experiment where this was pinned near 1.0.
- **Entropy**: stable in the 1.1-1.6 range throughout — no mode collapse.
- **Completion length**: noisy, oscillating 55-110 tokens (mean), no clean
  systematic lengthening trend visible at this step count.
- **This was NOT the dramatic "aha moment"** shown in published TinyZero
  writeups (which typically train for thousands of steps and show accuracy
  climbing well past 50%). 150 steps is ~0.05 epochs over the 3000-example
  subset — a small fraction of the compute those demos use. What we got
  instead is the *early, quantifiable* version of the same effect: real
  measurable gain in ~47 minutes on a free-tier GPU, with the format-before-
  correctness ordering already visible.

## What this demonstrates for the talk

This pairs well with the calendar-JSON pipeline's negative result on `date`:
that experiment showed GRPO *can't* create signal where none exists (near-
zero base success rate → no reward variance → no gradient). This experiment
shows GRPO *can* bootstrap real capability from nothing when the reward is
well-shaped and verifiable, even from a base model with zero instruction
tuning — going from "ignores the requested format entirely" to "sometimes
solves genuine arithmetic puzzles it was never shown a single labeled
example of," purely from reward signal.

## Is this demo-ready?

Yes, with the honest framing above: don't oversell it as the dramatic
TinyZero "aha moment" (that needs far more steps than fit in a live-audience
window), but the numbers are real, reproducible, and the mechanism (reward
went up because *format* went up, correctness followed) is a genuinely
interesting, defensible story about how RL shapes learning differently than
imitation. Extending training (e.g. 400-600 steps, ~2-3 hours) before Sept 12
would likely push solved-rate noticeably higher and is worth doing if time
allows, but even the current 150-step run is legitimate, presentable data.

## Reproducing this run

```bash
cd finetune-demo   # this folder is a self-contained uv project, see ../README.md
uv sync
cd countdown-rl
uv run --project .. python reward_fn.py  # self-test
uv run --project .. python colab_train_countdown_grpo.py --steps 150 --num-examples 3000 \
    --batch-size 8 --num-generations 8 --max-completion-length 300 \
    --output ./countdown-grpo-adapter
uv run --project .. python eval_countdown.py --adapter ./countdown-grpo-adapter --n 100
```

(In Colab, use plain `!pip install -q transformers accelerate peft trl datasets`
and `!python ...` instead — `uv` isn't relevant there.)

Dataset shuffle is seeded (`seed=42`), decoding is greedy (`do_sample=False`)
in eval — reruns should reproduce these numbers closely, modulo minor
floating-point/GPU nondeterminism.
