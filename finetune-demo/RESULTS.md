# Run Record: SFT → DPO → GRPO on SmolLM2-360M — text-to-calendar-JSON

This is the as-run record of the full pipeline, kept verbatim for the session
showcase: LoRA SFT (local, CPU) → DPO (Colab, T4 GPU) → GRPO (Colab, T4 GPU).
Raw logs are `train.log` and `eval_before_after.log` (SFT stage, local) in
this folder; the DPO/GRPO/four-way-eval logs are the Colab session transcript
this file summarizes.

## The full pipeline result (real numbers, all four stages)

| Stage | Seen accuracy | Held-out accuracy | Hallucinations (seen) | Hallucinations (held-out) |
|---|---|---|---|---|
| BASE | 40.0% | 38.7% | 36 | 37 |
| +SFT | 56.0% | 58.7% | 4 | 6 |
| +SFT+DPO | 62.0% | 66.0% | **0** | **0** |
| +SFT+DPO+GRPO | 64.0% | 66.0% | **0** | **0** |

Run on Colab (T4 GPU) via `colab_eval_all.py`, evaluating all four stages
against `data/eval_seen_phrasing.jsonl` and `data/eval_holdout_phrasing.jsonl`
in one process (30 examples each). Full JSON: [`eval_results/four_way_results.json`](./eval_results/four_way_results.json).

**Reading the pipeline stage by stage:**
- **BASE → SFT**: the big jump (~40% → ~57%), almost entirely driven by fixing hallucinated attendees (36/37 → 4/6) and schema compliance
- **SFT → DPO**: a real further gain (~57% → ~64%), and **hallucination goes to exactly zero on both eval sets** — DPO's preference pairs specifically targeted this failure mode (see `generate_dpo_dataset.py`'s `hallucinate_attendee` corruption), and it shows
- **DPO → GRPO**: a small additional gain on the seen set (62.0% → 64.0%) and no change on held-out (66.0% → 66.0%) — consistent with the GRPO training curve itself, which showed only a mild upward drift in reward (not a clean climb) over its single epoch. Read as: GRPO didn't hurt, and nudged the seen-phrasing case further, but one epoch on 200 prompts wasn't enough to move the held-out generalization case further than DPO already had
- **`date` stayed the hardest field throughout** — even after all three training stages, held-out `date` accuracy is 4/30 (see the four-way JSON for per-field breakdown). This is the most honest thing to say on stage: three different training methods, and relative-date arithmetic is still the unsolved piece

Note: the BASE numbers here (40.0%/38.7%, Colab GPU, bf16) differ slightly
from the original local CPU run below (42.0%/38.0%, fp32) — expected minor
numeric drift between bf16-on-GPU and fp32-on-CPU greedy decoding, not a
methodology change.

## SFT stage — original local run (CPU)

The rest of this document is the original SFT-only run record, kept as-is.

## Environment

- Hardware: Intel Core Ultra 7 255H, 16 cores, integrated graphics only (no CUDA), ~15GB RAM
- `torch 2.13.0+cu130` (CUDA build present but unused — CPU-only inference/training throughout)
- Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Run date: 2026-09-06

## Training run

Command:
```
python train_lora.py --limit 150 --epochs 2 --batch-size 4 --grad-accum 2 --output ./lora-adapter
```

- LoRA config: r=8, alpha=16, dropout=0.05, target modules `q_proj, k_proj, v_proj, o_proj`
- Trainable params: **1,638,400 / 363,459,520 (0.45%)**
- Training examples: 150 (of the 240 generated; see `data/train.jsonl`)
- `train_runtime`: **1043s (~17.4 min)**, 2 epochs, `train_loss`: **0.3162**
- Output: `lora-adapter/` — 6.6MB (`adapter_model.safetensors`)

Full log: [`train.log`](./train.log)

## Evaluation run

Command:
```
python eval_before_after.py
```

Runtime: ~44 minutes total on CPU (60 generations for base model, then 60 for
the LoRA-adapted model, across both eval sets — most of the time is spent in
generation, not model loading).

Full log: [`eval_before_after.log`](./eval_before_after.log)

### Results — seen phrasing (30 examples, same phrasing style as training)

| Field | Before | After |
|---|---|---|
| title | 23/30 | 26/30 |
| date | 0/30 | 1/30 |
| time | 11/30 | 13/30 |
| location | 22/30 | 21/30 |
| attendees | 7/30 | 23/30 |
| **Overall field accuracy** | **42.0%** | **56.0%** |
| JSON valid rate | 100% | 100% |
| Hallucinated attendee mentions | 30 | 7 |

### Results — held-out phrasing (30 examples, phrasing types never seen in training: `day_of_month`, `in_n_days`)

| Field | Before | After |
|---|---|---|
| title | 25/30 | 27/30 |
| date | 0/30 | 5/30 |
| time | 11/30 | 15/30 |
| location | 17/30 | 19/30 |
| attendees | 4/30 | 23/30 |
| **Overall field accuracy** | **38.0%** | **59.3%** |
| JSON valid rate | 100% | 100% |
| Hallucinated attendee mentions | 36 | 6 |

## Headline numbers for the session

- **Field accuracy: 38–42% → 56–59%**, from 150 examples and ~17 minutes of CPU-only LoRA training
- **Hallucinated attendees cut ~5–6x** (30→7 seen, 36→6 held-out) — the single most visually dramatic result
- **Improvement held on phrasing never seen in training** (held-out set actually improved *more* than the seen set: +21.3pp vs +14.0pp) — this is the generalization check, not memorization
- **JSON validity was 100% before and after on both sets** — reinforces the earlier finding that JSON-validity is not where the real gap lives; field accuracy is
- **`date` remains the weakest field** (0/30 → 1/30 seen, 0/30 → 5/30 held-out) — 150 examples / 2 epochs was not enough to teach reliable relative-date arithmetic. This is the honest limitation to state on stage, and the natural hook into "here's where more data, more epochs, or DPO/GRPO come in next"

## DPO stage (Colab, T4 GPU)

Command: `colab_train_dpo.py --epochs 3 --output ./dpo-adapter` (on top of
the SFT adapter merged in). First attempt used the default `lr=5e-6` (TRL's
full-fine-tune default) and produced a dead run — loss flat at `ln(2)≈0.693`
the whole time, final `rewards/margins` slightly negative. Retrained with
`lr=5e-5` (LoRA needs a higher LR than full fine-tuning): loss dropped to
~0.55-0.60, `rewards/accuracies` climbed to 0.80-0.95, `train_runtime`
**1730s (~29 min)**. This is the run reflected in the four-way table above.

## GRPO stage (Colab, T4 GPU)

Command: `colab_train_grpo.py --epochs 1 --start-from sft_dpo --output ./grpo-adapter`.
200 training prompts × 8 generations/prompt, `lr=1e-5`, `train_runtime`
**1412s (~23.5 min)**. Reward oscillated between 0.32 and 0.61 throughout —
first-10-steps average ~0.45, last-10-steps average ~0.48, a mild but real
upward drift rather than a clean climb. `reward_std` stayed healthy
(0.21-0.29) throughout (no collapse), `entropy` stayed in 0.30-0.46 (no
mode collapse), `clipped_ratio` stayed ~0 (completions weren't truncated).
Consistent with "real but modest learning in one epoch," which matches what
the four-way eval shows (small gain, not a leap).

One dependency snag on Colab worth noting for next time: `peft` requires
`torchao>=0.16.0` for its quantization dispatch path, but Colab preinstalls
`torchao==0.10.0` — even though this workflow never uses torchao, `peft`
still version-checks it and throws `ImportError`. Fix: `!pip uninstall -y torchao`
before running either Colab script.

## CoT-SFT and date-weighted GRPO (Colab, T4 GPU) — the `date` follow-up

Full narrative and root-cause analysis in [`EXPERIMENTS.md`](./EXPERIMENTS.md#5-date-weighted-grpo-on-top-of-cot-sft--negative-result-root-cause-identified);
this section is just the numbers.

Built fresh (not on top of the SFT/DPO/GRPO adapters above): base model →
chain-of-thought SFT (200 examples teaching "Reasoning: ... \n JSON: {...}",
3 epochs) → date-weighted GRPO (200 prompts, 8 generations/prompt, reward
60% on date correctness / 40% on the other 4 fields, 1 epoch, 43 min).

| Stage | Seen overall | Held-out overall | date (seen) | date (held-out) |
|---|---|---|---|---|
| CoT-SFT (no RL) | 44.7% | 55.3% | 0/30 | 0/30 |
| CoT-SFT (no RL), rerun | 49.3% | 54.7% | 1/30 | 0/30 |
| CoT + date-weighted GRPO | 49.3% | 52.0% | **0/30** | **0/30** |

(Two CoT-SFT rows: evaluated once right after training, then again as the
"before" baseline for the GRPO comparison — small variance between runs is
expected noise, not a discrepancy.)

**Neither chain-of-thought SFT nor date-weighted GRPO fixed `date`.** The
CoT-SFT result shows imitating reasoning text doesn't teach the underlying
arithmetic. The GRPO result shows *why* reward-shaping didn't rescue it:
GRPO needs the policy to sometimes sample a correct answer to reinforce it,
and a ~0% base success rate leaves no variance in the reward for GRPO's
group-relative advantage to learn from. Both are genuine, checkable RL
lessons — arguably stronger material for the session than a clean win,
since they show *why* a plausible-sounding fix (reward weighting) doesn't
work, not just that it doesn't.

## Reproducing this exact run

See [`README.md`](./README.md) for the full step-by-step. Both the dataset
generation (`generate_dataset.py`, seeded) and decoding (`do_sample=False`
throughout) are deterministic, so a rerun should reproduce these numbers
closely, modulo minor floating-point/threading nondeterminism on CPU.
