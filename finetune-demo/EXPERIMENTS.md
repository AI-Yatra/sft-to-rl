# Experiment Log & Roadmap — Text-to-Calendar-JSON

Full history of what's been tried, what's running now, and what's still worth
trying — kept as one place to look instead of scattered across chat history.
See `RESULTS.md` for the detailed numbers behind experiments 1-3.

## Completed experiments

### 1. SFT (LoRA), local CPU
150 examples, 2 epochs, ~17 min. **Result: 38-42% → 56-59% field accuracy**,
hallucinations cut ~5-6x. `date` field barely moved (0/30 → 1-5/30).

### 2. DPO on top of SFT, Colab T4
150 preference pairs, 3 epochs. First attempt used a full-fine-tune learning
rate (5e-6) and produced a dead run (flat loss at ln(2)). Retrained at
lr=5e-5 (LoRA needs a higher LR). **Result: 56-59% → 62-66%**, hallucinations
driven to **exactly zero** on both eval sets.

### 3. GRPO (general reward), Colab T4
200 prompts, 8 generations/prompt, 1 epoch, on top of SFT+DPO. Reward
averaged across all 5 fields equally. **Result: 62% → 64% (seen), flat at
66% (held-out)** — a small, real gain, but diluted because 4 of 5 fields
were already near-ceiling.

### 4. Chain-of-thought SFT (CoT), Colab T4 — negative result, but informative
Hypothesis: teaching the model to write out date arithmetic before the JSON
(via imitation) would fix `date`. Built a fresh 200-example CoT dataset
(reasoning line + JSON line), trained fresh LoRA from base, 3 epochs.

**Result: date accuracy stayed at 0/30 on BOTH eval sets.** Overall accuracy
(44.7% seen, 55.3% holdout) was roughly flat vs. plain SFT. **Conclusion:
imitating the *shape* of reasoning doesn't teach the *arithmetic itself*** —
the model learned to produce reasoning-shaped text without it being
computationally correct. This is the exact gap RL (reward on the outcome,
not the demonstration) is supposed to close.

### 5. Date-weighted GRPO on top of CoT-SFT — negative result, root cause identified
Hypothesis: GRPO's general reward diluted the date signal (experiment 3).
Concentrating reward on `date` specifically (60% of reward on date alone,
40% split across the other 4 fields), combined with the CoT format, should
succeed where both CoT-SFT alone and general-reward GRPO alone failed.

200 prompts, 8 generations/prompt, 1 epoch, on top of the CoT-SFT adapter,
Colab T4, ~43 min.

**Result: date accuracy stayed at 0/30 on both eval sets** (seen went 1→0,
held-out stayed 0→0). Overall accuracy flat-to-slightly-worse (49.3%→49.3%
seen, 54.7%→52.0% held-out). Reweighting the reward did not help.

**Root cause: GRPO can only reinforce behavior the policy already samples
by chance.** With 8 generations per prompt and the CoT-SFT baseline getting
`date` right ~0-3% of the time, the date-component of the reward is nearly
always constant (zero) across all 8 samples for a given prompt — there is no
variance in the group for GRPO's advantage estimate to learn from. Weighting
the reward toward date does nothing if the policy essentially never samples
a correct date in the first place. **This is the standard RL exploration
problem, not a reward-design bug** — reward shaping amplifies existing
signal, it doesn't manufacture signal from a near-zero success rate.

**Implication for the backlog below:** items that raise the *base* success
rate before RL touches it (explicit arithmetic scaffolding, tool use,
curriculum ordering, or simply a much larger/more repetitive SFT dataset)
are now the higher-priority experiments — GRPO is very unlikely to help
further until the policy can sometimes get `date` right on its own.

## Backlog — additional work worth trying

Roughly ordered by how directly each targets the still-open `date` problem:

0. **Raise base success rate before another RL pass** (see experiment 5's
   root cause) — any of the items below that increase how often the model
   gets `date` right *unassisted* should come before a third GRPO attempt.
1. **Explicit arithmetic reasoning format** — instead of free-form reasoning
   text, force a structured step ("today=Tuesday(1), target=Thursday(3),
   diff=(3-1)mod7=2, date=today+2"). More scaffolded than the current CoT,
   closer to how the answer is actually computed.
2. **Tool use / calculator escape hatch** — let the model emit a function
   call (`resolve_date(ref, "next thursday")`) instead of doing arithmetic
   in-context at all. Sidesteps the capability question entirely; more
   representative of how this would actually ship in production.
3. **Curriculum ordering** — train on `tomorrow`/`day_after_tomorrow` first
   (trivial offset), then `this/next weekday` (needs mod-7 arithmetic), then
   `day_of_month`/`in_n_days` last. Current training mixes all difficulty
   levels randomly from step one.
4. **Reward-weighting sweep** — try a few date-weight ratios (50/50, 70/30,
   90/10 date-vs-rest) to see where the tradeoff between "fixes date" and
   "regresses other fields" sits, rather than picking one weight blind.
5. **Scale check** — run the same CoT-SFT + date-weighted-GRPO recipe on a
   larger base (Qwen3.5-0.8B) to see whether this is a capability ceiling at
   360M specifically, or a data/recipe problem that persists at larger scale.
6. **DPO with date-only corruptions, more pairs** — the existing DPO run's
   `wrong_date` corruption was 1 of 6 corruption types (~24/150 pairs). A
   DPO run using *only* wrong-date corruptions, at higher volume, is a
   cheaper experiment than another GRPO pass and worth trying first.
7. **Few-shot prompting baseline** — for completeness, measure whether
   stuffing 3-5 worked date-resolution examples into the prompt (no training
   at all) gets partway there. Useful contrast for the "prompting vs.
   training" section of the talk even if it's not the deployed solution.

## Session logistics still open (Sept 12 talk)

- [ ] Minute-by-minute agenda fitting SFT→DPO→GRPO into the 3.5-hour window
- [ ] Decide which result becomes the live demo vs. the pre-run "here's what
      we found" slide (the date-arithmetic negative result is a strong
      candidate for a live moment — it's a real surprise, not a canned win)
- [ ] Package the repo for attendees to follow along with their own laptops
