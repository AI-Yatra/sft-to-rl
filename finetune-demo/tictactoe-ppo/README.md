# Part 3 — PPO on tic-tac-toe (GPT-2, no dataset at all)

A third method alongside the other two demos in this repo, and the most
"textbook RL" of the three:

| | task | method | where the signal comes from |
|---|---|---|---|
| Part 1 (`../`) | text → calendar JSON | LoRA SFT → DPO → GRPO | labelled examples, then preferences, then a verifiable reward |
| Part 2 (`../countdown-rl/`) | Countdown arithmetic | pure GRPO | a verifiable reward, no SFT |
| **Part 3 (here)** | **tic-tac-toe** | **PPO with a value head** | **nothing but the game result: +1 / 0 / −1** |

This is the Python-script version of `../../tictactoe_ppo_explained_final.ipynb`
— same logic, same hyperparameters, same measurements, split into modules you
can run from a terminal.

**There is no dataset.** Nobody ever tells the model a correct move. It plays
games, gets `+1 win / 0 draw / −1 loss` at the end, and that outcome is spread
back over its own decisions:

```
play a game → record each decision → game ends: +1 / 0 / −1
            → discount that score back over the decisions → PPO update → discard → repeat
```

There *is* a perfect minimax solver in `game.py`, used for exactly two things:
as a strong practice partner, and as the grader. It is never a training target.

## Files

| File | What it is |
|---|---|
| `game.py` | Game engine, exact minimax solver, opponents, the text prompt, and full enumeration of all 4,520 decision positions. No torch — run it directly to see the task and the exact reference numbers |
| `model.py` | `Config`, the tokenizer + nine answer tokens, the two-headed `Policy` (GPT-2 body → 9 masked logits + a value head), checkpoint save/load |
| `rollout.py` | Parallel-batch self-play, plus both scorers: game-level (win/draw/loss) and position-level (all 4,520 positions, zero sampling noise) |
| `train_ppo.py` | The PPO update and the training loop (critic warm-up + opponent curriculum). Saves `00_untrained` before touching a weight, then `01_ppo` |
| `evaluate.py` | Runs the identical evaluation on both checkpoints and prints before/after tables + an auto-generated verdict |
| `play.py` | Play it in the terminal; shows its per-cell probabilities, its value estimate, and the solver's answer next to them |

## How to run

This folder reuses the parent `finetune-demo` `uv` project — no separate setup.

```bash
cd finetune-demo
uv sync                      # one-time, if you haven't already

cd tictactoe-ppo
```

**0. See the task itself** (a few seconds, no model downloaded):

```bash
uv run --project .. python game.py
```

Prints a position with its solver scores, the exact prompt GPT-2 sees, and the
reference points every later number is judged against — a random player wins
0.437 against a random opponent, a perfect player wins 0.901 and never loses.

**1. Train.** GPT-2 (124M) is downloaded from Hugging Face on first run.

```bash
# GPU (Colab T4 or better): the real run — 150 iters x 128 games, ~70 min
uv run --project .. python train_ppo.py

# CPU: the sized-down preset — 30 iters x 16 games, first 8 blocks frozen
uv run --project .. python train_ppo.py --quick

# or set it yourself
uv run --project .. python train_ppo.py --iters 150 --games-per-iter 128 --plot
```

Watch these columns as it runs: `loss_rate` should fall; `vloss` should drop
below ~0.4 during the warm-up iterations; `gn_pol` should sit near or below 1.0
(much higher means the policy update is being clipped hard and learning will be
slow); `entropy` should decline gradually, not crash. `reward` looks flat **by
design** — the opponent gets harder at the same rate the model gets better.

Produces `checkpoints/00_untrained/` and `checkpoints/01_ppo/` (each ~500 MB:
this is a full model, not a LoRA adapter), plus `checkpoints/01_ppo/history.json`.
`--plot` also writes `training_curves.png` (needs `matplotlib`, which is not a
project dependency — `uv run --project .. --with matplotlib python train_ppo.py --plot`).

**2. Measure before vs. after:**

```bash
uv run --project .. python evaluate.py                  # both default checkpoints
uv run --project .. python evaluate.py --n-games 400 --device cpu
```

Runs the *same* evaluation on both checkpoints — same games, same opponents,
same position sweep — so the two columns are directly comparable. Prints the
game-outcome table (with 95% Wilson intervals and exact random/perfect
reference rows), the move-quality table over all 4,520 positions, a per-ply
breakdown, and a verdict generated from the measured numbers. Writes
`results/before_after.json`.

On CPU the exhaustive sweep is the slow part (4,520 forward passes per
checkpoint, ~10-20 min each); on GPU it's under a minute.

**3. Play against it:**

```bash
uv run --project .. python play.py                       # you vs the trained model
uv run --project .. python play.py --opponent untrained  # the "before" model, for contrast
uv run --project .. python play.py --mark O              # let it open
uv run --project .. python play.py --inspect             # probe positions, no game
```

`--inspect` takes a board as 9 characters and the side to move (`X..OO.... X`)
and prints the model's probability for every empty cell, its value estimate,
and which cells the solver considers correct — the fastest way to see *what*
it learned rather than just *whether* it learned.

## What the measurements mean

**Game-level** (N games vs a random opponent) is what people intuitively want,
but it saturates: an agent that can't defend still beats a random opponent most
of the time.

**Position-level** is the informative one. Every position in the game is shown
to the model and graded against the solver:

- **top-1** — its best move is a correct move. Chance = 0.494
- **blunder** — it throws away a position that was *not* already lost. Chance = 0.340, lower is better
- **top-1 forced** — restricted to the 3,142 positions with exactly one right answer

No sampling, no noise — a luxury you never get on a real task, which is exactly
why tic-tac-toe is the right toy for showing what RL did and didn't teach.

## Two design details worth knowing

**Illegal moves are impossible, not penalised.** The policy head reads only the
9 logits for the digit tokens `0`-`8`, then sets occupied cells to `-inf`. Their
probability is exactly zero, so every gradient step is spent on move *quality*
instead of on learning the output format.

**The prompt stops mid-sentence**, right after `Best cell:`. A language model
predicts what comes next — cut it off there and the next token it wants to
produce *is* the move.
