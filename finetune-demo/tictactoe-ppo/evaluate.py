"""
Before vs. after: the same evaluation run on the untrained checkpoint and on
the PPO-trained one, then printed side by side.

The reference rows are EXACT -- computed from the game tree, not measured. They
are what make the numbers mean anything: a random player wins 43.7% against a
random opponent, a perfect player wins 90.1% and never loses. Square brackets
are 95% Wilson confidence intervals.

Usage (from finetune-demo/tictactoe-ppo/):
    uv run --project .. python evaluate.py
    uv run --project .. python evaluate.py --before checkpoints/00_untrained \
                                           --after  checkpoints/01_ppo --n-games 400
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from game import chance_baselines, exact_reference
from model import load_checkpoint, pick_device
from rollout import full_evaluation, two_prop_z, wilson

BAR = "-" * 74


def _pct(k, n):
    return k / n


def outcomes_table(results, ref_random, ref_perfect):
    print("\nGame outcomes vs a random opponent")
    print(BAR)
    print(f"{'agent':<26}{'games':>8}{'win':>13}{'draw':>13}{'loss':>13}")
    print(BAR)
    print(f"{'Random play (floor)':<26}{'exact':>8}"
          f"{ref_random[0]:>13.3f}{ref_random[1]:>13.3f}{ref_random[2]:>13.3f}")
    for key, label in [("before", "BEFORE training"), ("after", "AFTER PPO")]:
        c = results[key]["vs_random"]
        n = c["n"]
        print(f"{label:<26}{n:>8}"
              f"{_pct(c['win'], n):>13.3f}{_pct(c['draw'], n):>13.3f}{_pct(c['loss'], n):>13.3f}")
        cis = "".join(f"{'[%.3f,%.3f]' % wilson(c[k], n):>13}" for k in ("win", "draw", "loss"))
        print(f"{'  95% CI':<26}{'':>8}{cis}")
    print(f"{'Perfect play (ceiling)':<26}{'exact':>8}"
          f"{ref_perfect[0]:>13.3f}{ref_perfect[1]:>13.3f}{ref_perfect[2]:>13.3f}")
    print(BAR)


def positions_table(results, chance_top1, chance_blunder):
    b, a = results["before"]["positions"], results["after"]["positions"]
    print(f"\nMove quality on all {b['n']} positions in the game (zero sampling noise)")
    print(BAR)
    print(f"{'metric':<34}{'chance':>12}{'before':>13}{'after':>13}")
    print(BAR)
    for key, label, chance in [("top1", "top-1 correct move", chance_top1),
                               ("top1_forced", "top-1 on forced positions", None),
                               ("blunder", "blunder rate (lower better)", chance_blunder)]:
        ch = f"{chance:.3f}" if chance is not None else "-"
        print(f"{label:<34}{ch:>12}{b[key]:>13.3f}{a[key]:>13.3f}")
    print(BAR)
    print("\ntop-1 correct by pieces on the board (early = strategic, late = forced)")
    for ply in sorted(a["per_ply"]):
        bb, aa = b["per_ply"][ply], a["per_ply"][ply]
        bar = "#" * int(aa * 40)
        print(f"  {ply} pieces  before {bb:.3f}  after {aa:.3f}  |{bar:<40}|")


def verdict(results, cfg_meta, ref_random, ref_perfect, chance_top1, chance_blunder, n_games):
    """Auto-generated from the measured numbers, so it cannot drift from them."""
    b, a = results["before"], results["after"]
    bl, al = _pct(b["vs_random"]["loss"], b["vs_random"]["n"]), _pct(a["vs_random"]["loss"], a["vs_random"]["n"])
    bw, aw = _pct(b["vs_random"]["win"], b["vs_random"]["n"]), _pct(a["vs_random"]["win"], a["vs_random"]["n"])
    zl, pl = two_prop_z(b["vs_random"]["loss"], b["vs_random"]["n"],
                        a["vs_random"]["loss"], a["vs_random"]["n"])
    gap_loss = (bl - al) / bl if bl > 0 else float("nan")
    gap_win = (aw - bw) / (ref_perfect[0] - bw) if ref_perfect[0] > bw else float("nan")
    learned = al < bl and a["positions"]["top1"] > b["positions"]["top1"]

    print(f"""
CONCLUSION
==========
Setup      : {cfg_meta.get('base_model')}, {cfg_meta.get('games_seen')} games played,
             reward = game outcome only (no move labels were ever shown to the model).

Game play  : loss rate vs a random opponent {bl:.3f} -> {al:.3f}
             ({gap_loss:.0%} of the gap to zero closed; z={zl:+.2f}, p={pl:.2e}, n={n_games} each)
             win rate {bw:.3f} -> {aw:.3f}, against a random floor of {ref_random[0]:.3f}
             and a perfect ceiling of {ref_perfect[0]:.3f} ({gap_win:.0%} of that gap closed)

Move quality (all {b['positions']['n']} positions, zero sampling noise):
             top-1 correct    {b['positions']['top1']:.3f} -> {a['positions']['top1']:.3f}   (chance {chance_top1:.3f})
             forced positions {b['positions']['top1_forced']:.3f} -> {a['positions']['top1_forced']:.3f}
             blunder rate     {b['positions']['blunder']:.3f} -> {a['positions']['blunder']:.3f}   (chance {chance_blunder:.3f}, lower better)

Read it as : the model {'DID learn' if learned else 'did NOT clearly learn'} to play.
             Any remaining loss against a *random* opponent means positions where it
             still fails to block a threat or to take a win that was on the board.""")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default="checkpoints/00_untrained")
    ap.add_argument("--after", default="checkpoints/01_ppo")
    ap.add_argument("--n-games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default="results/before_after.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    ref_random, ref_perfect = exact_reference("random"), exact_reference("perfect")
    chance_top1, chance_blunder = chance_baselines()

    results = {}
    tokens = None
    meta = {}
    for key, path, label in [("before", args.before, "BEFORE TRAINING (untrained GPT-2)"),
                             ("after", args.after, "AFTER PPO TRAINING")]:
        policy, man = load_checkpoint(path, device, tokens)
        tokens = policy.tokens          # reuse -- same base model, same answer tokens
        if key == "after":
            meta = man
        # Same seed before each run: the two columns face the same random
        # opponents, so the comparison is not a lottery.
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        results[key] = full_evaluation(policy, label, n_games=args.n_games)
        del policy

    outcomes_table(results, ref_random, ref_perfect)
    positions_table(results, chance_top1, chance_blunder)
    verdict(results, meta, ref_random, ref_perfect, chance_top1, chance_blunder, args.n_games)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(results=results, n_games=args.n_games,
                       ref_random=ref_random, ref_perfect=ref_perfect,
                       chance_top1=chance_top1, chance_blunder=chance_blunder,
                       before_ckpt=args.before, after_ckpt=args.after), f, indent=2)
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
