"""
Play against the trained model in the terminal, and inspect what it thinks.

After each of your turns it prints the model's probability for every empty
cell and its value-head estimate for the position -- i.e. what the policy
actually learned, not just the move it picked. The solver hint shows the
correct answer, so you can see exactly where it is wrong.

Usage (from finetune-demo/tictactoe-ppo/):
    uv run --project .. python play.py                       # you vs the trained model
    uv run --project .. python play.py --opponent untrained  # the before model, for contrast
    uv run --project .. python play.py --opponent perfect --no-probs
    uv run --project .. python play.py --mark O              # let the model open
    uv run --project .. python play.py --inspect             # no game, just probe positions
"""
import argparse
import random

import numpy as np
import torch

from game import (EMPTY, EMPTY_BOARD, O, SYM, X, _minimax, build_prompt, is_terminal, legal_mask,
                  legal_moves, optimal_moves, other, place, position_is_lost, winner)
from model import load_checkpoint, pick_device


@torch.no_grad()
def policy_probs(policy, board, mark):
    was = policy.training
    policy.eval()
    ids, attn, pos = policy.tokens.encode([build_prompt(board, mark)], policy.device)
    legal = torch.from_numpy(legal_mask(board)[None]).to(policy.device)
    logits, value = policy(ids, attn, pos, legal)
    p = logits.softmax(-1)[0].float().cpu().numpy()
    if was:
        policy.train()
    return p, float(value)


def show_board(board, last=None):
    print()
    for r in range(3):
        cells = []
        for c in range(3):
            i = r * 3 + c
            s = SYM[board[i]] if board[i] != EMPTY else str(i)
            cells.append(f"[{s}]" if i == last else f" {s} ")
        print("  " + "|".join(cells))
        if r < 2:
            print("  " + "-" * 11)
    print()


def show_thinking(policy, board, mark, show_probs, show_hint):
    if show_probs and policy is not None:
        probs, val = policy_probs(policy, board, mark)
        order = [c for c in np.argsort(-probs) if board[c] == EMPTY][:3]
        print(f"  model: value {val:+.2f} | top cells "
              + ", ".join(f"{c}:{probs[c]:.2f}" for c in order))
    if show_hint:
        opt = optimal_moves(board, mark)
        state = "already lost" if position_is_lost(board, mark) else "you can still draw/win"
        print(f"  solver: best cell(s) {opt} ({state})")


def model_move(policy, board, mark):
    probs, _ = policy_probs(policy, board, mark)
    return int(probs.argmax())


def opponent_move(kind, policy, board, mark):
    if kind == "random":
        return random.choice(legal_moves(board))
    if kind == "perfect":
        if random.random() < 0.15:            # a little noise so games aren't identical
            return random.choice(legal_moves(board))
        return _minimax(board, mark)[1]
    return model_move(policy, board, mark)    # "trained" / "untrained"


def ask_cell(board):
    while True:
        raw = input("your move (cell 0-8, or q to quit): ").strip().lower()
        if raw in ("q", "quit", "exit"):
            return None
        if raw.isdigit() and int(raw) in legal_moves(board):
            return int(raw)
        print(f"  pick one of {legal_moves(board)}")


def play_game(opp_policy, view_policy, kind, human_mark, show_probs, show_hint):
    board, to_move, last = EMPTY_BOARD, X, None
    while True:
        if to_move != human_mark:
            m = opponent_move(kind, opp_policy, board, to_move)
            board, last = place(board, m, to_move), m
            print(f"  opponent ({kind}) plays {m}")
            to_move = other(to_move)
            if is_terminal(board):
                break
            continue

        show_board(board, last)
        show_thinking(view_policy, board, human_mark, show_probs, show_hint)
        cell = ask_cell(board)
        if cell is None:
            return None
        board, last = place(board, cell, human_mark), cell
        to_move = other(to_move)
        if is_terminal(board):
            break

    show_board(board, last)
    w = winner(board)
    print({EMPTY: "Draw.", human_mark: "You win.", other(human_mark): "You lose."}[w])
    return w


def inspect(policy, show_hint=True):
    """Probe the model on positions you type in, no game attached.

    Enter 9 characters (. X O), e.g. `X..OO....`, then the mark to move.
    """
    print("Enter a board as 9 chars of . X O (row by row), then the mark to move.")
    print("Example:  X..OO....  X        (blank line to quit)\n")
    while True:
        raw = input("board mark> ").lstrip("﻿").strip().split()
        if not raw:
            return
        cells, mark = (raw + ["X"])[:2]
        if len(cells) != 9 or any(ch.upper() not in ".XO" for ch in cells):
            print("  need 9 chars of . X O")
            continue
        lut = {".": EMPTY, "X": X, "O": O}
        board = tuple(lut[ch.upper()] for ch in cells)
        mark = X if mark.upper() == "X" else O
        if is_terminal(board):
            print("  that position is already over")
            continue
        show_board(board)
        probs, val = policy_probs(policy, board, mark)
        print(f"  value {val:+.2f}")
        for c in np.argsort(-probs):
            if board[c] == EMPTY:
                star = "*" if c in optimal_moves(board, mark) else " "
                print(f"   {star} cell {c}: {probs[c]:.3f}")
        if show_hint:
            print(f"  (* = solver-optimal: {optimal_moves(board, mark)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/01_ppo")
    ap.add_argument("--untrained", default="checkpoints/00_untrained")
    ap.add_argument("--opponent", default="trained",
                    choices=["trained", "untrained", "perfect", "random"])
    ap.add_argument("--mark", default="X", choices=["X", "O"], help="your mark; O means it opens")
    ap.add_argument("--no-probs", action="store_true", help="hide the model's per-cell probabilities")
    ap.add_argument("--no-hint", action="store_true", help="hide the solver's best move")
    ap.add_argument("--inspect", action="store_true", help="probe positions instead of playing")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = pick_device(args.device)
    path = args.untrained if args.opponent == "untrained" else args.checkpoint
    policy = None
    if args.opponent in ("trained", "untrained") or not args.no_probs or args.inspect:
        policy, man = load_checkpoint(path, device)
        print(f"loaded {path}  (stage={man.get('stage')}, games_seen={man.get('games_seen')})")

    if args.inspect:
        inspect(policy, show_hint=not args.no_hint)
        return

    human_mark = X if args.mark == "X" else O
    tally = {"win": 0, "loss": 0, "draw": 0}
    while True:
        w = play_game(policy, policy, args.opponent, human_mark,
                      show_probs=not args.no_probs, show_hint=not args.no_hint)
        if w is None:
            break
        tally["draw" if w == EMPTY else ("win" if w == human_mark else "loss")] += 1
        print(f"  running score -- you {tally['win']}W {tally['draw']}D {tally['loss']}L")
        if input("\nplay again? [Y/n] ").strip().lower().startswith("n"):
            break


if __name__ == "__main__":
    main()
