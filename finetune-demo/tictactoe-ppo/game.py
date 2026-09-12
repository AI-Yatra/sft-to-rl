"""
Tic-tac-toe: game engine, exact solver, practice opponents, and the text
format the model reads. Pure Python + numpy -- no torch, no model here.

There is NO dataset of correct moves anywhere in this demo. The solver in
this file is used for two things only: as a strong practice partner, and as
the grader. It is never a training target -- the only signal PPO ever sees
is the game outcome (+1 win / 0 draw / -1 loss).
"""
import math
import random
from functools import lru_cache

import numpy as np

EMPTY, X, O = 0, 1, 2
SYM = {EMPTY: ".", X: "X", O: "O"}
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
EMPTY_BOARD = (EMPTY,) * 9

# Bump this if build_prompt() changes -- old checkpoints become incomparable.
PROMPT_VERSION = "v1"


# --------------------------------------------------------------------------
# 1. The game engine
#
# A board is a tuple of 9 ints, left-to-right, top-to-bottom. Tuples because
# they are immutable and hashable: lets us memoise the solver and enumerate
# states without accidental mutation bugs.
# --------------------------------------------------------------------------
def other(mark):
    return O if mark == X else X


def winner(board):
    """Returns X, O, or EMPTY if nobody has three in a row."""
    for i, j, k in LINES:
        if board[i] != EMPTY and board[i] == board[j] == board[k]:
            return board[i]
    return EMPTY


def legal_moves(board):
    return [i for i in range(9) if board[i] == EMPTY]


def is_terminal(board):
    return winner(board) != EMPTY or EMPTY not in board


def place(board, cell, mark):
    assert board[cell] == EMPTY, f"illegal move {cell} on {board}"
    b = list(board)
    b[cell] = mark
    return tuple(b)


def render(board):
    return "\n".join(" ".join(SYM[board[r * 3 + c]] for c in range(3)) for r in range(3))


def outcome_reward(board, mark):
    """The only reward signal in this demo: +1 win, 0 draw, -1 loss, from `mark`'s view."""
    w = winner(board)
    if w == EMPTY:
        return 0.0
    return 1.0 if w == mark else -1.0


# --------------------------------------------------------------------------
# 2. The exact solver (minimax)
#
# Tic-tac-toe is small enough to solve perfectly, so we always know the right
# answer. Scores decay by 0.99 per ply so quick wins outrank slow ones.
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _minimax(board, to_move):
    """(value for X, best move). Memoised, so the whole tree is solved once."""
    w = winner(board)
    if w != EMPTY:
        return (1.0 if w == X else -1.0), -1
    moves = legal_moves(board)
    if not moves:
        return 0.0, -1
    best_score, best_move = (-math.inf if to_move == X else math.inf), moves[0]
    for m in moves:
        s, _ = _minimax(place(board, m, to_move), other(to_move))
        s *= 0.99
        if (to_move == X and s > best_score) or (to_move == O and s < best_score):
            best_score, best_move = s, m
    return best_score, best_move


def move_scores(board, mark):
    """Value of each legal move, always expressed from X's point of view."""
    return {c: _minimax(place(board, c, mark), other(mark))[0] * 0.99
            for c in legal_moves(board)}


def optimal_moves(board, mark, tol=1e-9):
    s = move_scores(board, mark)
    best = max(s.values()) if mark == X else min(s.values())
    return [c for c, v in s.items() if abs(v - best) <= tol]


def position_is_lost(board, mark):
    s = move_scores(board, mark)
    best = max(s.values()) if mark == X else min(s.values())
    return (best < -0.5) if mark == X else (best > 0.5)


def is_blunder(board, mark, move):
    """True if `move` loses a position that was NOT already lost. The sharpest error measure."""
    if position_is_lost(board, mark):
        return False
    v = move_scores(board, mark)[move]
    return (v < -0.5) if mark == X else (v > 0.5)


# --------------------------------------------------------------------------
# 3. Practice partners
#
# `eps` adds a little randomness even to the perfect opponent: a deterministic
# opponent against a greedy model replays the SAME game every time, so 400
# test games would contain exactly 1 distinct game.
# --------------------------------------------------------------------------
def random_opponent(board, mark):
    return random.choice(legal_moves(board))


def minimax_opponent(board, mark, eps=0.0):
    if eps and random.random() < eps:
        return random.choice(legal_moves(board))
    return _minimax(board, mark)[1]


def mixed_opponent(random_prob, eps=0.0):
    def move(board, mark):
        if random.random() < random_prob:
            return random_opponent(board, mark)
        return minimax_opponent(board, mark, eps=eps)
    return move


def eps_minimax(eps=0.15):
    return lambda b, m: minimax_opponent(b, m, eps=eps)


# --------------------------------------------------------------------------
# 4. Board -> text
#
# Two deliberate details: the prompt stops mid-sentence right after
# "Best cell:", so the next token the model wants to produce IS the move; and
# legal cells are listed explicitly, so the model can attend to that line
# instead of inferring emptiness from the grid.
# --------------------------------------------------------------------------
def build_prompt(board, mark):
    moves = " ".join(str(m) for m in legal_moves(board))
    return (
        "Tic-tac-toe. Cells are numbered 0-8, left to right, top to bottom.\n"
        f"You are {SYM[mark]}. Opponent is {SYM[other(mark)]}.\n"
        "Board:\n"
        f"{render(board)}\n"
        f"Legal cells: {moves}\n"
        "Best cell:"
    )


def legal_mask(board):
    """Boolean array of length 9: True where a move is allowed."""
    return np.array([board[i] == EMPTY for i in range(9)], dtype=bool)


# --------------------------------------------------------------------------
# 5. Enumerating the whole game
#
# Small enough to grade on EVERY position that exists -- zero sampling noise,
# a luxury you never get on a real task.
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def decision_states():
    """Every (board, to_move) pair where a move is still required. 4,520 of them."""
    seen = set()

    def walk(board, to_move):
        if (board, to_move) in seen:
            return
        seen.add((board, to_move))
        if is_terminal(board):
            return
        for c in legal_moves(board):
            walk(place(board, c, to_move), other(to_move))

    walk(EMPTY_BOARD, X)
    return sorted(s for s in seen if not is_terminal(s[0]))


@lru_cache(maxsize=1)
def chance_baselines():
    """What pure guessing scores: (top-1 correct, blunder rate). ~0.494 / ~0.340."""
    states = decision_states()
    top1 = sum(len(optimal_moves(b, m)) / len(legal_moves(b)) for b, m in states) / len(states)
    blunder = sum(
        0.0 if position_is_lost(b, m) else
        sum(is_blunder(b, m, c) for c in legal_moves(b)) / len(legal_moves(b))
        for b, m in states) / len(states)
    return top1, blunder


@lru_cache(maxsize=None)
def _exact(board, to_move, agent_mark, agent):
    if is_terminal(board):
        w = winner(board)
        if w == EMPTY:
            return (0.0, 1.0, 0.0)
        return (1.0, 0.0, 0.0) if w == agent_mark else (0.0, 0.0, 1.0)
    moves = legal_moves(board)
    if to_move == agent_mark and agent == "perfect":
        moves = [_minimax(board, to_move)[1]]
    res = [_exact(place(board, m, to_move), other(to_move), agent_mark, agent) for m in moves]
    return tuple(sum(r[i] for r in res) / len(res) for i in range(3))


def exact_reference(agent):
    """(win, draw, loss) vs a uniform-random opponent, averaged over playing X and O.

    Computed by full expansion of the game tree -- these are exact, not simulated.
    Without them, a number like "79% wins" means nothing.
    """
    a = _exact(EMPTY_BOARD, X, X, agent)
    b = _exact(EMPTY_BOARD, X, O, agent)
    return tuple((a[i] + b[i]) / 2 for i in range(3))


if __name__ == "__main__":
    b = EMPTY_BOARD
    for c, m in [(0, X), (3, O), (8, X), (4, O)]:
        b = place(b, c, m)
    print(render(b))
    print("scores :", {k: round(v, 2) for k, v in move_scores(b, X).items()})
    print("optimal:", optimal_moves(b, X), "(cell 5 is the only non-losing move)")
    print()
    print(build_prompt(b, X))
    print()
    states = decision_states()
    forced = sum(1 for bb, mm in states if len(optimal_moves(bb, mm)) == 1)
    top1, blunder = chance_baselines()
    print(f"positions where a move is required : {len(states)}")
    print(f"  ... with exactly one correct move: {forced} ({forced / len(states):.1%})")
    print(f"random guessing would score        : top-1 {top1:.3f}, blunder {blunder:.3f}")
    r, p = exact_reference("random"), exact_reference("perfect")
    print(f"random player vs random opponent   : win {r[0]:.3f}  draw {r[1]:.3f}  loss {r[2]:.3f}")
    print(f"perfect player vs random opponent  : win {p[0]:.3f}  draw {p[1]:.3f}  loss {p[2]:.3f}")
