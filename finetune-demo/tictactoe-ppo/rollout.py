"""
Playing games (in parallel) and scoring the result.

Games are played in parallel batches, not one at a time: every game waiting on
the model has its prompt encoded together and one forward pass produces every
move. That is the difference between minutes and hours.

Two kinds of test:
  game-level     -- play N complete games, count wins/draws/losses. What people
                    intuitively want, but it saturates (an agent that can't
                    defend still beats a random opponent most of the time).
  position-level -- show the model all 4,520 positions and check its move
                    against the solver. No sampling noise, and it pinpoints
                    WHICH skills are missing. The more informative measurement.
"""
import math
import random
import time

import numpy as np
import torch

from game import (EMPTY, EMPTY_BOARD, X, O, build_prompt, chance_baselines, decision_states,
                  eps_minimax, is_blunder, is_terminal, legal_mask, legal_moves, optimal_moves,
                  other, outcome_reward, place, random_opponent)


@torch.no_grad()
def act(policy, boards, marks, greedy=False):
    prompts = [build_prompt(b, m) for b, m in zip(boards, marks)]
    ids, attn, pos = policy.tokens.encode(prompts, policy.device)
    legal = torch.from_numpy(np.stack([legal_mask(b) for b in boards])).to(policy.device)
    logits, value = policy(ids, attn, pos, legal)
    dist = torch.distributions.Categorical(logits=logits)
    action = logits.argmax(-1) if greedy else dist.sample()
    return prompts, action, dist.log_prob(action), value, legal


def _random_opening(rng, max_plies=2):
    board = EMPTY_BOARD
    for _ in range(rng.choice([1, 2]) if max_plies else 0):
        if is_terminal(board):
            break
        to_move = X if board.count(EMPTY) % 2 == 1 else O
        board = place(board, rng.choice(legal_moves(board)), to_move)
    return board


def play_batch(policy, n_games, opponent, greedy=False, collect=True, random_open=0.0, gamma=0.98):
    """Plays n_games in parallel. Agent is X in half of them, O in the other half.

    For each of the model's own decisions we record the four things PPO needs:
    the prompt, the move, how confident it was (logp), and what it expected (value).
    """
    rng = random
    boards = [_random_opening(rng) if rng.random() < random_open else EMPTY_BOARD
              for _ in range(n_games)]
    to_move = [X if b.count(EMPTY) % 2 == 1 else O for b in boards]
    agent_mark = [X if g % 2 == 0 else O for g in range(n_games)]
    steps = [[] for _ in range(n_games)]
    live = list(range(n_games))

    while live:
        for g in list(live):                      # let the opponent move until it's our turn
            while not is_terminal(boards[g]) and to_move[g] != agent_mark[g]:
                boards[g] = place(boards[g], opponent(boards[g], to_move[g]), to_move[g])
                to_move[g] = other(to_move[g])
            if is_terminal(boards[g]):
                live.remove(g)
        if not live:
            break

        prompts, actions, logps, values, legals = act(
            policy, [boards[g] for g in live], [agent_mark[g] for g in live], greedy=greedy)
        for i, g in enumerate(live):
            a = int(actions[i])
            if collect:
                steps[g].append(dict(prompt=prompts[i], action=a, logp=float(logps[i]),
                                     value=float(values[i]), legal=legals[i].cpu().numpy()))
            boards[g] = place(boards[g], a, agent_mark[g])
            to_move[g] = other(to_move[g])
        live = [g for g in live if not is_terminal(boards[g])]

    buffer, results = [], []
    for g in range(n_games):
        r = outcome_reward(boards[g], agent_mark[g])
        results.append(r)
        T = len(steps[g])
        for t, s in enumerate(steps[g]):          # discount the outcome back over our own moves
            s["ret"] = r * (gamma ** (T - 1 - t))
            buffer.append(s)
    return buffer, np.array(results)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def two_prop_z(k1, n1, k2, n2):
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (k1 / n1 - k2 / n2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@torch.no_grad()
def play_and_score(policy, n_games, opponent, greedy=True):
    was = policy.training
    policy.eval()
    _, res = play_batch(policy, n_games, opponent, greedy=greedy, collect=False)
    if was:
        policy.train()
    return dict(n=len(res), win=int((res > 0).sum()),
                draw=int((res == 0).sum()), loss=int((res < 0).sum()))


@torch.no_grad()
def exhaustive_score(policy, states=None, batch=256):
    """Grade the model on EVERY position in the game. No sampling noise.

    top-1       -- fraction where its best move is a correct move (chance 0.494)
    blunder     -- fraction where it throws away a position that was not already
                   lost (chance 0.340, lower is better)
    top1_forced -- restricted to positions with exactly one right answer
    """
    states = states or decision_states()
    was = policy.training
    policy.eval()
    hits = blunders = forced_n = forced_hits = 0
    per_ply = {}
    for i in range(0, len(states), batch):
        chunk = states[i:i + batch]
        ids, attn, pos = policy.tokens.encode([build_prompt(b, m) for b, m in chunk], policy.device)
        legal = torch.from_numpy(np.stack([legal_mask(b) for b, _ in chunk])).to(policy.device)
        logits, _ = policy(ids, attn, pos, legal)
        for (b, m), a in zip(chunk, logits.argmax(-1).tolist()):
            opt = optimal_moves(b, m)
            ok = a in opt
            hits += ok
            blunders += is_blunder(b, m, a)
            if len(opt) == 1:
                forced_n += 1
                forced_hits += ok
            ply = 9 - b.count(EMPTY)
            d = per_ply.setdefault(ply, [0, 0])
            d[0] += ok
            d[1] += 1
    if was:
        policy.train()
    n = len(states)
    return dict(top1=hits / n, blunder=blunders / n, top1_forced=forced_hits / max(1, forced_n),
                n=n, per_ply={k: v[0] / v[1] for k, v in sorted(per_ply.items())})


def full_evaluation(policy, label, n_games=400, opp_eps=0.15):
    """The one evaluation used both before and after training -- identical both times."""
    t0 = time.time()
    out = dict(label=label,
               vs_random=play_and_score(policy, n_games, random_opponent),
               vs_strong=play_and_score(policy, n_games, eps_minimax(opp_eps)),
               positions=exhaustive_score(policy))
    chance_top1, chance_blunder = chance_baselines()
    r, s, p = out["vs_random"], out["vs_strong"], out["positions"]
    print(f"===== {label} =====")
    print(f"vs RANDOM opponent   (n={r['n']}): win {r['win']/r['n']:.3f}  "
          f"draw {r['draw']/r['n']:.3f}  loss {r['loss']/r['n']:.3f}")
    print(f"vs STRONG opponent   (n={s['n']}): win {s['win']/s['n']:.3f}  "
          f"draw {s['draw']/s['n']:.3f}  loss {s['loss']/s['n']:.3f}")
    print(f"all {p['n']} positions      : top-1 {p['top1']:.3f} (chance {chance_top1:.3f})  "
          f"blunder {p['blunder']:.3f} (chance {chance_blunder:.3f})  "
          f"top-1 forced {p['top1_forced']:.3f}")
    print(f"({time.time()-t0:.0f}s)")
    return out
