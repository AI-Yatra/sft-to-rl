"""
PPO training: GPT-2 learns tic-tac-toe from game outcomes alone.

What PPO does, in one paragraph: play games; for each move compare what
actually happened to what the value head expected -- the difference is the
surprise (the advantage). Make surprisingly-good moves more likely and
surprisingly-bad ones less likely. The clip stops any single batch from moving
the policy more than +-20% on one move, which is what lets us reuse each batch
of games four times instead of throwing it away after one gradient step.

  loss = policy term + 0.5 * value error - 0.02 * entropy

The entropy bonus keeps the model from committing to one cell before it has
explored.

Usage (from finetune-demo/tictactoe-ppo/):
    uv run --project .. python train_ppo.py --quick        # CPU smoke run, ~15 min
    uv run --project .. python train_ppo.py                # full run, ~70 min on a T4
    uv run --project .. python train_ppo.py --iters 150 --games-per-iter 128

Writes checkpoints/00_untrained (saved before a single weight is touched) and
checkpoints/01_ppo, plus checkpoints/01_ppo/history.json.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import mixed_opponent, random_opponent
from model import Config, build_policy, make_optimizer, pick_device, save_checkpoint
from rollout import exhaustive_score, play_and_score, play_batch


def ppo_update(policy, optim, buffer, cfg, pol_params, val_params, train_policy=True):
    prompts = [s["prompt"] for s in buffer]
    device = policy.device
    actions = torch.tensor([s["action"] for s in buffer], device=device)
    old_logp = torch.tensor([s["logp"] for s in buffer], device=device)
    rets = torch.tensor([s["ret"] for s in buffer], device=device, dtype=torch.float32)
    values = torch.tensor([s["value"] for s in buffer], device=device, dtype=torch.float32)
    legal = torch.from_numpy(np.stack([s["legal"] for s in buffer])).to(device)

    adv = rets - values                                   # terminal-only reward => GAE reduces to this
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n, logs = len(buffer), []
    for _ in range(cfg.ppo_epochs):
        for idx in torch.randperm(n, device=device).split(cfg.minibatch):
            ids, attn, pos = policy.tokens.encode([prompts[i] for i in idx.tolist()], device)
            logits, v = policy(ids, attn, pos, legal[idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions[idx])

            ratio = (logp - old_logp[idx]).exp()
            pg = -torch.min(ratio * adv[idx],
                            ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[idx]).mean()
            vloss = F.mse_loss(v, rets[idx])
            ent = dist.entropy().mean()
            loss = (pg if train_policy else pg.detach() * 0) + cfg.vf_coef * vloss - cfg.ent_coef * ent

            optim.zero_grad(set_to_none=True)
            loss.backward()
            # Separate budgets: one global clip lets the fresh critic's large
            # gradient throttle the policy update, which silently stalls learning.
            gn_p = nn.utils.clip_grad_norm_(pol_params, cfg.max_grad_norm)
            gn_v = nn.utils.clip_grad_norm_(val_params, cfg.value_max_grad_norm)
            optim.step()
            logs.append((pg.item(), vloss.item(), ent.item(), gn_p.item(), gn_v.item(),
                         (old_logp[idx] - logp).mean().item()))

    pg, vl, en, gnp, gnv, kl = np.mean(logs, axis=0)
    return dict(pg=pg, vloss=vl, entropy=en, gn_pol=gnp, gn_val=gnv, kl=kl)


def train(policy, optim, pol_params, val_params, cfg, out_dir):
    """Each iteration: play games -> build examples -> four PPO passes -> discard.

    Two scheduled behaviours:
      critic warm-up   -- for the first `critic_warmup` iterations the backbone
                          lr is 0, so only the value head trains. Advantages are
                          meaningless until the critic is roughly calibrated.
      opponent curriculum -- the practice partner goes from 100% random to 20%
                          random over the first 60% of training.

    Watch these columns: `loss_rate` should fall; `vloss` should drop below ~0.4
    during warm-up; `gn_pol` should sit near or below 1.0 (much higher means the
    policy update is being clipped hard and learning will be slow); `entropy`
    should decline gradually, not crash.
    """
    history = []
    t0 = time.time()

    for it in range(1, cfg.iters + 1):
        warming = it <= cfg.critic_warmup
        optim.param_groups[0]["lr"] = 0.0 if warming else cfg.lr

        frac = min(1.0, it / max(1, cfg.iters * cfg.opp_anneal_frac))
        p_rand = cfg.opp_random_start + frac * (cfg.opp_random_end - cfg.opp_random_start)
        opponent = mixed_opponent(p_rand, eps=cfg.opp_eps)

        buffer, res = play_batch(policy, cfg.games_per_iter, opponent,
                                 random_open=cfg.random_open_prob, gamma=cfg.gamma)
        stats = ppo_update(policy, optim, buffer, cfg, pol_params, val_params,
                           train_policy=not warming)
        stats.update(iter=it, phase="warmup" if warming else "ppo", p_rand=round(p_rand, 2),
                     reward=float(res.mean()), win=float((res > 0).mean()),
                     loss_rate=float((res < 0).mean()), games=it * cfg.games_per_iter,
                     mins=round((time.time() - t0) / 60, 1))
        history.append(stats)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()},
              flush=True)

        if cfg.eval_every and it % cfg.eval_every == 0:
            q = play_and_score(policy, 200, random_opponent)
            p = exhaustive_score(policy)
            print(f"   >> vs random: win {q['win']/q['n']:.3f} loss {q['loss']/q['n']:.3f}"
                  f" | positions: top-1 {p['top1']:.3f} blunder {p['blunder']:.3f}", flush=True)

    mins = (time.time() - t0) / 60
    print(f"\ndone: {cfg.iters} iterations, {cfg.iters * cfg.games_per_iter} games, {mins:.1f} min")

    save_checkpoint(policy, out_dir, stage="ppo",
                    games_seen=cfg.iters * cfg.games_per_iter, train_minutes=round(mins, 1))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved -> {out_dir}")
    return history


def plot_history(history, path):
    """Optional (needs matplotlib, not a project dependency).

    `loss_rate` is the honest progress signal. Mean `reward` looks flat by
    design -- the opponent gets harder at the same rate the model gets better.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = [h["games"] for h in history]
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.5))
    for a, key, title in [
        (ax[0][0], "loss_rate", "loss rate (lower = better)"),
        (ax[0][1], "win", "win rate in training games"),
        (ax[0][2], "reward", "mean reward (flat by design)"),
        (ax[1][0], "vloss", "value loss (critic fit)"),
        (ax[1][1], "entropy", "policy entropy"),
        (ax[1][2], "gn_pol", "policy grad norm (vs clip=1.0)"),
    ]:
        a.plot(g, [h[key] for h in history])
        a.set_title(title, fontsize=10)
        a.set_xlabel("games played")
        a.grid(alpha=.3)
    ax[1][2].axhline(1.0, ls="--", c="r", lw=1)
    ax[0][2].plot(g, [h["p_rand"] for h in history], ls=":", c="gray", label="opponent randomness")
    ax[0][2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"curves -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--games-per-iter", type=int, default=None)
    ap.add_argument("--freeze-below", type=int, default=None,
                    help="freeze the first N transformer blocks (use 8 on CPU)")
    ap.add_argument("--eval-every", type=int, default=None, help="0 to disable mid-run evals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default="checkpoints/01_ppo")
    ap.add_argument("--untrained-out", default="checkpoints/00_untrained")
    ap.add_argument("--quick", action="store_true",
                    help="CPU-sized preset: 30 iters x 16 games, first 8 blocks frozen")
    ap.add_argument("--plot", action="store_true", help="also write training_curves.png (matplotlib)")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, seed=args.seed)
    if args.quick:
        cfg.iters, cfg.games_per_iter, cfg.freeze_below = 30, 16, 8
        cfg.critic_warmup, cfg.minibatch, cfg.eval_every, cfg.n_games = 5, 32, 0, 200
    for name, value in [("iters", args.iters), ("games_per_iter", args.games_per_iter),
                        ("freeze_below", args.freeze_below), ("eval_every", args.eval_every)]:
        if value is not None:
            setattr(cfg, name, value)

    device = pick_device(args.device)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    policy, tokens = build_policy(cfg, device)
    optim, pol_params, val_params = make_optimizer(policy, cfg)
    n_all = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"device: {device} | action scheme: {tokens.scheme!r} | token ids: {tokens.ids}")
    print(f"parameters: {n_all/1e6:.2f}M total, {n_train/1e6:.2f}M trainable")
    print(f"config: {cfg.iters} iters x {cfg.games_per_iter} games "
          f"= {cfg.iters * cfg.games_per_iter} games\n")

    # Save the untrained policy BEFORE touching a single weight. "Before RL"
    # later means exactly this network, not a re-initialised one.
    save_checkpoint(policy, args.untrained_out, stage="untrained", games_seen=0)
    print(f"saved -> {args.untrained_out}\n")

    history = train(policy, optim, pol_params, val_params, cfg, args.out)
    if args.plot:
        plot_history(history, os.path.join(args.out, "training_curves.png"))


if __name__ == "__main__":
    main()
