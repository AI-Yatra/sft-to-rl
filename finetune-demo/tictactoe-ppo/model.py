"""
The policy: GPT-2 with two heads, plus config and checkpointing.

Policy head -- GPT-2's own output layer, but we read only the 9 rows that
belong to the digit tokens 0-8 instead of all 50,257. Occupied cells are then
set to -inf, so their probability is exactly zero: the model CANNOT produce an
illegal move, which means all of training is spent on move quality rather than
on learning the output format.

Value head -- a small new network predicting "how well is this position going
to go for me?". PPO needs it to judge whether an outcome beat expectations. It
starts random, hence its own higher learning rate and a warm-up phase.
"""
import json
import os
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from game import PROMPT_VERSION


def pick_device(name="auto"):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    model_name: str = "gpt2"              # 124M
    seed: int = 0
    # --- training
    iters: int = 150
    games_per_iter: int = 128             # 64 was too noisy -> unusable advantages
    ppo_epochs: int = 4
    minibatch: int = 64
    lr: float = 3e-5
    value_lr: float = 3e-4
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    gamma: float = 0.98
    max_grad_norm: float = 1.0            # policy only
    value_max_grad_norm: float = 10.0     # the critic needs its own budget
    critic_warmup: int = 10               # fit the critic before moving the policy
    random_open_prob: float = 0.5         # start some games from a random opening
    # --- opponent curriculum
    opp_random_start: float = 1.0
    opp_random_end: float = 0.2
    opp_anneal_frac: float = 0.6
    opp_eps: float = 0.15
    # --- evaluation
    n_games: int = 400                    # N in "test with N games"
    eval_every: int = 25
    freeze_below: int = 0                 # freeze the first N blocks (raise this on CPU)


class Tokens:
    """Tokenizer + the nine answer tokens, resolved rather than hardcoded.

    The model can emit ~50,000 tokens; we only ever look at nine. Tokenizers
    differ, so we try " 0" first (leading space -- how GPT-2 usually spells a
    digit mid-sentence) and fall back to "0". The winning scheme is recorded in
    the checkpoint so evaluation later uses exactly the same tokens.
    """

    def __init__(self, model_name):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"   # keeps the decision position at index -1 for every prompt
        self.scheme, self.ids = self._resolve()
        self.action_tokens = torch.tensor(self.ids)

    def _resolve(self):
        for scheme in (" {}", "{}"):
            ids = [self.tok.encode(scheme.format(i), add_special_tokens=False) for i in range(9)]
            if all(len(t) == 1 for t in ids):
                return scheme, [t[0] for t in ids]
        raise RuntimeError("no single-token scheme found for digits 0-8")

    def encode(self, prompts, device):
        enc = self.tok(prompts, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        # GPT-2 does NOT derive position ids from the mask; with left padding we
        # must build them, or a padded prompt believes it starts partway through
        # a document.
        pos = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
        return ids, mask, pos


class Policy(nn.Module):
    def __init__(self, cfg, tokens):
        super().__init__()
        self.cfg = cfg
        self.tokens = tokens
        self.lm = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        self.lm.config.pad_token_id = tokens.tok.pad_token_id
        n_embd = self.lm.config.hidden_size
        self.value_head = nn.Sequential(nn.Linear(n_embd, n_embd), nn.Tanh(), nn.Linear(n_embd, 1))
        nn.init.normal_(self.value_head[-1].weight, std=0.01)
        nn.init.zeros_(self.value_head[-1].bias)
        self.register_buffer("action_tokens", tokens.action_tokens.clone(), persistent=False)
        blocks = (self.lm.base_model.h if hasattr(self.lm.base_model, "h")
                  else self.lm.base_model.layers)
        for i, block in enumerate(blocks):
            if i < cfg.freeze_below:
                for p in block.parameters():
                    p.requires_grad_(False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, ids, attn, pos, legal):
        h = self.lm.base_model(input_ids=ids, attention_mask=attn,
                               position_ids=pos).last_hidden_state[:, -1]
        logits = h @ self.lm.get_output_embeddings().weight[self.action_tokens].t()
        logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        return logits, self.value_head(h).squeeze(-1)


def make_optimizer(policy, cfg):
    """Two param groups: backbone at `lr`, value head at the higher `value_lr`."""
    backbone = [p for n, p in policy.named_parameters()
                if p.requires_grad and not n.startswith("value_head")]
    value = list(policy.value_head.parameters())
    optim = torch.optim.AdamW([
        {"params": backbone, "lr": cfg.lr},
        {"params": value, "lr": cfg.value_lr},
    ], weight_decay=0.0)
    return optim, backbone, value


def build_policy(cfg, device):
    tokens = Tokens(cfg.model_name)
    return Policy(cfg, tokens).to(device), tokens


def save_checkpoint(policy, path, **meta):
    os.makedirs(path, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(path, "model.pt"))
    manifest = dict(base_model=policy.cfg.model_name, prompt_version=PROMPT_VERSION,
                    action_scheme=policy.tokens.scheme, seed=policy.cfg.seed,
                    config=asdict(policy.cfg), **meta)
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def load_checkpoint(path, device, tokens=None):
    """Rebuild the exact policy that was saved, refusing incomparable checkpoints."""
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    assert man["prompt_version"] == PROMPT_VERSION, \
        f"prompt changed ({man['prompt_version']} -> {PROMPT_VERSION}) - checkpoint is incomparable"
    cfg = Config(**{k: v for k, v in man["config"].items() if k in Config.__dataclass_fields__})
    tokens = tokens or Tokens(cfg.model_name)
    assert man["action_scheme"] == tokens.scheme, "action token scheme changed"
    policy = Policy(cfg, tokens).to(device)
    policy.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=device))
    policy.eval()
    return policy, man
