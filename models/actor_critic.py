import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.pi_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)

        nn.init.orthogonal_(self.pi_head.weight, gain=0.01)
        nn.init.constant_(self.pi_head.bias, 0.0)

    def forward(self, obs):
        h = self.body(obs)
        return Categorical(logits=self.pi_head(h)), self.v_head(h).squeeze(-1)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    returns = adv + values
    return adv, returns


class TokenActorCritic(nn.Module):
    """Actor-critic with its own transformer over raw tokens, separate from the world model.

    Input: (B, T) token ids, left-padded with `pad_id`. Heads read the last hidden
    state and the mean hidden state of the last `obs_len` tokens (the current
    observation), like the world-model policy's "both" features.
    """

    def __init__(self, config, vocab_size, pad_id, obs_len, n_actions, hidden=256):
        super().__init__()
        from .transformer import Transformer

        self.config, self.pad_id, self.obs_len = config, pad_id, obs_len
        self.embed = nn.Embedding(vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        self.backbone = Transformer(config)
        for module in (self.embed, self.pos_emb, self.backbone):
            module.apply(self._init_weights)
        self.body = nn.Sequential(
            nn.Linear(2 * config.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.pi_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)

        nn.init.orthogonal_(self.pi_head.weight, gain=0.01)
        nn.init.constant_(self.pi_head.bias, 0.0)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, tokens):
        tokens = tokens[:, -self.config.max_tokens :]
        T = tokens.size(1)
        # <PAD> only ever appears as left padding.
        pad = (tokens == self.pad_id).sum(dim=1)
        positions = torch.arange(T, device=tokens.device)
        key_valid = None
        if pad.any():
            key_valid = positions >= pad[:, None]
            positions = (positions - pad[:, None]).clamp(min=0)
        h = self.backbone(self.embed(tokens) + self.pos_emb(positions), None, key_valid)
        features = torch.cat((h[:, -1], h[:, -self.obs_len :].mean(dim=1)), dim=-1)
        body = self.body(features)
        return Categorical(logits=self.pi_head(body)), self.v_head(body).squeeze(-1)
