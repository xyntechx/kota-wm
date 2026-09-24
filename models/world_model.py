from typing import Optional

import torch
import torch.nn as nn

from .kv_caching import KeysValues
from .transformer import Transformer, TransformerConfig


class WorldModel(nn.Module):
    def __init__(
        self, config: TransformerConfig, vocab_size: int, hidden=256, reward_scale=1.0
    ):
        super().__init__()
        self.config = config
        self.reward_scale = reward_scale
        self.embed = nn.Embedding(vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        self.backbone = Transformer(config)

        # observation head
        self.obs_head = nn.Sequential(
            nn.Linear(config.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, vocab_size),  # output is logits
        )

        # reward head
        self.rew_head = nn.Sequential(
            nn.Linear(config.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),  # output is float, MSE loss optimized
        )

        # termination head
        self.ter_head = nn.Sequential(
            nn.Linear(config.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),  # output is boolean
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @property
    def max_tokens(self):
        return self.config.max_tokens

    def generate_empty_keys_values(self, n: int) -> KeysValues:
        return self.backbone.generate_empty_keys_values(n, self.max_tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        past_keys_values: Optional[KeysValues] = None,
        pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        offset = 0 if past_keys_values is None else past_keys_values.size
        T = tokens.size(1)
        if offset + T > self.max_tokens:
            raise ValueError(
                f"{offset + T} tokens exceed the world model's {self.max_tokens}"
            )
        positions = torch.arange(offset, offset + T, device=tokens.device)
        key_valid = None
        if pad is not None:
            key_valid = torch.arange(offset + T, device=tokens.device) >= pad[:, None]
            positions = (positions - pad[:, None]).clamp(min=0)
        return self.backbone(
            self.embed(tokens) + self.pos_emb(positions), past_keys_values, key_valid
        )

    def outcomes(self, action_hidden: torch.Tensor):
        reward = self.rew_head(action_hidden).squeeze(-1) * self.reward_scale
        return reward, self.ter_head(action_hidden)
