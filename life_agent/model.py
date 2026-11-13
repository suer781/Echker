"""A compact byte Transformer with function-preserving MLP neurogenesis."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class DynamicBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("activation_ema", torch.ones(config.hidden_dim))
        self.register_buffer("activation_updates", torch.tensor(0, dtype=torch.long))

    @property
    def hidden_dim(self) -> int:
        return self.fc1.out_features

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attended = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0]
        x = x + self.dropout(attended)
        h = self.norm2(x)
        hidden = F.gelu(self.fc1(h))
        if self.training:
            with torch.no_grad():
                current = hidden.detach().abs().mean(dim=(0, 1)).to(self.activation_ema.device)
                rate = 0.05 if self.activation_updates.item() > 0 else 1.0
                self.activation_ema.lerp_(current, rate)
                self.activation_updates.add_(1)
        return x + self.dropout(self.fc2(hidden))

    def importance(self) -> torch.Tensor:
        outgoing = self.fc2.weight.detach().abs().norm(dim=0)
        incoming = self.fc1.weight.detach().abs().norm(dim=1)
        activity = self.activation_ema.detach().clamp_min(1e-6)
        return outgoing * activity + 1e-4 * incoming

    def _replacement_linear(self, old: nn.Linear, in_features: int, out_features: int) -> nn.Linear:
        return nn.Linear(
            in_features,
            out_features,
            bias=old.bias is not None,
            device=old.weight.device,
            dtype=old.weight.dtype,
        )

    def resize(self, new_dim: int) -> dict[str, Any]:
        old_dim = self.hidden_dim
        if new_dim == old_dim:
            return {"old_dim": old_dim, "new_dim": new_dim, "mode": "none"}
        if new_dim < 1:
            raise ValueError("hidden dimension must remain positive")
        growing = new_dim > old_dim
        importance = self.importance()
        if growing:
            # Split high-importance neurons. Halving both branches' outgoing
            # weights preserves their summed contribution (apart from 1e-5 noise).
            count = new_dim - old_dim
            source = importance.topk(min(count, old_dim)).indices
            if count > old_dim:
                source = source.repeat(math.ceil(count / old_dim))[:count]
            new_fc1 = self._replacement_linear(self.fc1, self.fc1.in_features, new_dim)
            new_fc2 = self._replacement_linear(self.fc2, new_dim, self.fc2.out_features)
            with torch.no_grad():
                new_fc1.weight[:old_dim].copy_(self.fc1.weight)
                new_fc2.weight[:, :old_dim].copy_(self.fc2.weight)
                if self.fc1.bias is not None:
                    new_fc1.bias[:old_dim].copy_(self.fc1.bias)
                if self.fc2.bias is not None:
                    new_fc2.bias.copy_(self.fc2.bias)
                for offset, index in enumerate(source.tolist()):
                    target = old_dim + offset
                    new_fc1.weight[target].copy_(self.fc1.weight[index])
                    new_fc1.weight[target].add_(torch.randn_like(new_fc1.weight[target]) * 1e-5)
                    if self.fc1.bias is not None:
                        new_fc1.bias[target].copy_(self.fc1.bias[index])
                    new_fc2.weight[:, target].copy_(self.fc2.weight[:, index])
                    new_fc2.weight[:, index].mul_(0.5)
                    new_fc2.weight[:, target].mul_(0.5)
                new_activity = torch.cat((self.activation_ema, self.activation_ema[source]))
        else:
            keep = importance.topk(new_dim, sorted=False).indices.sort().values
            new_fc1 = self._replacement_linear(self.fc1, self.fc1.in_features, new_dim)
            new_fc2 = self._replacement_linear(self.fc2, new_dim, self.fc2.out_features)
            with torch.no_grad():
                new_fc1.weight.copy_(self.fc1.weight[keep])
                new_fc2.weight.copy_(self.fc2.weight[:, keep])
                if self.fc1.bias is not None:
                    new_fc1.bias.copy_(self.fc1.bias[keep])
                if self.fc2.bias is not None:
                    new_fc2.bias.copy_(self.fc2.bias)
                new_activity = self.activation_ema[keep].clone()
        self.fc1 = new_fc1
        self.fc2 = new_fc2
        self.register_buffer("activation_ema", new_activity.to(new_fc1.weight.device))
        return {"old_dim": old_dim, "new_dim": new_dim, "mode": "grow" if growing else "prune"}


class DynamicByteLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(DynamicBlock(config) for _ in range(config.n_layers))
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        length = token_ids.shape[1]
        if length > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured context length")
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=token_ids.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.lm_head(self.norm(x))

    def resize_hidden(self, layer: int, new_dim: int) -> dict[str, Any]:
        new_dim = max(self.config.min_hidden_dim, min(self.config.max_hidden_dim, new_dim))
        return {"layer": layer, **self.blocks[layer].resize(new_dim)}

    def hidden_dims(self) -> list[int]:
        return [block.hidden_dim for block in self.blocks]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
