"""ByT5-DNA byte-level causal language model.

Transplanted from Google's ByT5 architecture (Radosavljević et al., 2021)
at the project's own scale, keeping every architecture contract:

- vocabulary stays 262 (0..255 raw bytes, 256..261 control tokens);
- one model, one objective: causal next-byte cross-entropy;
- hidden width stays growable/prunable (DynamicBlock.resize carries the
  neuron correspondence row_map/keep_map used by the momentum-preserving
  optimizer rebuild);
- RMSNorm pre-norm, gated-GELU feed-forward and T5 relative attention bias
  replace learned absolute positions, so the context ceiling is no longer
  baked into a position embedding (max_seq_len remains a batching cap).

What is deliberately NOT transplanted: byT5's pre-trained weights (they
cannot fit this machine's GPU budget) and the encoder-decoder span-corruption
objective (it would break the unified next-byte objective).
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig

NUM_RELATIVE_BUCKETS = 32
MAX_RELATIVE_DISTANCE = 128


class T5LayerNorm(nn.Module):
    """HF T5LayerNorm: RMS norm with clamped variance and no mean centering."""

    def __init__(self, hidden_size: int, variance_epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        variance = torch.clamp(variance, min=self.variance_epsilon)
        normalized = hidden_states / torch.sqrt(variance).to(hidden_states.dtype)
        return self.weight * normalized


def relative_position_bucket(length: int, device: torch.device) -> torch.Tensor:
    """T5 causal (bidirectional=False) bucketing of key age relative to query."""
    context = torch.arange(length, device=device).unsqueeze(1)
    memory = torch.arange(length, device=device).unsqueeze(0)
    relative = memory - context
    relative = -torch.min(relative, torch.zeros_like(relative))
    num_buckets = NUM_RELATIVE_BUCKETS
    max_exact = num_buckets // 2
    is_small = relative < max_exact
    scaled = max_exact + (
        torch.log(relative.float().clamp(min=1) / max_exact)
        / math.log(MAX_RELATIVE_DISTANCE / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    scaled = torch.min(scaled, torch.full_like(scaled, num_buckets - 1))
    return torch.where(is_small, relative, scaled).clamp(0, num_buckets - 1)


class T5Attention(nn.Module):
    """Causal T5 attention: no score scaling, relative bias added pre-softmax."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        query, key, value = split(self.q(hidden_states)), split(self.k(hidden_states)), split(self.v(hidden_states))
        # T5 omits the 1/sqrt(head_dim) scaling; the relative bias is part of
        # the score, exactly as in google-research/byt5.
        scores = torch.matmul(query, key.transpose(3, 2)) + position_bias
        scores = scores + causal_mask.to(scores.dtype) * torch.finfo(scores.dtype).min
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        merged = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, -1)
        return self.o(merged)


class DynamicBlock(nn.Module):
    """Growable T5 decoder block over raw bytes."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = T5LayerNorm(config.d_model)
        self.attn = T5Attention(config.d_model, config.n_heads, dropout=config.dropout)
        self.relative_attention_bias = nn.Embedding(NUM_RELATIVE_BUCKETS, config.n_heads)
        self.norm2 = T5LayerNorm(config.d_model)
        self.wi_0 = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.wo = nn.Linear(config.hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("activation_ema", torch.ones(config.hidden_dim))
        self.register_buffer("activation_updates", torch.tensor(0, dtype=torch.long))

    @property
    def hidden_dim(self) -> int:
        return self.wi_0.out_features

    def compute_position_bias(self, length: int, device: torch.device) -> torch.Tensor:
        buckets = relative_position_bucket(length, device)
        bias = self.relative_attention_bias(buckets)          # (L, L, n_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)              # (1, heads, L, L)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        position_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        if position_bias is None:
            position_bias = self.compute_position_bias(x.shape[1], x.device)
        attended = self.attn(h, causal_mask, position_bias)
        x = x + self.dropout(attended)
        h = self.norm2(x)
        hidden = F.gelu(self.wi_0(h)) * self.wi_1(h)
        if self.training:
            with torch.no_grad():
                current = hidden.detach().abs().mean(dim=(0, 1)).to(self.activation_ema.device)
                rate = 0.05 if self.activation_updates.item() > 0 else 1.0
                self.activation_ema.lerp_(current, rate)
                self.activation_updates.add_(1)
        return x + self.dropout(self.wo(hidden)), position_bias

    def importance(self) -> torch.Tensor:
        outgoing = self.wo.weight.detach().abs().norm(dim=0)
        incoming = self.wi_0.weight.detach().abs().norm(dim=1) + self.wi_1.weight.detach().abs().norm(dim=1)
        activity = self.activation_ema.detach().clamp_min(1e-6)
        return outgoing * activity + 1e-4 * incoming

    @staticmethod
    def _replacement_linear(old: nn.Linear, in_features: int, out_features: int) -> nn.Linear:
        return nn.Linear(
            in_features,
            out_features,
            bias=False,
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
            # Split high-importance neurons. Halving wo's outgoing columns
            # preserves their summed contribution (apart from 1e-5 noise).
            count = new_dim - old_dim
            source = importance.topk(min(count, old_dim)).indices
            if count > old_dim:
                source = source.repeat(math.ceil(count / old_dim))[:count]
            row_map: list[list[int]] = []
            new_wi_0 = self._replacement_linear(self.wi_0, self.wi_0.in_features, new_dim)
            new_wi_1 = self._replacement_linear(self.wi_1, self.wi_1.in_features, new_dim)
            new_wo = self._replacement_linear(self.wo, new_dim, self.wo.out_features)
            with torch.no_grad():
                new_wi_0.weight[:old_dim].copy_(self.wi_0.weight)
                new_wi_1.weight[:old_dim].copy_(self.wi_1.weight)
                new_wo.weight[:, :old_dim].copy_(self.wo.weight)
                for offset, index in enumerate(source.tolist()):
                    target = old_dim + offset
                    row_map.append([target, int(index)])
                    new_wi_0.weight[target].copy_(self.wi_0.weight[index])
                    new_wi_0.weight[target].add_(torch.randn_like(new_wi_0.weight[target]) * 1e-5)
                    new_wi_1.weight[target].copy_(self.wi_1.weight[index])
                    new_wo.weight[:, target].copy_(self.wo.weight[:, index])
                    new_wo.weight[:, index].mul_(0.5)
                    new_wo.weight[:, target].mul_(0.5)
                new_activity = torch.cat((self.activation_ema, self.activation_ema[source]))
        else:
            keep = importance.topk(new_dim, sorted=False).indices.sort().values
            new_wi_0 = self._replacement_linear(self.wi_0, self.wi_0.in_features, new_dim)
            new_wi_1 = self._replacement_linear(self.wi_1, self.wi_1.in_features, new_dim)
            new_wo = self._replacement_linear(self.wo, new_dim, self.wo.out_features)
            with torch.no_grad():
                new_wi_0.weight.copy_(self.wi_0.weight[keep])
                new_wi_1.weight.copy_(self.wi_1.weight[keep])
                new_wo.weight.copy_(self.wo.weight[:, keep])
                new_activity = self.activation_ema[keep].clone()
        self.wi_0 = new_wi_0
        self.wi_1 = new_wi_1
        self.wo = new_wo
        self.register_buffer("activation_ema", new_activity.to(new_wi_0.weight.device))
        result: dict[str, Any] = {"old_dim": old_dim, "new_dim": new_dim, "mode": "grow" if growing else "prune"}
        if growing:
            # Neuron correspondence for optimizer-state carry-over (SYNC-02):
            # the new unit at row_map[i][0] is a split of the old unit at
            # row_map[i][1]; rows/columns below old_dim keep their positions.
            result["row_map"] = row_map
        else:
            # Pruning keeps an arbitrary subset: new index i corresponds to the
            # old index keep_map[i].
            result["keep_map"] = [int(index) for index in keep.tolist()]
        return result


class DynamicByteLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(DynamicBlock(config) for _ in range(config.n_layers))
        self.norm = T5LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        length = token_ids.shape[1]
        if length > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured context length")
        x = self.token_embedding(token_ids)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=token_ids.device), diagonal=1
        )
        position_bias: torch.Tensor | None = None
        for block in self.blocks:
            x, position_bias = block(x, causal_mask, position_bias)
        return self.lm_head(self.norm(x))

    def resize_hidden(self, layer: int, new_dim: int) -> dict[str, Any]:
        new_dim = max(self.config.min_hidden_dim, min(self.config.max_hidden_dim, new_dim))
        return {"layer": layer, **self.blocks[layer].resize(new_dim)}

    def hidden_dims(self) -> list[int]:
        return [block.hidden_dim for block in self.blocks]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
