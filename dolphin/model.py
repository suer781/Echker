"""字节级因果 Transformer——全自研实现。

律 L1（架构设计.md）：模型直接吃原始字节（0-255），无固定 tokenizer，
无任何现成 GPT 工程代码。唯一依赖 PyTorch 张量运算。
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    block_size: int = 256
    dropout: float = 0.0
    vocab: int = 256  # 原始字节


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.p = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = lambda t: t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.p = cfg.dropout
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + F.dropout(self.attn(self.ln1(x)), self.p, self.training)
        x = x + F.dropout(self.mlp(self.ln2(x)), self.p, self.training)
        return x


class ByteTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.head.weight = self.wte.weight  # 权重绑定
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "上下文超长"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8):
        for _ in range(max_new):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    @torch.no_grad()
    def mean_nll(self, data: bytes, device, max_chunks=8) -> float:
        """对一段原始字节流的平均负对数似然 = 系统对它的“惊讶度”。"""
        self.eval()
        b = torch.tensor(list(data), dtype=torch.long, device=device)
        if b.numel() < 2:
            return 0.0
        losses = []
        bs = self.cfg.block_size
        step = max(1, (b.numel() - 1) // max_chunks)
        for s in range(0, b.numel() - 1, step):
            e = min(s + bs, b.numel())
            if e - s < 2:
                break
            _, loss = self(b[s:e - 1].unsqueeze(0), b[s + 1:e].unsqueeze(0))
            losses.append(loss.item())
            if len(losses) >= max_chunks:
                break
        return sum(losses) / len(losses) if losses else 0.0

    def cfg_dict(self):
        return asdict(self.cfg)
