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
    def generate(self, idx, max_new, temperature=0.8, top_k=None, top_p=None):
        for _ in range(max_new):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            # top_k / top_p 过滤（默认 None=不启用，保持旧行为——直接走 softmax+
            # multinomial，不引入重新归一化的数值差异）。top_k=0 或 top_p=1.0
            # （杏仁核托管初值=不启用）同样视为未启用，原路径严格不变。
            # 两个参数都启用时按 top_k 先行过滤、top_p 再核采样，最后统一重新
            # 归一化保证概率和为 1。
            if ((top_k is not None and top_k > 0)
                    or (top_p is not None and 0 < top_p < 1)):
                # top_k：只保留概率最高的 k 个词，其余置 0
                if top_k is not None and top_k > 0:
                    k = min(int(top_k), probs.shape[-1])
                    kth = torch.topk(probs, k, dim=-1).values[..., -1:]
                    probs = torch.where(probs < kth, torch.zeros_like(probs), probs)
                # top_p：核采样——累积概率达到 p 的最小集合（至少保留一个词）
                if top_p is not None and 0 < top_p < 1:
                    sorted_probs, sort_idx = torch.sort(probs, dim=-1, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum - sorted_probs > top_p  # 超过阈值的尾部置 0
                    sorted_probs = torch.where(
                        mask, torch.zeros_like(sorted_probs), sorted_probs)
                    # 核集合内重新归一化
                    sorted_probs = sorted_probs / (
                        sorted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                    # 按原索引放回
                    probs = torch.zeros_like(probs).scatter_(-1, sort_idx, sorted_probs)
                # 重新归一化（top_k 置 0 后必须归一化，保证概率和为 1）
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    @torch.no_grad()
    def mean_nll(self, data: bytes, device, max_chunks=8) -> float:
        """对一段原始字节流的平均负对数似然 = 系统对它的“惊讶度”。"""
        return self.mean_nll_batch([data], device, max_chunks=max_chunks)[0]

    @torch.no_grad()
    def mean_nll_batch(self, datas: list, device, max_chunks=8, max_batch=64) -> list:
        """批量计算多条字节流的平均负对数似然（惊讶度）。

        与原 mean_nll 逐窗口串行在数学上等价：每条数据按相同 step 切最多
        max_chunks 个窗口，完整长度窗口堆叠成 batch 一次 forward（受 max_batch
        限制分批），末尾不足 block_size 的短窗口保持单条 forward（保证与原实现
        逐位一致）。返回与 datas 等长的 float 列表；长度 <2 的数据返回 0.0。
        """
        self.eval()
        bs = self.cfg.block_size
        full_x, full_y, full_owner = [], [], []  # 完整窗口：长度 == bs，可堆叠
        short_jobs = []                          # 短窗口：长度 < bs，保持单条 forward
        for data_idx, data in enumerate(datas):
            b = torch.tensor(list(data), dtype=torch.long, device=device)
            if b.numel() < 2:
                continue
            step = max(1, (b.numel() - 1) // max_chunks)
            n_windows = 0
            for s in range(0, b.numel() - 1, step):
                e = min(s + bs, b.numel())
                if e - s < 2:
                    break
                x = b[s:e - 1]
                y = b[s + 1:e]
                # 常规窗口跨满 block_size 步长（e-s == bs，x 长度 == bs-1），
                # 长度一致可堆叠成 batch；末尾不足 block_size 的短窗口保持单条。
                if e - s == bs:
                    full_x.append(x)
                    full_y.append(y)
                    full_owner.append(data_idx)
                else:
                    short_jobs.append((x, y, data_idx))
                n_windows += 1
                if n_windows >= max_chunks:
                    break
        # 完整窗口分批 stack 后一次 forward（受 max_batch 限制）。
        # 注意：self() 返回的 loss 是 F.cross_entropy 默认 reduction='mean' 的
        # 全体标量（不可拆回 per-sample），故这里只取 logits，再用 reduction='none'
        # 手动按样本求平均 CE——与原实现逐窗口 forward 的标量 loss 数学上逐位等价。
        losses = [[] for _ in datas]
        if full_x:
            for i in range(0, len(full_x), max_batch):
                xx = torch.stack(full_x[i:i + max_batch])
                yy = torch.stack(full_y[i:i + max_batch])
                logits, _ = self(xx)
                per_token = F.cross_entropy(
                    logits.reshape(-1, self.cfg.vocab), yy.reshape(-1), reduction="none")
                per_sample = per_token.reshape(xx.shape[0], -1).mean(dim=1)
                vals = per_sample.tolist()
                for k, v in enumerate(vals):
                    losses[full_owner[i + k]].append(v)
        # 短窗口保持单条 forward（与原实现逐位一致）
        for x, y, owner in short_jobs:
            _, loss = self(x.unsqueeze(0), y.unsqueeze(0))
            losses[owner].append(loss.item())
        return [sum(l) / len(l) if l else 0.0 for l in losses]

    def cfg_dict(self):
        return asdict(self.cfg)
