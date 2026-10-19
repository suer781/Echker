# -*- coding: utf-8 -*-
"""在线知识蒸馏（2026-10-17 用户定案：边学数据边看 Qwen 指导）。

核心：Byte-Prefix Marginalization（BPM）——把 Qwen（词表级）的 next-token
概率分布，按每个 token 的 UTF-8 首字节汇总成 Dolphin（字节级）能学的
next-byte 软分布（256 维）。

角色：
- Qwen2.5-0.5B 只做"教师"：对训练文本前向一次，输出字节级软标签；
- Dolphin 训练时，损失 = 真实字节 CE + 半球 KD + Qwen 软标签 KL；
- Qwen 随时可卸（qwen_kd_enabled=False），不影响 Dolphin 自学习/上网。

实现要点：
- 预计算 first_byte 映射表：Qwen 词表中每个 token id → 其 UTF-8 编码首字节。
- 对一段文本：Qwen tokenize → 前向得到 [T, vocab] logits；
  用 top-k（默认 2000）近似，按 first_byte scatter_add 汇总成 [T, 256]；
  再把 token 边界映射到字节索引，逐字节组装 soft_targets [L, 256]：
    * token 首字节位置：BPM 后的 next-byte 软分布；
    * token 内部字节位置：one-hot（下一个字节是该 token 的已知后续字节）。
"""
import os

import torch
import torch.nn.functional as F

# Qwen 词表 15 万 token 全部参与 BPM 太慢；top-k 已覆盖 >99.9% 概率质量。
_BPM_TOP_K = 2000
# KD 温度（软标签平滑度；越高分布越软）
_KD_T = 2.0


def _byte_offsets(text: str):
    """把字符序列的每个字符边界换算成 UTF-8 字节偏移。

    返回 (byte_len_prefix, utf8_bytes)：
    - byte_len_prefix[i] = 前 i 个字符的字节总数；
    - utf8_bytes = text.encode('utf-8')。
    """
    data = text.encode("utf-8")
    prefix = [0]
    total = 0
    for ch in text:
        total += len(ch.encode("utf-8"))
        prefix.append(total)
    return prefix, data


class QwenTeacher:
    """Qwen 教师：加载一次，对任意文本输出字节级软标签。"""

    def __init__(self, model_dir, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.first_byte = self._build_first_byte_map()

    def _build_first_byte_map(self):
        """词表中每个 token id → 其 UTF-8 编码的首字节（0-255）。

        长度用模型配置的 vocab_size（Qwen2.5 为 151936），与 logits 维度一致；
        不能用 tokenizer.vocab_size（151643，不含部分特殊/added token）。
        """
        vocab = self.model.config.vocab_size
        fb = torch.zeros(vocab, dtype=torch.long)
        for i in range(vocab):
            try:
                t = self.tok.decode([i])
                b = t.encode("utf-8")
                fb[i] = b[0] if b else 0
            except Exception:
                fb[i] = 0
        return fb

    @torch.no_grad()
    def soft_targets_for(self, text, temperature=_KD_T):
        """把 Qwen 对 text 的 next-token 分布边际化为 next-byte 软标签。

        输入：text（str 或 bytes，UTF-8）。
        返回：torch.Tensor [L, 256]，soft_targets[k] = 在字节位置 k 预测
        bytes[k] 的软分布（Qwen 指导）。L = text 的 UTF-8 字节数。
        """
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if not text:
            return None
        try:
            enc = self.tok(text, return_offsets_mapping=True,
                           return_tensors="pt")
        except Exception:
            return None
        input_ids = enc["input_ids"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()  # 字符偏移（含特殊 token）

        # Qwen 前向分块（2026-10-19 修复）：原实现一次性产生 [T, vocab] logits，
        # 长文本（缓冲 >64KB 时 T 达数万）会分配数十 GB 张量，导致内存爆炸、
        # 训练线程卡死、系统换页。改为按 _FWD_CHUNK 个 token 分块前向，每块
        # 立即 topk+scatter 后释放；_FWD_OVERLAP 前缀保留因果上下文，语义基本不变。
        _FWD_CHUNK = 512
        _FWD_OVERLAP = 64
        _n_tok = input_ids.size(1)
        _bpm_parts = []
        for _s in range(0, _n_tok, _FWD_CHUNK):
            _e = min(_s + _FWD_CHUNK, _n_tok)
            _seg_s = max(0, _s - _FWD_OVERLAP)
            # 分块前向：logits 只占 [chunk+overlap, vocab]，峰值内存可控
            _logits = self.model(input_ids[:, _seg_s:_e]).logits[0]
            _probs = F.softmax(_logits / temperature, dim=-1)
            _top_probs, _top_idx = torch.topk(
                _probs, k=min(_BPM_TOP_K, _probs.size(-1)), dim=-1)
            _top_fb = self.first_byte.to(self.device)[_top_idx]
            _bpm_seg = torch.zeros(_probs.size(0), 256, device=self.device,
                                    dtype=torch.float32)
            _bpm_seg.scatter_add_(1, _top_fb, _top_probs)
            _row_sum = _bpm_seg.sum(dim=1, keepdim=True)
            _bpm_seg = _bpm_seg / _row_sum.clamp_min(1e-8)
            _keep = _s - _seg_s  # 丢弃 overlap 前缀的预测
            _bpm_parts.append(_bpm_seg[_keep:])
        bpm = torch.cat(_bpm_parts, dim=0)  # [T, 256]

        # —— 字符/字节边界映射 ——
        prefix, utf8_bytes = _byte_offsets(text)
        L = len(utf8_bytes)
        soft = torch.zeros(L, 256, device=self.device, dtype=torch.float32)
        # 记录每个字节位置所属 token 的起始字节和 token 字节内容
        # token 字节表示：用 tokenizer 对每个 token id 解码再编码可能不稳，
        # 直接用 offsets 对应文本片段的字节，保证与输入完全一致。
        token_bytes = []   # 每个非空 token 的字节内容
        token_bstart = []  # 每个非空 token 的起始字节索引
        token_bend = []    # 每个非空 token 的结束字节索引
        for t, (cs, ce) in enumerate(offsets):
            if cs == ce:  # 特殊 token（BOS 等）无文本
                continue
            chunk = text[cs:ce]
            cb = chunk.encode("utf-8")
            bs = prefix[cs]
            be = prefix[ce]
            token_bytes.append(cb)
            token_bstart.append(bs)
            token_bend.append(be)
        # 把每个 token 的 next-byte 分布填到其首字节位置：
        # 第 i 个非空 token 的首字节位置 bs_i，其"下一个字节"由 Qwen 在
        # 位置 i-1 的 logits 预测（即 bpm[i-1]）。i=0（序列开头）没有前驱，
        # 用 bpm[0] 近似（影响极小）。
        n_tok = len(token_bytes)
        for i in range(n_tok):
            bs = token_bstart[i]
            be = token_bend[i]
            if be <= bs:
                continue
            src = bpm[i - 1] if i >= 1 else bpm[i]
            soft[bs] = src
            # token 内部字节：one-hot 到 token 的已知后续字节
            for k in range(bs + 1, be):
                idx_in_tok = k - bs
                nxt = token_bytes[i][idx_in_tok] if idx_in_tok < len(token_bytes[i]) else 0
                one = torch.zeros(256, device=self.device)
                one[int(nxt)] = 1.0
                soft[k] = one
        return soft.cpu()

    def kd_loss(self, s_logits, soft_targets):
        """Dolphin 在 [B, T, 256] 的 logits 与软标签的 KL 蒸馏损失。"""
        if soft_targets is None:
            return torch.tensor(0.0, device=s_logits.device)
        st = soft_targets.to(s_logits.device)
        # soft_targets: [B*T, 256]（由调用方切好）或 [B, T, 256]
        if st.dim() == 2:
            st = st.unsqueeze(0)
        if st.dim() == 3 and st.size(0) == 1 and s_logits.size(0) > 1:
            st = st.expand(s_logits.size(0), -1, -1)
        log_p = F.log_softmax(s_logits / _KD_T, dim=-1)
        kl = F.kl_div(log_p, st, reduction="batchmean") * (_KD_T ** 2)
        return kl


# ---------- 便捷工厂（供 feed.py 使用） ----------

_teacher_cache = {}


def get_teacher(model_dir, device=None):
    """全局单例：Qwen 教师只加载一次，多个睡眠周期复用。"""
    key = os.path.abspath(model_dir)
    if key not in _teacher_cache:
        _teacher_cache[key] = QwenTeacher(model_dir, device=device)
    return _teacher_cache[key]