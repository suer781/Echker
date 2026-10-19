# -*- coding: utf-8 -*-
"""片段级金发姑娘门控测试（2026-10-17 用户定案：屏蔽噪声片段而非整条略过）。

对应实现：dolphin/dolphin.py 的 learn() 片段级门控 + _split_windows。

设计语义：
- 输入按 block_size 无重叠切窗，逐窗口金发姑娘三态判决；
- 太熟窗口（skip）→ 丢弃；太怪窗口（suspend）→ 进冷层待后续；
- 中间带窗口（pass）→ 拼接后进入经验缓冲（有效内容正常学习）；
- 计数按**窗口**累加（skipped_too_easy / suspended_too_hard），
  被屏蔽字节数计入 noise_masked_bytes（观测）。

红线自守：所有 Dolphin 用沙箱冷层（MemoryStore cold_path 指向临时目录），
绝不触碰生产 dolphin/memory_cold.jsonl 与 probes/probe.txt。
"""
import os
import shutil
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.dolphin import Dolphin
from dolphin.goldilocks import PRIOR_ABILITY
from dolphin.memory import MemoryStore
from dolphin.model import Config

PASS = []


def check(name, cond, detail=""):
    PASS.append(cond)
    print(f"  {'✓' if cond else '✗'} {name}  {detail}")


def _tmpdir(prefix="dolphin_frag_"):
    return tempfile.mkdtemp(prefix=prefix)


def make_dolphin(cfg=None, probe_path=None, cold_path=None):
    cfg = cfg or Config(d_model=32, n_layers=2, n_heads=2, block_size=64)
    d = Dolphin(cfg=cfg, device="cpu", probe_path=probe_path)
    d.param_adaptive_enabled = False
    # 沙箱冷层：测试绝不触碰生产记忆冷层
    d.memory = MemoryStore(cold_path=cold_path)
    return d


def _fresh(tmp):
    probe = os.path.join(tmp, "probe.txt")
    with open(probe, "wb") as f:
        f.write(b"0123456789abcdef" * 8)
    cold = os.path.join(tmp, "cold.jsonl")
    d = make_dolphin(probe_path=probe, cold_path=cold)
    return d


# ==================== _split_windows ====================

def t_split_windows():
    d = _fresh(_tmpdir())
    check("空输入返回 []", d._split_windows(b"", 64) == [])
    ws = d._split_windows(b"abc", 2)
    check("尾块不足独立保留不丢弃", ws == [b"ab", b"c"])
    data = bytes(range(256)) * 2  # 512 字节
    ws = d._split_windows(data, 64)
    check("512 字节按 64 切分为 8 窗",
          len(ws) == 8 and all(len(w) == 64 for w in ws))
    check("拼接还原原始数据", b"".join(ws) == data)
    # 非整数倍：尾部独立成块，任何数据都不被切分逻辑丢弃
    data2 = bytes(range(100))  # 100 = 64 + 36
    ws2 = d._split_windows(data2, 64)
    check("非整数倍尾部独立成块且拼接还原",
          len(ws2) == 2 and b"".join(ws2) == data2,
          f"块={[len(w) for w in ws2]}")
    # 整条不足一块也返回（哪怕 1 字节）
    ws3 = d._split_windows(b"x", 64)
    check("整条不足一块也返回不丢弃", ws3 == [b"x"])


# ==================== learn 片段级门控 ====================

def t_empty():
    d = _fresh(_tmpdir())
    t0 = d.learn_total
    eid = d.learn("", source="test")
    check("空数据不进 buffer 且 learn_total+1",
          eid is None and d.learn_total == t0 + 1 and len(d.buffer.items) == 0)


def t_single_window_skip():
    d = _fresh(_tmpdir())
    d.ability = 10.0  # 抬高基线 → 新鲜模型 surprise≈5.55 ≪ 0.95×10 → 太熟
    eid = d.learn("金发姑娘太熟跳过记录，内容足够长。", source="test")
    check("单窗口太熟：不进 buffer、skip 计数 +1",
          eid is None and len(d.buffer.items) == 0
          and d.skipped_too_easy == 1 and d.suspended_too_hard == 0)


def t_single_window_suspend():
    d = _fresh(_tmpdir())
    d.ability = 3.0  # 压低基线 → surprise≈5.55 ≫ 1.20×3 → 太怪
    eid = d.learn("金发姑娘太怪挂起记录，内容足够长。", source="test")
    cold = [e for e in d.memory.cold_entries if e.kind == "suspended"]
    check("单窗口太怪：不进 buffer、suspend 计数 +1、进冷层",
          eid is None and len(d.buffer.items) == 0
          and d.suspended_too_hard == 1 and len(cold) == 1)


def t_single_window_pass():
    d = _fresh(_tmpdir())
    d.ability = PRIOR_ABILITY  # 先验 → delta≈0 → 中间带放行
    eid = d.learn("金发姑娘中间带放行记录，内容足够长。", source="test")
    check("单窗口中间带：进 buffer",
          eid is not None and len(d.buffer.items) == 1
          and d.skipped_too_easy == 0 and d.suspended_too_hard == 0)


def t_multi_window_all_skip():
    d = _fresh(_tmpdir())
    d.ability = 10.0
    long_text = "这是一段足够长的中文内容用来验证多窗口行为。" * 30
    data = long_text.encode("utf-8")
    n_windows = len(d._split_windows(data, d.cfg.block_size))
    before = d.skipped_too_easy
    eid = d.learn(long_text, source="test")
    check("多窗口全太熟：skip 按窗口数累加、不进 buffer",
          eid is None and d.skipped_too_easy - before == n_windows
          and len(d.buffer.items) == 0,
          f"窗口={n_windows} skip增量={d.skipped_too_easy - before}")


def t_multi_window_all_suspend():
    d = _fresh(_tmpdir())
    d.ability = 3.0
    long_text = "这是一段足够长的中文内容用来验证多窗口行为。" * 30
    data = long_text.encode("utf-8")
    n_windows = len(d._split_windows(data, d.cfg.block_size))
    before_susp = d.suspended_too_hard
    before_cold = len([e for e in d.memory.cold_entries if e.kind == "suspended"])
    eid = d.learn(long_text, source="test")
    after_cold = len([e for e in d.memory.cold_entries if e.kind == "suspended"])
    check("多窗口全太怪：suspend 按窗口累加、冷层按窗口增加、不进 buffer",
          eid is None and d.suspended_too_hard - before_susp == n_windows
          and after_cold - before_cold == n_windows
          and len(d.buffer.items) == 0,
          f"窗口={n_windows} suspend增量={d.suspended_too_hard - before_susp}")


def t_multi_window_all_pass_concatenate():
    d = _fresh(_tmpdir())
    d.ability = PRIOR_ABILITY
    # 长度 = block_size × 3（整数倍，拼接应还原原始数据）
    seg = "片段级门控拼接验证，这一段是有效正文内容。" * 4
    long_text = seg * 3
    data = long_text.encode("utf-8")
    n_windows = len(d._split_windows(data, d.cfg.block_size))
    eid = d.learn(long_text, source="test")
    check("多窗口全中间带：进 buffer 且内容=原始数据（整数倍无碎片）",
          eid is not None and len(d.buffer.items) == 1
          and d.buffer.items[0].data == data,
          f"窗口={n_windows} buffer字节={len(d.buffer.items[0].data)}")


def t_noise_masked_bytes():
    d = _fresh(_tmpdir())
    d.ability = 3.0  # 所有窗口 suspend → 全部屏蔽
    long_text = "这是一段足够长的中文内容用来验证多窗口行为。" * 30
    data = long_text.encode("utf-8")
    before = d.noise_masked_bytes
    d.learn(long_text, source="test")
    check("noise_masked_bytes 累加被屏蔽字节数",
          d.noise_masked_bytes - before == len(data),
          f"屏蔽 {d.noise_masked_bytes - before} 字节（总 {len(data)}）")


def t_save_load_noise_masked():
    tmp = _tmpdir()
    d = _fresh(tmp)
    d.noise_masked_bytes = 12345
    ckpt = os.path.join(tmp, "fed_state.pt")
    d.save(ckpt)
    d2 = _fresh(tmp)
    d2.load(ckpt)
    check("noise_masked_bytes 随档往返", d2.noise_masked_bytes == 12345)
    shutil.rmtree(tmp, ignore_errors=True)


# ==================== _dynamic_band 动态偏差 ====================

def t_dynamic_band():
    d = _fresh(_tmpdir())
    # 不足 8 个历史：直接返回基准
    d._surprise_hist = [5.5, 5.6]
    lo, hi = d._dynamic_band(0.05, 0.20)
    check("历史不足 8 返回基准", lo == 0.05 and hi == 0.20)
    # 分布集中（CV≈0）：带收窄（lo 更高、hi 更低）
    d._surprise_hist = [5.5] * 16
    lo, hi = d._dynamic_band(0.05, 0.20)
    check("分布集中带收窄", lo >= 0.05 and hi <= 0.20,
          f"lo={lo:.4f} hi={hi:.4f}")
    # 分布分散（CV 大）：带放宽（lo 更低、hi 更高）
    d._surprise_hist = [1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 16.0, 20.0,
                        1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    lo, hi = d._dynamic_band(0.05, 0.20)
    check("分布分散带放宽", lo <= 0.05 and hi >= 0.20,
          f"lo={lo:.4f} hi={hi:.4f}")
    # 钳位域：lo 不超 [0.01,0.15]，hi 不超 [0.05,0.60]
    check("lo 在域内", 0.01 <= lo <= 0.15)
    check("hi 在域内", 0.05 <= hi <= 0.60)


# ==================== 双尺度语义饱和 ====================

def t_satiation_dual_scale():
    d = _fresh(_tmpdir())
    data = "重复内容用于语义饱和测试。".encode("utf-8")
    # 短期窗口内连续大量重复 → 强抑制（factor 显著低于 1.0），
    # 但长期下限 0.60 把因子拉回（取较大者，避免被完全排斥）。
    for _ in range(10):
        d._apply_semantic_saturation(data, 5.5)
    f_short_low = d._apply_semantic_saturation(data, 5.5)
    check("短期大量重复被长期下限拉回", abs(f_short_low - 0.60) < 0.01,
          f"factor={f_short_low:.4f}（期望 0.60）")
    # 内容被挤出短期窗口后不再抑制（短期尺度主导，历史重复不排斥当前学习）
    d2 = _fresh(_tmpdir())
    for _ in range(3):
        d2._apply_semantic_saturation(data, 5.5)
    other = "其他不同内容用于填充长期窗口。".encode("utf-8")
    for i in range(60):  # 填充短期窗口，把 data 挤出短期计数
        d2._apply_semantic_saturation(other + bytes([i % 256]), 5.5)
    f_long = d2._apply_semantic_saturation(data, 5.5)
    check("短期挤出后不再抑制", abs(f_long - 1.0) < 0.01,
          f"factor={f_long:.4f}（期望 1.0）")
    # 双尺度取较大者：同一内容短期计数 3 次 → factor_short=0.10（强抑制），
    # 长期计数 3 次 → factor_long=0.85（弱抑制）；取较大者=0.85，不被短期拉低。
    d3 = _fresh(_tmpdir())
    for _ in range(3):
        d3._apply_semantic_saturation(data, 5.5)
    f_combined = d3._apply_semantic_saturation(data, 5.5)
    check("双尺度取较大者（长期弱抑制拉回）",
          abs(f_combined - 0.85) < 0.01,
          f"factor={f_combined:.4f}（期望 0.85）")


# ==================== forget_cold limit 模式（睡眠杂交小任务用） ====================

def t_forget_cold_limit():
    d = _fresh(_tmpdir())
    m = d.memory
    m._ensure_cold()
    now = time.time()
    # 5 条过期零命中 + 1 条被想起过的记忆（hits>0，永不遗忘）
    for i in range(5):
        m.add_cold(f"过期条目{i}".encode("utf-8"), 5.5, 0, kind="residue")
    m.add_cold("被想起过的记忆".encode("utf-8"), 5.5, 0, kind="note")
    for e in m.cold_entries:
        e.created = now - 10 * 86400  # 10 天前（超过 residue TTL）
    m.cold_entries[-1].hits = 1  # 被想起过
    stat = m.forget_cold(limit=2)
    check("limit 模式只淘汰最旧 2 条", stat.get("cold_forget_expired") == 2)
    remaining = [e.text for e in m.cold_entries]
    check("被想起过的记忆保留", "被想起过的记忆" in remaining)
    check("其余过期条目保留（limit 截断）", len(remaining) == 4)
    # 无过期条目时返回空 dict
    for e in m.cold_entries:
        e.created = now
    stat2 = m.forget_cold(limit=8)
    check("无过期返回空 dict", stat2 == {})


# ==================== 冷层自适应上限 + 梦境热提升 ====================

def t_adaptive_cold_cap():
    d = _fresh(_tmpdir())
    cap = d.adaptive_cold_cap()
    check("冷层自适应上限在域内", 20000 <= cap <= 1_000_000,
          f"cap={cap}")
    # 模型参数越多 → 上限越高（构造一个更大模型比较）
    big = make_dolphin(cfg=Config(d_model=64, n_layers=4, n_heads=4, block_size=64))
    cap_big = big.adaptive_cold_cap()
    check("更大模型上限更高", cap_big >= cap,
          f"cap_big={cap_big} vs cap={cap}")


def t_promote_cold_related():
    d = _fresh(_tmpdir())
    m = d.memory
    m._ensure_cold()
    # 冷层添加一条"梦境记忆"
    dream_text = "梦境记忆：用户昨天聊过蓝色鲸鱼与深海探测。"
    m.add_cold(dream_text.encode("utf-8"), 5.5, 0, kind="note")
    # 不相关的 query 不应提升
    n0 = m.promote_cold_related("完全无关的话题内容".encode("utf-8"), k=2)
    hot_before = [e.text for e in m.entries]
    check("不相关查询不提升", n0 == 0 and dream_text not in hot_before)
    # 相关 query 提升到热层
    n1 = m.promote_cold_related(
        "用户提到了蓝色鲸鱼和深海探测的事情".encode("utf-8"), k=2)
    hot_after = [e.text for e in m.entries]
    check("相关查询提升到热层", n1 >= 1 and dream_text in hot_after,
          f"n={n1} 热层={hot_after}")
    # 空冷层不抛错
    m2 = MemoryStore(cold_path=os.path.join(_tmpdir(), "cold2.jsonl"))
    check("空冷层不抛错", m2.promote_cold_related(b"abc") == 0)


# ==================== 杏仁核三层调节 ====================

def t_amygdala_three_layer():
    from dolphin.amygdala import Amygdala
    am = Amygdala()
    # L3 情绪检测（兜底统计路径）：生气负分、开心正分、中性近 0
    s_angry = am.sentiment("你真是个混蛋，太让人生气了！")
    s_happy = am.sentiment("谢谢！你太棒了哈哈")
    s_neutral = am.sentiment("今天天气怎么样")
    check("生气文本负分", s_angry < 0, f"s={s_angry:.3f}")
    check("开心文本正分", s_happy > 0, f"s={s_happy:.3f}")
    check("中性文本近 0", abs(s_neutral) < 0.3, f"s={s_neutral:.3f}")
    # L3 温度：负面降温、正面升温
    t_low = am.temperature_for_sentiment(-0.8, base=0.8)
    t_high = am.temperature_for_sentiment(0.8, base=0.8)
    check("负面情绪降温", t_low < 0.8, f"t={t_low:.3f}")
    check("正面情绪升温", t_high > 0.8, f"t={t_high:.3f}")
    # L1+L2+L3 综合：生气输入 → 最终温度低于中性输入
    t_angry = am.temperature_for_user("你回答错了！真让人生气！", 5.5, base=0.8)
    t_after = am.temperature_for_user("谢谢你的帮助，太棒了！", 5.5, base=0.8)
    check("综合：生气降温后开心升温", t_after > t_angry,
          f"angry={t_angry:.3f} happy={t_after:.3f}")
    # 上下文统计窗口记录
    check("上下文统计窗口已记录", len(am.recent_surprises) == 2
          and len(am.recent_sentiments) == 2)
    # 连续负面交互 → L2 上下文降温
    am2 = Amygdala()
    for _ in range(10):
        am2.temperature_for_user("你真是个笨蛋，太令人失望了！", 6.0, base=0.8)
    t_ctx = am2.temperature_for_context(base=0.8)
    check("上下文负面统计降温", t_ctx < 0.8, f"t_ctx={t_ctx:.3f}")
    # —— 语义理解路径（2026-10-17 用户定案：不靠关键词）——
    d = _fresh(_tmpdir())
    am3 = Amygdala()
    # 用模型语义表示编码原型并判断情绪：不抛错、值有界、原型缓存命中
    s_sem = am3.sentiment("你太让我生气了，这个回答完全错误！",
                          model=d.awake().model, device=d.device)
    check("语义情绪值有界", -1.0 <= s_sem <= 1.0, f"s={s_sem:.3f}")
    vecs = am3._get_prototypes(d.awake().model, d.device)
    check("语义原型三类齐备", set(vecs.keys()) == {"angry", "happy", "neutral"})
    check("语义原型缓存命中", am3._proto_cache is not None
          and am3._proto_cache.get("model_id") == id(d.awake().model))
    t_sem = am3.temperature_for_user(
        "谢谢你的帮助，我太开心了！", 5.5, base=0.8,
        model=d.awake().model, device=d.device)
    check("语义三层综合不抛错且有界", 0.4 <= t_sem <= 1.4, f"t={t_sem:.3f}")


def main():
    print("== _split_windows ==")
    t_split_windows()
    print("== learn 片段级门控 ==")
    t_empty()
    t_single_window_skip()
    t_single_window_suspend()
    t_single_window_pass()
    t_multi_window_all_skip()
    t_multi_window_all_suspend()
    t_multi_window_all_pass_concatenate()
    t_noise_masked_bytes()
    t_save_load_noise_masked()
    print("== _dynamic_band 动态偏差 ==")
    t_dynamic_band()
    print("== 双尺度语义饱和 ==")
    t_satiation_dual_scale()
    print("== forget_cold limit 模式 ==")
    t_forget_cold_limit()
    print("== 冷层自适应上限 + 梦境热提升 ==")
    t_adaptive_cold_cap()
    t_promote_cold_related()
    print("== 杏仁核三层调节 ==")
    t_amygdala_three_layer()
    n_fail = sum(1 for c in PASS if not c)
    print(f"\n通过 {len(PASS) - n_fail}/{len(PASS)}，失败 {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())