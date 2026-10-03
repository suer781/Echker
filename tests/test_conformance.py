"""律符合度常备测试：每次改动后必须全绿。

来源：2026-10-03 实现符合度审计的 8 组实验固化。审计员发现的问题
（L9 蒸发、负奖励钳位、L6 算子缺失）由此类实验抓获——本文件是
"声明 vs 代码"的常设对账机制，不依赖任何人自觉。

用法：python tests/test_conformance.py   （纯 CPU，约 1 分钟）
"""
import contextlib
import os
import secrets
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.dolphin import Dolphin
from dolphin.experience import ExperienceBuffer
from dolphin.model import ByteTransformer, Config
from dolphin.sleep import varied_replay

PASS = []
SKIP = []


def check(name, cond, detail="", skip=False):
    """skip=True：本环境不适用（如生产探测集不在场），显式记为跳过而非判红。

    跳过项仍进 PASS 以保持断言总数跨环境稳定（副本实验里生产卷不在场，
    断言数不缩水），但单独记账并在结尾汇总，绝不与真绿混淆。
    """
    if skip:
        SKIP.append(name)
        PASS.append(True)
        print(f"  ⏭ {name}  {detail}（本环境不适用，跳过）")
        return
    PASS.append(cond)
    print(f"  {'✓' if cond else '✗'} {name}  {detail}")


# ============ 共用探测集夹具（律 L8 脱钩，2026-10-04） ============
# 为什么要它：此前 22 处 Dolphin(cfg=..., device="cpu") 里绝大多数不传 probe_path，
# 于是吃生产默认probes/probe.txt —— 让"78/78 全绿"取决于一个与被测性质无关的
# 外部文件是否躺在原处。实测把它挪走：第 12 项 t_l5_l8_training_paths 抛
# GateDisabled 直接中断，78 项只跑出 11 项，后面 60+ 项压根跑不到。
#
# 本夹具只换测试夹具，生产律一个字节都不动：
#   - 律 L8（任何 hemisphere 永不在探测集上训练）是正确的生产律，不许碰；
#   - G8 fail-fast（ensure_gate_ready 缺失即抛 GateDisabled）是明令禁止削弱的守卫，
#     原样保留——t_g8_gate_failfast 故意用不存在的 probe 路径验它，不套用本夹具；
#   - 合成卷用自然语言逻辑推理片段而非随机字节：门控在随机数据上没有意义
#     （逐块 NLL 无结构，配对差的 SE 退化），语义片段才让体检有可判的卷面。
# 语料风格取自 probes/build_logic_probe.py 的逻辑推理域（若→则传递、金属导电、
# 逆否、充分必要条件），与 probes/probe.txt 同域但独立撰写，不复制生产卷内容。

_PROBE_SEG = (
    "逻辑推理探测片段：若甲高于乙，乙高于丙，则甲高于丙；"
    "所有金属都导电，铁是金属，故铁导电；下雨地必湿，此地不湿，"
    "故此地未必下雨；鸟会飞，企鹅是鸟，然企鹅不会飞，故前提有误；"
    "此段仅供体检评分，永不进训练粮。"
).encode()


def _write_synthetic_probe(path, n_chunks, block_size=256):
    """写一份合成探测卷：n_chunks 个整块（尾块由整除保证不缺字节）。

    补齐到 block_size 整数倍 → 恰好 n_chunks 块、每块 >= 2 字节（probe.evaluate
    的可评分下限），n_chunks >= 2 时 gate_decision 的 SE 才有 n-1>0 的方差可估。
    """
    body = (_PROBE_SEG * (n_chunks * block_size // len(_PROBE_SEG) + 1))
    body = body[: n_chunks * block_size]
    with open(path, "wb") as f:
        f.write(body)
    return path


@contextlib.contextmanager
def _probe_fixture(prefix, n_chunks=12):
    """共用探测集夹具：临时目录 → 写合成 probe → yield 路径 → rmtree 收尾。

    形态沿用本文件既有的 t_g8_gate_failfast / t_g3_guard_ruminate 写法
    （mkdtemp + 写临时 probe + finally rmtree），只是提为共用 helper。
    n_chunks 默认 12（十几块足矣）：这些测试都是小 Config（d_model=64/2 层/
    block_size=256），12 块足够门控判决，又不白白拖慢 78 项测试——不做成
    生产卷那样的 242 块。
    """
    tmp = tempfile.mkdtemp(prefix=prefix)
    try:
        yield _write_synthetic_probe(os.path.join(tmp, "probe.txt"), n_chunks)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_l1_bytes():
    """L1：任意二进制直接进前向，无 tokenizer。"""
    cfg = Config(d_model=64, n_layers=1, n_heads=2, block_size=64)
    m = ByteTransformer(cfg)
    x = torch.randint(0, 256, (1, 64))  # 伪装成 PNG 头的随机字节
    logits, loss = m(x[:, :-1], x[:, 1:])
    check("L1 字节流前向有限", torch.isfinite(loss).item(), f"loss={loss.item():.3f}")


def t_l4_band():
    """L4：校准带下太熟/太怪必须被压到近零（≥64 条时纯经验带）。"""
    buf = ExperienceBuffer()
    for i in range(64):
        buf.add(f"校准带记录{i}".encode(), surprise=3.0 + (i % 10) * 0.01)
    buf.add("极端太熟的低惊讶度记录".encode(), surprise=0.1)
    buf.add("极端太怪的高惊讶度记录".encode(), surprise=10.0)
    sel, _ = buf.select(1.0)
    rank = {e.surprise: r for r, (s, e) in enumerate(sel)}
    check("L4 带通压制两尾", rank[0.1] > rank[3.0] and rank[10.0] > rank[3.0],
          f"太熟排名{rank[0.1]} 太怪排名{rank[10.0]}（正常带应为前两名）")


def t_l6_variation():
    """L6：重放=改写不复读。统计性断言（40 次中 ≥25 次非逐字）——拒绝掷硬币门禁。"""
    rng = secrets.SystemRandom()
    a = "海豚是海洋哺乳动物。它们用肺呼吸。每次浮出水面都会换气。".encode()
    donor = "鲸鱼用喷气孔换气。企鹅用翅膀游泳。".encode()
    non_verbatim = sum(varied_replay(a, rng, donor) != a for _ in range(40))
    distinct = len({varied_replay(a, rng, donor) for _ in range(40)})
    check("L6 非逐字率 ≥25/40", non_verbatim >= 25, f"{non_verbatim}/40")
    check("L6 输出多样性 ≥3 种", distinct >= 3, f"{distinct} 种")


def t_l7_decay():
    """L7：重复惩罚严格 1/k 谐波衰减（缓冲≥64 条进入纯经验带，排除先验因子）。"""
    buf = ExperienceBuffer(satiation_window=50)
    for i in range(59):
        buf.add(f"填充记录{i}号，使缓冲达到纯经验带样本量。".encode(), 3.0)
    dup = "同一条记录原样重复五次以验证惩罚衰减。".encode()
    for _ in range(5):
        buf.add(dup, 3.0)
    sel, _ = buf.select(1.0)
    scores = [round(s, 4) for s, e in sel if e.data == dup]
    check("L7 谐波衰减 1/0.5/0.333/0.25/0.2",
          scores == [1.0, 0.5, 0.3333, 0.25, 0.2], f"{scores}")


def t_l9_no_evaporation():
    """L9：选拔域=全量缓冲，窗外经验绝不蒸发。"""
    buf = ExperienceBuffer(satiation_window=10)
    for i in range(25):
        buf.add(f"审计场景记录{i}号，内容长度足够避免碎片过滤。".encode(), 3.0)
    sel, rest = buf.select(0.45)
    check("L9 无蒸发（窗10/25条）", len(sel) + len(rest) == 25,
          f"{len(sel)}+{len(rest)}={len(sel)+len(rest)}")


def t_l9_negative_reward():
    """L7/L9：负奖励真实压分（修复前 max(0,·) 使点踩无效）。"""
    ba, bb = ExperienceBuffer(), ExperienceBuffer()
    i1 = ba.add("完全相同的记录内容用于对照实验。".encode(), 3.0)
    ba.feedback(i1, -1.0)
    bb.add("完全相同的记录内容用于对照实验。".encode(), 3.0)
    sa, _ = ba.select(1.0)
    sb, _ = bb.select(1.0)
    suppression = 1 - sa[0][0] / sb[0][0]
    check("L9 负奖励区分", suppression > 0.5, f"压制 {suppression:.0%}")


def t_l5_l8_training_paths():
    """L5：全部训练路径（sleep.py 与 feed.py 复刻体）无 generate 调用；回滚逐张量。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("dolphin/sleep.py", "feed.py"):  # 审计④：两条训练路径都要扫
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        check(f"L5 训练路径无自生成（{rel}）", "generate(" not in src)
        check(f"L5 训练用真实字节流（{rel}）", "varied_replay" in src)

    with _probe_fixture("dolphin_l5l8_") as probe:
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        for i in range(12):
            d.learn(f"回滚验证记录第{i}条，内容足够长。{i}", source="test")
        before = [p.clone() for p in d.awake().model.parameters()]
        r = d.maybe_sleep(force=True, steps=3)
        if r and r.get("swapped"):
            old_awake = d.sleeping()  # 换班后原醒脑变睡脑
            after = [p.clone() for p in old_awake.model.parameters()]
            unchanged = all(torch.equal(a, b) for a, b in zip(before, after))
            check("L3 服务期间的醒脑权重零改动", unchanged)
        else:
            after = [p.clone() for p in d.sleeping().model.parameters()]
            unchanged = all(torch.equal(a, b) for a, b in zip(before, after))
            check("L3 回滚路径恢复醒脑权重", unchanged)


def t_l8_gate():
    """L8：睡脑劣化时体检门控必须拒绝换班（防越学越傻的保险丝，此前零守护）。"""
    with _probe_fixture("dolphin_l8_") as probe:
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        # 修复既有 flake（2026-10-03 G3 施工发现，断言未动、只加长喂食文本）：
        # 旧文本每条仅 47B，选拔 5 条拼流 ≈234B < block_size+2=258B → 命中"样本过短"
        # 早退路径（report 无 passed/swapped 键），断言靠变异重放的供体拼接随机凑长
        # 才偶发通过（实测 5/15 失败率）。加长后选拔流确定超过训练下限。
        for i in range(12):
            d.learn(f"门控验证记录第{i}条，内容足够长以稳定通过选拔训练，避免样本过短。{i}",
                    source="test")
        before = [p.clone() for p in d.awake().model.parameters()]
        with torch.no_grad():  # 重污染睡脑：2 步训练无法恢复
            for p in d.sleeping().model.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        r = d.maybe_sleep(force=True, steps=2)
        check("L8 门控拒绝劣化睡脑", r is not None and r.get("swapped") is False,
              f"passed={r.get('passed') if r else None}")
        after = [p.clone() for p in d.awake().model.parameters()]
        check("L8 醒脑权重未被污染", all(torch.equal(a, b) for a, b in zip(before, after)))


def t_l10_dual_channel():
    """L10：换班后快通道笔记入记忆库且可检索（此前零守护）。"""
    with _probe_fixture("dolphin_l10_") as probe:
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        for i in range(12):
            # 修复既有 flake（2026-10-04 审计发现，与 G3 修L8/persistence 同缺陷族、
            # 唯独L10 被漏掉；断言未动、只加长喂食文本）：旧文本每条仅 50B，
            # 预算 45% 选出 5 条 ≈254B < block_size+2=258B —— 只差 4 字节，
            # 变异重放的供体拼接与字符 dropout 随机增减字节 → 约 8~18% 概率
            # 跌破下限，命中 dolphin/sleep.py 的"样本过短"早退（report 无
            # swapped 键）→ 零换班 → notes=0 → 断言红。实测 HEAD 版 40 次红 3 次。
            # 加长后选拔流确定超过训练下限，maybe_sleep 必然真训练。
            d.learn(f"双通道验证记录第{i}条，内容足够长以稳定通过选拔训练，避免样本过短早退。{i}",
                    source="test")
        swapped = False
        for _ in range(3):
            r = d.maybe_sleep(force=True, steps=20)
            swapped = swapped or bool(r and r.get("swapped"))
        notes = [e for e in d.memory.entries if e.kind == "note"]
        check("L10 快通道笔记随换班入记忆库", bool(swapped) and len(notes) >= 1,
              f"notes={len(notes)}")


def t_persistence_v2():
    """v2 持久化：缓冲、优化器动量、值自成常量全部随档幸存。"""
    with _probe_fixture("dolphin_pers_") as probe:
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        # 修复既有 flake（2026-10-03 G3 施工发现，断言未动、只加长喂食文本）：
        # 旧文本每条仅 47B，10 条预算 45% 只选出 4 条 ≈191B < block_size+2=258B →
        # "样本过短"早退 → 零训练 → 动量为空 → mom_ok 恒假（约 18% 概率，靠供体
        # 拼接抽签通过）。加长后选拔流确定超过训练下限，maybe_sleep 必然真训练。
        for i in range(10):
            d.learn(f"持久化回环测试记录第{i}条，内容足够长以稳定通过选拔训练。{i}",
                    source="test")
        d.maybe_sleep(force=True, steps=3)
        for i in range(6):
            d.learn(f"睡后补喂的第{i}条记录，内容足够长。{i}", source="test")
        d.note_k = 7
        d.target_interval = 9
        path = os.path.join(tempfile.gettempdir(), "dolphin_conf_test.pt")
        d.save(path)
        d2 = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                     probe_path=probe, device="cpu")
        d2.load(path)
        os.remove(path)
        buf_ok = (len(d2.buffer.items) == len(d.buffer.items) and all(
            a.data == b.data for a, b in zip(d.buffer.items, d2.buffer.items)))
        m0 = [st["exp_avg"].clone() for h in (d.h[0], d.h[1])
              for st in h.opt.state.values()]
        nonempty = len(m0) > 0  # 空转守卫：动量断言必须真有东西可比（审计② M4 变异教训）
        m1 = [st["exp_avg"].clone() for h in (d2.h[0], d2.h[1])
              for st in h.opt.state.values()]
        mom_ok = (nonempty and len(m0) == len(m1)
                  and all(torch.equal(a, b) for a, b in zip(m0, m1)))
        check("持久化 v2 缓冲/动量/常量回环",
              buf_ok and mom_ok and d2.note_k == 7 and d2.target_interval == 9,
              f"缓冲{len(d2.buffer.items)} 动量{mom_ok}(非空{nonempty})")


def t_dual_thread_no_stop():
    """M1.5：SleepTrainer 睡眠周期进行中，部署喂入持续滚动（不停机证据）。"""
    import threading
    import time
    from feed import SleepTrainer
    # 本组唯一需要"够长探测卷"的测试（其余各组用默认 12 块即可）：
    # 它量的是"一个睡眠周期进行中部署线程喂入了几条"，而喂入量正比于
    # 周期墙钟时长 —— 周期时长又正比于体检块数（实测 12 块 ≈0.17s/周期、
    # 64 块 ≈0.7s、242 块 ≈2.4s）。生产卷 242 块时余量 ~105 条；
    # 若沿用 12 块小卷，周期缩到 0.17s，喂入余量塌到个位数，
    # 在全量套件的 CPU 争抢下曾实测 fed_during=0（断言红）。
    # 断言未动（仍是 fed_during > 0），只把测量窗口恢复到与生产同量级。
    with _probe_fixture("dolphin_m15_", n_chunks=64) as probe:
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        trainer = SleepTrainer(d, steps=4, save_every=10000)
        trainer.start()
        fed = {"n": 0}
        stop = threading.Event()

        def feeder():
            i = 0
            while not stop.is_set() and i < 600:  # 修复既有竞态（断言未动）：150 条约 1s 喂完，
                d.learn(f"并发喂食记录第{i}条，内容足够长避免过滤。{i}", source="test")
                fed["n"] += 1                     # 测量窗口可能落在喂尽之后 → fed_during 恒 0
                i += 1
                time.sleep(0.003)

        th = threading.Thread(target=feeder)
        th.start()
        deadline = time.time() + 30
        # 启动期小周期：阈值 4.0 < 随机 NLL，训练线程上线即睡一次（值自成的正常表现）
        while trainer.cycles < 1 and time.time() < deadline:
            time.sleep(0.005)
        while fed["n"] < 120 and time.time() < deadline:  # 等缓冲真有存量再测量（原注释声明的意图）：
            time.sleep(0.005)                             # 存量足够 → 被测周期必然真训练而非"样本过短"瞬回
        n_before = fed["n"]
        c0 = trainer.cycles
        trainer.request_sleep()              # 触发被测量的周期（此时缓冲已有真实存量）
        deadline = time.time() + 30
        while trainer.cycles <= c0 and time.time() < deadline:  # 等"在 request_sleep 之后完成"的周期：
            time.sleep(0.005)                                 # 启动期周期可能已把 cycles 推过 2（旧竞态）
        fed_during = fed["n"] - n_before     # 周期进行中部署线程的喂入量 = 不停机证据
        stop.set()
        trainer.shutdown()
        th.join()
        trainer.join(timeout=10)
        check("双线程并发喂入不阻塞", fed_during > 0, f"训练周期内喂入 {fed_during} 条")
        check("睡眠周期完整收尾", trainer.cycles >= 2, f"cycles={trainer.cycles}")


# ==================== 2026-10-03 G1/G2 骨头施工新增（自成长完备性审计实锤回归） ====================

def t_g1_cold_layer():
    """G1-a：逐出=降级落盘冷层，零丢失；冷层可检索、可复活、跨实例可重载。"""
    import json
    import shutil
    from dolphin.memory import MemoryStore
    tmp = tempfile.mkdtemp(prefix="dolphin_g1a_")
    try:
        cold = os.path.join(tmp, "memory_cold.jsonl")
        m = MemoryStore(cold_path=cold)  # cap=500 默认档
        texts = [f"冷层回归条目第{i:03d}号，逐字可核对零丢失。{i}" for i in range(501)]
        for t in texts:
            m.add(t.encode(), 0.0, 0, "residue")
        hot = {e.text for e in m.entries}
        check("G1-a 满库逐出不扩容（零命中候选尚在）",
              len(m.entries) == 500 and m.cap == 500, f"{len(m.entries)}/cap={m.cap}")
        lines = [json.loads(l) for l in open(cold, encoding="utf-8") if l.strip()]
        ev = lines[0]["text"] if lines else None
        check("G1-a 冷层文件恰好记录被逐出条目", len(lines) == 1 and ev == texts[0],
              f"{len(lines)} 行")
        check("G1-a 冷层 JSON 行字段完整",
              bool(lines) and set(lines[0]) == {"text", "score", "hits", "cycle", "kind", "created"})
        check("G1-a 零丢失（热∪冷=全部、交为空）",
              ev is not None and (hot | {ev}) == set(texts) and ev not in hot)
        found = m.retrieve(ev.encode(), k=1)
        check("G1-a 冷层可检索（旧实现永久查无此人）",
              bool(found) and found[0].text == ev and found[0].hits == 1)
        m.retrieve(ev.encode(), k=1)
        m.retrieve(ev.encode(), k=1)
        m.resurrect()
        check("G1-a 冷层命中达标复活入热层",
              any(e.text == ev for e in m.entries)
              and all(e.text != ev for e in m.cold_entries),
              f"热层{len(m.entries)}条")
        m.flush_cold()
        m2 = MemoryStore(cold_path=cold)
        m2._ensure_cold()
        check("G1-a 冷层落盘跨实例重载一致",
              {e.text for e in m2.cold_entries} == {e.text for e in m.cold_entries},
              f"重载{len(m2.cold_entries)}条")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_g1_no_suicide():
    """G1-b：全员 hits>0 时 add 必须 +1（新条目自杀 bug 回归）；扩容=事件驱动值自成。"""
    import shutil
    from dolphin.memory import MemoryStore
    tmp = tempfile.mkdtemp(prefix="dolphin_g1b_")
    try:
        cold = os.path.join(tmp, "c.jsonl")
        m = MemoryStore(cap_entries=10, cold_path=cold)
        for i in range(10):
            m.add(f"自杀回归旧条目第{i}号，内容足够。{i}".encode(), 0.0, 0, "residue")
        for e in m.entries:
            e.hits = 1  # 旧实现：zero 集只含新条目 → min(created) 选中它自己 → add 净删除
        m.add("全家被检索过时到来的新条目，绝不能被自己顶替。".encode(), 0.0, 0, "residue")
        check("G1-b 新条目存活（库存 +1）", len(m.entries) == 11, f"{len(m.entries)}")
        check("G1-b cap 事件驱动扩容一档", m.cap == 15, f"cap={m.cap}")
        cold_empty = (not os.path.exists(cold)) or open(cold, encoding="utf-8").read().strip() == ""
        check("G1-b 扩容不是逐出（冷层零写入）", cold_empty)

        m1 = MemoryStore(cap_entries=10, cold_path=os.path.join(tmp, "c1.jsonl"))
        for i in range(10):
            m1.add(f"旧条目{i}号，用于逐旧不逐新对照。{i}".encode(), 0.0, 0, "residue")
        m1.entries[0].hits = 1  # 只有一条被检索过
        m1.add("新条目必须幸存于逐出。".encode(), 0.0, 0, "residue")
        hot1 = {e.text for e in m1.entries}
        check("G1-b 有零命中旧条目时逐旧不逐新",
              len(m1.entries) == 10 and "新条目必须幸存于逐出。" in hot1
              and "旧条目0号，用于逐旧不逐新对照。0" in hot1
              and "旧条目1号，用于逐旧不逐新对照。1" not in hot1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_g1_dream_feed():
    """G1-c：喂食模式做梦采样注入发生，hits 由做梦供血（复活通道不再死亡）。"""
    from dolphin.memory import MemoryStore
    from feed import trainer_cycle
    # 冷层落盘与探测集都隔离到临时目录：吃生产默认 probes/probe.txt 会让
    # 探测集一挪走就G8 fail-fast 拒绝开睡，做梦注入照常但断言全灭。
    with _probe_fixture("dolphin_g1c_") as probe:
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        d.memory = MemoryStore(cold_path=os.path.join(os.path.dirname(probe), "c.jsonl"))
        cue = "做梦联想线索条目：海豚用回声定位寻找沙丁鱼群，记忆库旧事。"
        d.memory.add(cue.encode(), 3.0, 0, "residue")
        d.learn(f"喂食记录携带与旧事重叠的联想线索：{cue}", source="test")
        hits0 = d.memory.entries[0].hits
        report = trainer_cycle(d, steps=2)
        check("G1-c 做梦注入发生", report.get("dreamed", 0) >= 1,
              f"dreamed={report.get('dreamed')}")
        check("G1-c 做梦计入检索命中（复活供血）",
              d.memory.entries[0].hits == hits0 + 1,
              f"hits {hits0}→{d.memory.entries[0].hits}")


def _write_synthetic_volume(path, n_bytes):
    """写一份指定字节数的合成卷（不必是block_size 整数倍——用于验证尾块路径）。"""
    body = (_PROBE_SEG * (n_bytes // len(_PROBE_SEG) + 1))[:n_bytes]
    with open(path, "wb") as f:
        f.write(body)
    return path


def t_g2_full_coverage():
    """G2-a：体检全卷覆盖——10.1% 缺陷（max_chunks=48 截断 + 50% 重叠）回归。

    历史缺陷（对齐 dolphin/probe.py 模块注释）：旧实现 load_chunks 只取前
    max_chunks=48 块且相邻块 50% 重叠，在 61.9KB 生产卷上实际只体检了 10.1%
    的字节——四分之九十的卷面从未参与评分，门控在自欺。现实现为全卷不重叠切分，
    覆盖率恒等式 sum(len(c)) == 文件总字节 是结构性回归点。

    分三层（断言只增不减）：
      ① 合成小卷验切分算法性质（含非整倍尾块）——与生产卷内容无关，纯算法。
      ② 合成大卷验规模——必须造到 242 块（≈生产卷 61.9KB 同量级），
         否则 len(chunks) > 48 这条防回归断言在小卷上恒真、等于白写：
         48 块截断缺陷只有在卷面远大于 48*256 字节时才会显形。
      ③ 生产探测集只读守卫——在场时核对真实卷的切分性质与规模；不在场则跳过，
         绝不崩掉、绝不写入（律 L8：探测集是生产资产，G7 另有 sha256 校验）。
    """
    from dolphin.probe import evaluate, load_chunks
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ① 合成小卷：1317B = 5*256 + 37 → 6 块，尾块 37 字节（非整倍，逼出尾块路径）
    with _probe_fixture("dolphin_g2a_small_") as small:
        _write_synthetic_volume(small, 1317)
        total_s = os.path.getsize(small)
        cs = load_chunks(small, 256)
        check("G2-a 合成小卷 覆盖字节 == 全卷字节",
              sum(len(c) for c in cs) == total_s,
              f"{sum(len(c) for c in cs)}/{total_s}")
        check("G2-a 合成小卷 块数=不重叠切分",
              len(cs) == (total_s - 1 + 255) // 256 == 6,
              f"{len(cs)} 块（{total_s}B）")
        check("G2-a 合成小卷 尾块保留且可评分（≥2 字节）",
              all(len(c) >= 2 for c in cs) and len(cs[-1]) == 37,
              f"尾块 {len(cs[-1])}B")
        # 无重叠无截断的最强形式：按顺序拼回原卷逐字节相等
        # （50% 重叠的旧实现会让拼接结果长度虚增、字节错位）
        raw_s = open(small, "rb").read()
        check("G2-a 合成小卷 拼接还原 == 原卷（无重叠无错位）",
              b"".join(cs) == raw_s,
              f"拼回 {len(b''.join(cs))}B vs 原卷 {len(raw_s)}B")

    # ② 合成大卷：242 块 ≈ 61.9KB，与生产卷同量级——48 块截断缺陷的照妖镜
    with _probe_fixture("dolphin_g2a_big_") as big:
        _write_synthetic_probe(big, 242)
        total_b = os.path.getsize(big)
        cb = load_chunks(big, 256)
        check("G2-a 合成大卷 覆盖字节 == 全卷字节（旧缺陷仅 10.1%）",
              sum(len(c) for c in cb) == total_b,
              f"{sum(len(c) for c in cb)}/{total_b}（100%）")
        check("G2-a 合成大卷 块数=不重叠切分（旧缺陷仅 48 块）",
              len(cb) == (total_b - 1 + 255) // 256 and len(cb) > 48,
              f"{len(cb)} 块 / {total_b}B")
        check("G2-a 合成大卷 每块皆满块（无截断无重叠）",
              all(len(c) == 256 for c in cb), f"全 {len(cb)} 块均 256B")

    # evaluate 契约：返回 (mean, per_chunk, n)，且 mean == per 均值（合成小卷驱动）
    with _probe_fixture("dolphin_g2a_eval_") as ev:
        _write_synthetic_volume(ev, 1317)
        ce = load_chunks(ev, 256)
        cfg = Config(d_model=32, n_layers=1, n_heads=2, block_size=256)
        mean, per, n = evaluate(ByteTransformer(cfg), ce[:3], "cpu")
        check("G2-a evaluate 返回 (mean, per_chunk, n)",
              isinstance(per, list) and len(per) == n == 3
              and abs(mean - sum(per) / 3) < 1e-9)
        mean_all, per_all, n_all = evaluate(ByteTransformer(cfg), ce, "cpu")
        check("G2-a evaluate 全卷块数自洽（尾块也进评分）",
              n_all == len(ce) and len(per_all) == n_all
              and abs(mean_all - sum(per_all) / n_all) < 1e-9,
              f"n={n_all}/{len(ce)} 块")

    # ③ 生产探测集只读守卫：存在则核对真实卷，不存在则跳过（不崩、不写）
    prod = os.path.join(root, "probes", "probe.txt")
    if os.path.exists(prod):
        total_p = os.path.getsize(prod)
        cp = load_chunks(prod, 256)  # 只读
        check("G2-a 生产探测集 覆盖字节 == 全卷字节",
              sum(len(c) for c in cp) == total_p,
              f"{sum(len(c) for c in cp)}/{total_p}")
        check("G2-a 生产探测集 块数=不重叠切分且 >48（242 块）",
              len(cp) == (total_p - 1 + 255) // 256 and len(cp) > 48,
              f"{len(cp)} 块 / {total_p}B")
        check("G2-a 生产探测集 尾块保留且可评分（≥2 字节）",
              all(len(c) >= 2 for c in cp), f"尾块 {len(cp[-1])}B")
    else:
        check("G2-a 生产探测集守卫（缺失则跳过，只读不写）", True,
              f"{prod} 不在场", skip=True)


def t_g2_gate_margin():
    """G2-b：margin 判决带——0.5×SE 拒绝、3×SE 放行、零差/劣化拒绝（反掷硬币）。"""
    import statistics
    from dolphin.probe import gate_decision
    n = 40
    noise = [0.1 if i % 2 == 0 else -0.1 for i in range(n)]
    se = statistics.stdev(noise) / n ** 0.5  # 与 gate_decision 同式：样本标准差 / √n
    base = [5.0 + (i % 7) * 0.01 for i in range(n)]
    p_rej, d_rej = gate_decision(base, [o - (0.5 * se + noise[i]) for i, o in enumerate(base)])
    p_ok, _ = gate_decision(base, [o - (3 * se + noise[i]) for i, o in enumerate(base)])
    p_eq, _ = gate_decision(base, base)
    p_bad, _ = gate_decision(base, [o + 0.5 for o in base])
    check("G2-b 亚噪声改善 0.5×SE 拒绝", p_rej is False,
          f"margin={d_rej['gate_margin']} eps={d_rej['gate_eps']}")
    check("G2-b 真实改善 3×SE 放行", p_ok is True)
    check("G2-b 零改善（同源双脑）拒绝", p_eq is False)
    check("G2-b 劣化拒绝", p_bad is False)
    check("G2-b SE 全部由数据估计", d_rej["chunks"] == n and d_rej["gate_se"] > 0,
          f"n={d_rej['chunks']} se={d_rej['gate_se']}")
    # 同步守卫（M4 漂移警告）：两条训练路径都必须走公共判决，掷硬币判决不得复活
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel, call in (("dolphin/sleep.py", "dolphin.gate("), ("feed.py", "d.gate(")):
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        check(f"G2-b 公共判决唯一接入（{rel}）",
              call in src and "probe_new < probe_old" not in src)


def t_g2_probe_rotation():
    """G2-c：试卷池自滚动——转正/退休/指针推进/池空不动卷/UTF-8 边界零割裂。"""
    import shutil
    from probes.rolling import rotate, status
    tmp = tempfile.mkdtemp(prefix="dolphin_g2c_")
    try:
        probe = os.path.join(tmp, "probe.txt")
        pool = os.path.join(tmp, "pool")
        retired = os.path.join(tmp, "retired")
        os.makedirs(pool)
        open(probe, "wb").write(b"A" * 100 + b"B" * 100 + b"C" * 100)
        open(os.path.join(pool, "cand_0001.txt"), "wb").write(b"X" * 100 + b"Y" * 100)
        open(os.path.join(pool, "cand_0002.txt"), "wb").write(b"Z" * 100)

        r1 = rotate(probe_path=probe, pool_dir=pool, retired_dir=retired, section_bytes=100)
        c1 = open(probe, "rb").read()
        check("G2-c 转正入卷 + 最旧段退休",
              r1["rotated"] and c1 == b"B" * 100 + b"C" * 100 + b"X" * 100)
        ret_files = sorted(os.listdir(retired))
        check("G2-c 退休档落盘（可回流训练粮）",
              len(ret_files) == 1
              and open(os.path.join(retired, ret_files[0]), "rb").read() == b"A" * 100)
        rotate(probe_path=probe, pool_dir=pool, retired_dir=retired, section_bytes=100)
        c2 = open(probe, "rb").read()
        check("G2-c 指针推进：同文件第二节转正",
              c2 == b"C" * 100 + b"X" * 100 + b"Y" * 100)
        rotate(probe_path=probe, pool_dir=pool, retired_dir=retired, section_bytes=100)
        c3 = open(probe, "rb").read()
        check("G2-c 跨文件取候选", c3 == b"X" * 100 + b"Y" * 100 + b"Z" * 100)
        r4 = rotate(probe_path=probe, pool_dir=pool, retired_dir=retired, section_bytes=100)
        check("G2-c 池空不动卷",
              r4["rotated"] is False and open(probe, "rb").read() == c3, r4.get("reason", ""))
        st = status(probe_path=probe, pool_dir=pool, retired_dir=retired, section_bytes=100)
        check("G2-c 现状对账（池尽/退休3段/卷面守恒）",
              st["pool_bytes_left"] == 0 and st["retired_files"] == 3
              and st["probe_bytes"] == 300,
              f"池{st['pool_bytes_left']}B 退{st['retired_files']} 卷{st['probe_bytes']}B")

        seg = ("甲" * 40).encode()  # 120 字节
        probe2 = os.path.join(tmp, "probe2.txt")
        pool2 = os.path.join(tmp, "pool2")
        ret2 = os.path.join(tmp, "retired2")
        os.makedirs(pool2)
        open(probe2, "wb").write(seg * 3)
        open(os.path.join(pool2, "c1.txt"), "wb").write(("丁" * 40).encode())
        r5 = rotate(probe_path=probe2, pool_dir=pool2, retired_dir=ret2, section_bytes=98)
        c5 = open(probe2, "rb").read()
        check("G2-c 多字节字符零割裂",
              c5.decode("utf-8") == "甲" * 88 + "丁" * 32
              and open(os.path.join(ret2, r5["retired"]), "rb").read().decode("utf-8") == "甲" * 32)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==================== 2026-10-03 G3/G7/G8/G9 骨头施工新增（自续骨+护航骨回归） ====================

def t_g3_guard_ruminate():
    """G3-a：数据尽后守护模式不退场——反刍注入经部署管线、周期照常发生、
    部署侧流入自动恢复正式喂食、记忆库空时 max-idle 收工（无死循环）。"""
    import shutil
    import threading
    import time as _time
    from feed import SleepTrainer, guard_loop

    # 测试夹具自造探测集，不绑死生产 probes/probe.txt（与 t_g8_gate_failfast 同一套做法）：
    # 本组要验的是反刍自续，不是探测集。若吃生产默认路径，探测卷一旦被挪走/改名/清空，
    # G8 fail-fast 会让 trainer_cycle 拒绝开睡 → 反刍注入照常但周期恒为 0，
    # 且场景 2 的 guard_loop 因idle_cycles永不递增而死挂——让一个无关的外部文件
    # 成为本组通过的前提。生产律不动，这里只换夹具。
    tmp = tempfile.mkdtemp(prefix="dolphin_g3a_")
    try:
        probe = os.path.join(tmp, "probe.txt")
        # 自然语言逻辑片段（非随机字节：门控在随机数据上没意义），
        # 补齐到 block_size 整数倍 → 19 块完整块，probe.evaluate 可正常算 NLL。
        seg = ("逻辑推理探测片段：若甲高于乙，乙高于丙，则甲高于丙；"
               "所有金属都导电，铁是金属，故铁导电；下雨地必湿，此地不湿，"
               "故此地未必下雨；鸟会飞，企鹅是鸟，然企鹅不会飞，故前提有误；"
               "此段仅供体检评分，永不进训练粮。").encode()
        body = (seg * 16).ljust(4864, b" ")  # 4864 = 19 * block_size(256)
        open(probe, "wb").write(body)

        # 场景 1：记忆库有料 → 反刍自续；外部流入 → 恢复正式喂食
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        for i in range(3):
            d.memory.add(f"反刍旧事第{i}条：海豚用回声定位寻找沙丁鱼群，记忆库里的旧经验。{i}".encode(),
                         3.0, 0, "residue")
        trainer = SleepTrainer(d, steps=2, save_every=10000)
        trainer.start()
        result = {}

        def run_guard():
            result["out"] = guard_loop(d, trainer, max_idle=2)

        th = threading.Thread(target=run_guard, daemon=True)
        th.start()
        deadline = _time.time() + 30
        while _time.time() < deadline and (d.learn_total < 1 or trainer.cycles < 1):
            _time.sleep(0.01)  # 等反刍注入与周期发生的实证
        check("G3-a 反刍材料经部署管线注入且周期照常发生",
              d.learn_total >= 1 and trainer.cycles >= 1,
              f"注入 {d.learn_total} 条  周期 {trainer.cycles} 次")
        d.learn("部署侧新经验流入：守护模式应自动恢复正式喂食，绝不退场。", source="deploy")
        th.join(timeout=15)
        out = result.get("out", (False, 0))
        check("G3-a 检测到流入自动恢复正式喂食（非死守反刍）",
              not th.is_alive() and out[0] is True and out[1] >= 1,
              f"resume={out[0]} 反刍 {out[1]} 条")
        trainer.shutdown()
        trainer.join(timeout=10)

        # 场景 2：记忆库也空 → 连续空选拔周期 → --max-idle 收工（防真死转）
        d2 = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                     probe_path=probe, device="cpu")
        trainer2 = SleepTrainer(d2, steps=2, save_every=10000)
        trainer2.start()
        t0 = _time.time()
        resumed2, fed2 = guard_loop(d2, trainer2, max_idle=2)  # 应自行收敛返回
        trainer2.shutdown()
        trainer2.join(timeout=10)
        check("G3-a 记忆库空时 max-idle 收工（无死循环）",
              resumed2 is False and fed2 == 0 and trainer2.idle_cycles >= 2,
              f"resume={resumed2} idle={trainer2.idle_cycles}  耗时 {_time.time() - t0:.1f}s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_g3_feed_cursor():
    """G3-b：喂食游标存读档一致；--resume 语义下从游标续喂，不再从第 0 条重放。"""
    import feed as feed_mod
    from feed import feed_from
    orig_iter = feed_mod.iter_source
    try:
        def mk(i):
            return {"prompt": f"游标验证题目第{i}号：甲乙丙丁戊己庚辛壬癸顺序编号。",
                    "answer": f"标准答案正文第{i}号，长度足以通过碎片过滤线。", "source": "fake"}
        src_a = [mk(i) for i in range(4)]
        src_b = [mk(10 + i) for i in range(3)]
        feed_mod.iter_source = lambda name: iter(src_a if name == "fakeA" else src_b)
        with _probe_fixture("dolphin_g3b_") as probe:
            d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                        probe_path=probe, device="cpu")
            names = ["fakeA", "fakeB"]
            cursor = {"sources": list(names), "source_idx": 0, "record_idx": 0}
            d.feed_cursor = cursor
            fed, hit, _ = feed_from(d, names, cursor, {})
            check("G3-b 全量喂入且游标推进到尽头", fed == 7 and hit is False
                  and cursor == {"sources": names, "source_idx": 2, "record_idx": 0},
                  f"fed={fed} cursor={cursor}")
            path = os.path.join(tempfile.gettempdir(), "dolphin_g3b_state.pt")
            d.save(path)
            d2 = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                         probe_path=probe, device="cpu")
            d2.load(path)
            os.remove(path)
            check("G3-b 游标随档存读一致", d2.feed_cursor == cursor, f"{d2.feed_cursor}")
            # 续喂：游标 (0,2) → 跳过 fakeA 前两条
            d3 = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                         probe_path=probe, device="cpu")
            cursor3 = {"sources": list(names), "source_idx": 0, "record_idx": 2}
            d3.feed_cursor = cursor3
            fed3, _, _ = feed_from(d3, names, cursor3, {})
            first = d3.buffer.items[0].data.decode("utf-8", errors="replace")
            check("G3-b 续喂从游标起（跳过已吃记录）",
                  fed3 == 2 + 3 and first == feed_mod.serialize(src_a[2]), f"fed3={fed3}")
    finally:
        feed_mod.iter_source = orig_iter


def t_g3_structural_reward():
    """G3-c：数据集内重复条目的自动负奖励真实写进 reward 并压分（端到端）。"""
    import feed as feed_mod
    from feed import feed_from
    orig_iter = feed_mod.iter_source
    try:
        rec = {"prompt": "重复条目压分验证题干，一字不差地出现三次以检验结构信号。",
               "answer": "标准答案内容同样保持一字不差。", "source": "fake"}
        feed_mod.iter_source = lambda name: iter([rec, dict(rec), dict(rec)])
        with _probe_fixture("dolphin_g3c_") as probe:
            d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                        probe_path=probe, device="cpu")
            names = ["dup"]
            cursor = {"sources": names, "source_idx": 0, "record_idx": 0}
            fed, _, _ = feed_from(d, names, cursor, {})
            rewards = [e.reward for e in d.buffer.items]
            check("G3-c 结构信号：首现零、重复递减", fed == 3 and rewards == [0.0, -1.0, -2.0],
                  f"{rewards}")
            sel, _ = d.buffer.select(1.0)  # 预算拉满：三条全部入榜
            head = max(s for s, e in sel if e.reward == 0.0)
            dup_scores = sorted((s for s, e in sel if e.reward < 0), reverse=True)
            check("G3-c 重复条目被负奖励压到原件之后",
                  len(dup_scores) >= 2 and dup_scores[-1] < head,
                  f"重复条目分数 {dup_scores} vs 原件 {head:.4f}")
    finally:
        feed_mod.iter_source = orig_iter


def t_g7_atomic_save():
    """G7-a：save 中途断电（torch.save 抛异常）→ 旧档完好、临时残尸清理。"""
    import shutil
    tmp = tempfile.mkdtemp(prefix="dolphin_g7a_")
    try:
        probe = os.path.join(tmp, "probe.txt")
        open(probe, "wb").write("体检探针内容，用于哈希校验与门控。".encode() * 20)
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        for i in range(6):
            d.learn(f"原子写验证记录第{i}条，内容足够长。{i}", source="test")
        path = os.path.join(tmp, "state.pt")
        d.save(path)
        good = torch.load(path, weights_only=True)
        orig = torch.save
        crashed = False

        def blow(*a, **k):
            raise OSError("断电模拟：torch.save 中途崩溃")

        torch.save = blow
        try:
            d.save(path)
        except OSError:
            crashed = True
        finally:
            torch.save = orig
        check("G7-a 断电模拟抛出（不吞异常）", crashed)
        check("G7-a 临时文件残尸已清理", not os.path.exists(path + ".saving"))
        after = torch.load(path, weights_only=True)
        same = all(torch.equal(good["states"][i][k], after["states"][i][k])
                   for i in range(2) for k in good["states"][i])
        check("G7-a 旧档完好（权重逐张量一致）", same and after["version"] == 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_g7_rolling_backup():
    """G7-b：滚动备份——存 5 次留 3 份（K=3 律定），最旧被删。"""
    import shutil
    import time as _time
    tmp = tempfile.mkdtemp(prefix="dolphin_g7b_")
    try:
        probe = os.path.join(tmp, "probe.txt")
        open(probe, "wb").write("滚动备份验证探测内容。".encode() * 10)
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        d.learn("滚动备份验证记录，内容足够长。", source="test")
        path = os.path.join(tmp, "state.pt")
        for _ in range(5):
            d.save(path)
            _time.sleep(0.01)  # 时间戳推进，防同微秒撞名干扰计数
        baks = sorted(os.listdir(os.path.join(tmp, "archive")))
        check("G7-b 存 5 次留 3 份", len(baks) == 3, f"{len(baks)} 份")
        check("G7-b 备份带时间戳命名（字典序=时间序）",
              all(b.startswith("state.pt.") and b.endswith(".bak") for b in baks),
              f"{baks}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_g7_eid_stability():
    """G7-c：身份入档——逐出+存档+重启后幸存条目 eid 不平移，feedback 命中原条目。"""
    import shutil
    tmp = tempfile.mkdtemp(prefix="dolphin_g7c_")
    try:
        probe = os.path.join(tmp, "probe.txt")
        open(probe, "wb").write("身份平移验证探测内容。".encode() * 10)
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        d.buffer.cap = 400  # 压小缓冲逼逐出（ExperienceBuffer 的容量属性为 cap）
        texts = {}
        for i in range(8):
            t = f"逐出平移验证记录第{i}条，内容足够长触发缓冲逐出。{i}"
            texts[d.learn(t, source="test")] = t
        survivors = [e.id for e in d.buffer.items]
        check("G7-c 逐出确实发生", 0 < len(survivors) < 8, f"幸存 {len(survivors)}/8")
        path = os.path.join(tmp, "state.pt")
        d.save(path)
        d2 = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                     probe_path=probe, device="cpu")
        d2.load(path)
        check("G7-c 幸存条目 eid 逐字复原（不平移）",
              [e.id for e in d2.buffer.items] == survivors,
              f"{[e.id for e in d2.buffer.items]} vs {survivors}")
        target = survivors[-1]
        check("G7-c feedback 命中原条目", d2.feedback(target, -0.5) is True)
        hit = [e for e in d2.buffer.items if e.id == target][0]
        check("G7-c 奖励挂到正确内容上",
              hit.reward == -0.5 and hit.data.decode("utf-8") == texts[target])
        nid = d2.learn("重启后的新记录，绝不与幸存条目撞身份。")
        check("G7-c 新经验 id 接续计数器（不撞幸存条目）",
              nid not in survivors and nid >= max(survivors) + 1, f"new id={nid}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_g7_probe_hash():
    """G7-d：probe.txt 被篡改后 load 拒绝（门控语义完整性优先于可用性）。"""
    import shutil
    tmp = tempfile.mkdtemp(prefix="dolphin_g7d_")
    try:
        probe = os.path.join(tmp, "probe.txt")
        body = "体检探测集原文，哈希校验用。".encode() * 30
        open(probe, "wb").write(body)
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        d.learn("哈希校验验证记录，内容足够长。", source="test")
        path = os.path.join(tmp, "state.pt")
        d.save(path)
        with open(probe, "ab") as f:  # 篡改：追加一字节即改变卷面语义
            f.write(b"X")
        d2 = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                     probe_path=probe, device="cpu")
        rejected = False
        try:
            d2.load(path)
        except RuntimeError as e:
            rejected = "探测" in str(e) or "sha256" in str(e)
        check("G7-d 篡改 probe 后 load 拒绝", rejected)
        open(probe, "wb").write(body)  # 恢复原文
        d3 = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                     probe_path=probe, device="cpu")
        d3.load(path)  # 不抛即放行
        check("G7-d probe 恢复后 load 放行", d3.awake_idx == d.awake_idx)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_g8_gate_failfast():
    """G8：probe 缺失 → 醒目报警 + 拒绝开睡（GateDisabled），恢复后自动解禁自愈。"""
    import contextlib
    import io
    import shutil
    import time as _time
    from dolphin.dolphin import GateDisabled
    from feed import SleepTrainer, trainer_cycle
    tmp = tempfile.mkdtemp(prefix="dolphin_g8_")
    try:
        probe = os.path.join(tmp, "probe.txt")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                        probe_path=probe, device="cpu")
        check("G8 缺失即置 gate_disabled 并 stderr 醒目报警",
              d.gate_disabled is True and "探测集" in err.getvalue())
        for i in range(12):
            d.learn(f"门控失效验证记录第{i}条，内容足够长以稳定通过选拔训练。{i}",
                    source="test")
        raised = False
        try:
            d.maybe_sleep(force=True, steps=2)
        except GateDisabled:
            raised = True
        check("G8 单线程睡眠路径拒绝开睡（fail-fast 非静默）", raised)
        raised2 = False
        try:
            trainer_cycle(d, steps=2)
        except GateDisabled:
            raised2 = True
        check("G8 喂食训练路径拒绝开睡（同一入口）", raised2)
        # SleepTrainer 端到端：拒绝开睡 = 零周期、零静默回滚，部署喂入不被拖死
        trainer = SleepTrainer(d, steps=2, save_every=10000)
        trainer.start()
        trainer.request_sleep()
        _time.sleep(1.0)
        trainer.shutdown()
        trainer.join(timeout=5)
        check("G8 训练线程零周期零回滚（部署不空转）",
              trainer.cycles == 0 and trainer.rollbacks == 0,
              f"cycles={trainer.cycles}")
        open(probe, "wb").write("恢复后的体检探测内容，足以逐块评分。".encode() * 30)
        err2 = io.StringIO()
        with contextlib.redirect_stderr(err2):
            r = d.maybe_sleep(force=True, steps=2)
        check("G8 probe 恢复后自动解禁并正常开睡",
              d.gate_disabled is False and r is not None and "恢复" in err2.getvalue(),
              f"passed={r.get('passed') if r else None}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_g9_threshold_clamp():
    """G9：极端反馈下睡眠阈值不出域 [1.0, 50.0]（律定域，端点值自成）。"""
    from dolphin.dolphin import clamp_threshold
    from feed import _threshold_feedback
    check("G9 clamp 单元：上界/下界/恒等",
          clamp_threshold(1e12) == 50.0 and clamp_threshold(-3.0) == 1.0
          and clamp_threshold(7.5) == 7.5)
    with _probe_fixture("dolphin_g9_") as probe:
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                    probe_path=probe, device="cpu")
        d.buffer.sleep_threshold = 1e12
        d._since_sleep = d.target_interval * 3  # 睡得太稀 → 降
        _threshold_feedback(d)
        check("G9 喂食反馈路径：天文数字压回上界", d.buffer.sleep_threshold == 50.0,
              f"{d.buffer.sleep_threshold}")
        d.buffer.sleep_threshold = 1e-9
        d._since_sleep = 0  # 睡得太勤 → 抬
        _threshold_feedback(d)
        check("G9 喂食反馈路径：趋零抬回下界", d.buffer.sleep_threshold == 1.0,
              f"{d.buffer.sleep_threshold}")
        for i in range(12):
            d.learn(f"阈值钳位验证记录第{i}条，内容足够长以稳定通过选拔训练。{i}", source="test")
        d.buffer.sleep_threshold = 1e12
        d._since_sleep = d.target_interval * 3
        d.maybe_sleep(force=True, steps=2)
        check("G9 maybe_sleep 路径同样钳位",
              1.0 <= d.buffer.sleep_threshold <= 50.0, f"{d.buffer.sleep_threshold}")


if __name__ == "__main__":
    print("== 律符合度常备测试 ==")
    t_l1_bytes()
    t_l4_band()
    t_l6_variation()
    t_l7_decay()
    t_l9_no_evaporation()
    t_l9_negative_reward()
    t_l5_l8_training_paths()
    t_l8_gate()
    t_l10_dual_channel()
    t_persistence_v2()
    t_dual_thread_no_stop()
    t_g1_cold_layer()
    t_g1_no_suicide()
    t_g1_dream_feed()
    t_g2_full_coverage()
    t_g2_gate_margin()
    t_g2_probe_rotation()
    t_g3_guard_ruminate()
    t_g3_feed_cursor()
    t_g3_structural_reward()
    t_g7_atomic_save()
    t_g7_rolling_backup()
    t_g7_eid_stability()
    t_g7_probe_hash()
    t_g8_gate_failfast()
    t_g9_threshold_clamp()
    n_ok = sum(PASS)
    if SKIP:
        print(f"（跳过 {len(SKIP)} 项：{'; '.join(SKIP)}）")
    print(f"\n== 结论：{n_ok}/{len(PASS)} 绿 ==")
    sys.exit(0 if n_ok == len(PASS) else 1)
