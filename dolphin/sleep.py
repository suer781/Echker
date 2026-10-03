"""睡眠周期（律 L5-L10）。

流程：复活（间隔重复）→ 选拔（带通+去重+预算）→ 滞流入记忆库 →
变异重放（改写不复读）→ 双损失训练（硬标签 + 蒸馏软标签，锚定真实数据）→
体检门控 → 换班 / 作废回滚 → 快通道交接 → 预算与学习率反馈。
"""
import random

import torch
import torch.nn.functional as F

PUNCT = "。！？；\n.!?;"


def _sentences(text):
    out, cur = [], ""
    for ch in text:
        cur += ch
        if ch in PUNCT:
            if len(cur.strip()) > 1:
                out.append(cur)
            cur = ""
    if len(cur.strip()) > 1:
        out.append(cur)
    return out


def varied_replay(data: bytes, rng: random.Random, donor=None) -> bytes:
    """律 L6：重放 = 改写。

    换序、跨经验拼接、字符 dropout。逐字复读会语义饱和（只剩笔画丢了意义），
    改写才能逼出要点。
    """
    text = data.decode("utf-8", errors="replace")
    sents = _sentences(text)
    if len(sents) >= 2 and rng.random() < 0.7:
        rng.shuffle(sents)
    if donor and rng.random() < 0.35:
        ds = _sentences(donor.decode("utf-8", errors="replace"))
        if ds:
            sents.insert(rng.randrange(len(sents) + 1), rng.choice(ds))
    body = "".join(sents)
    # L6 第四算子"遮盖重填"：字符池=原文+供体（跨经验取材；不从模型自生成内容取材，L5 安全）
    pool = body + (donor.decode("utf-8", errors="replace") if donor else "")
    chars = [rng.choice(pool) if (ch != "\n" and rng.random() < 0.02) else ch for ch in body]
    return "".join(chars).encode("utf-8", errors="replace")


def run_cycle(dolphin, steps=40, kd_alpha=0.5, kd_T=2.0, verbose=True):
    # G8 fail-fast：探测集缺失时抛 GateDisabled 拒绝开睡（与 feed.trainer_cycle
    # 共用 Dolphin.ensure_gate_ready 同一入口）——取代旧版"体检 inf<inf 恒假 →
    # 每周期正常训练后无条件作废回滚"的静默永久回滚。probe 恢复后自动解禁。
    dolphin.ensure_gate_ready()
    buf, rng, dev = dolphin.buffer, dolphin.rng, dolphin.device
    awake, sleeping = dolphin.awake(), dolphin.sleeping()
    report = {"cycle": dolphin.cycle, "candidates": len(buf.items)}

    # 间隔重复：记忆库中命中达阈值的条目复活，以带通中心惊讶度重新入选拔
    center, _ = buf.band()
    for rb in dolphin.memory.resurrect():
        buf.add(rb, surprise=center)

    sel, rest = buf.select(dolphin.budget)
    report["selected"], report["residue"] = len(sel), len(rest)

    # L9 周期层：落选/未选拔经验一律降级进记忆库（审计③-1：空选拔周期 rest 不得蒸发）
    for s, e in rest:
        dolphin.memory.add(e.data, s, dolphin.cycle, "residue")

    if not sel:
        buf.clear()
        dolphin.cycle += 1
        report["note"] = "无可训经验（滞留已全部入记忆库）"
        return report

    # 律 L6：变异重放，拼成一条字节流
    donors = [e.data for _, e in sel]
    parts = []
    for i, (_, e) in enumerate(sel):
        donor = donors[(i + 1) % len(donors)] if len(donors) > 1 else None
        parts.append(varied_replay(e.data, rng, donor))
    stream = b"\n".join(parts)

    blk = dolphin.cfg.block_size
    bt = torch.tensor(list(stream), dtype=torch.long, device=dev)
    if bt.numel() < blk + 2:
        for s, e in sel:  # L9：训不了的选拔经验同样降级记忆库，不得蒸发
            dolphin.memory.add(e.data, s, dolphin.cycle, "residue")
        buf.clear()
        dolphin.cycle += 1
        report["note"] = "样本过短"
        return report

    # 训练：律 L5——蒸馏锚定真实数据。硬标签=真实字节，软标签=醒脑对同批真实字节的输出
    sleeping.model.train()
    awake.model.eval()
    opt = sleeping.opt
    ce_hist, kd_hist = [], []
    N = bt.numel()
    for _ in range(steps):
        s0 = rng.randrange(0, N - blk - 1)
        x = bt[s0:s0 + blk].unsqueeze(0)
        y = bt[s0 + 1:s0 + blk + 1].unsqueeze(0)
        with torch.no_grad():
            t_logits, _ = awake.model(x)
        s_logits, _ = sleeping.model(x)
        ce = F.cross_entropy(s_logits.reshape(-1, dolphin.cfg.vocab), y.reshape(-1))
        p_t = F.softmax(t_logits / kd_T, dim=-1)
        kd = (kd_T ** 2) * F.kl_div(
            F.log_softmax(s_logits / kd_T, dim=-1), p_t, reduction="batchmean")
        loss = (1 - kd_alpha) * ce + kd_alpha * kd
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sleeping.model.parameters(), 1.0)
        opt.step()
        ce_hist.append(ce.item())
        kd_hist.append(kd.item())

    # 体检（律 L8）：全卷逐块评分 + margin 判决带。判决唯一实现在 Dolphin.gate
    # （→ probe.gate_decision），feed.trainer_cycle 调用同一方法——两份训练路径
    # 的门控零复制、无从漂移（M4 评审警告消解）
    passed, gate = dolphin.gate(awake, sleeping)
    report.update(gate)
    report["passed"] = passed
    report["ce_last"] = round(ce_hist[-1], 4)
    report["kd_last"] = round(kd_hist[-1], 4)

    if passed:
        dolphin.swap()  # 睡脑上岗，旧醒脑转睡
        report["swapped"] = True
        # 律 L10 快通道：新醒脑最自信的片段作为蒸馏笔记入记忆库（预支，立即可引用）
        # 必须显式给 key：并列 NLL 会让元组比较去比 Experience，而它没有 __lt__（律：不可比语义）
        # 次级键用稳定的 Experience.id，保证并列时顺序确定、可复现（禁止随机）
        ranked = sorted(
            ((dolphin.awake().model.mean_nll(e.data, dev), e) for _, e in sel),
            key=lambda t: (t[0], t[1].id),
        )
        for nll, e in ranked[: dolphin.note_k]:
            dolphin.memory.add(e.data, nll, dolphin.cycle, "note")
        # 值自成：体检连续通过 → 传输预算放宽
        dolphin.budget = min(0.60, dolphin.budget * 1.05)
    else:
        report["swapped"] = False
        # 作废回滚：睡脑重置为与醒脑同源，从好状态再睡
        sleeping.model.load_state_dict(awake.model.state_dict())
        dolphin.budget = max(0.25, dolphin.budget * 0.90)  # 值自成：收紧预算

    # 成年灵敏度（Plasticity：睡眠学习率的唯一控制器）——惊讶度门控带通 + 体检否决退火
    center, width = buf.band()
    mean_surp = sum(e.surprise for _, e in sel) / len(sel)
    cur_lr = sleeping.opt.param_groups[0]["lr"]
    new_lr = dolphin.plasticity.next_lr(cur_lr, mean_surp, center, width, passed)
    for h in dolphin.h:
        h.opt.param_groups[0]["lr"] = new_lr
    report["lr"] = new_lr

    buf.clear()
    dolphin.cycle += 1
    if verbose:
        print(f"[睡眠 {report['cycle']}] 选拔 {report['selected']} / 滞留 {report['residue']}"
              f"  体检 {report.get('probe_old')} → {report.get('probe_new')}"
              f"  lr={new_lr:.2e}"
              f"  {'✓ 换班' if report.get('swapped') else '✗ 作废回滚'}")
    return report
