# -*- coding: utf-8 -*-
"""层数自动动态增长（深度轴）测试：surgery.grow_layer 精确函数保持、
形态持久化往返、life 控制器触发/执行/配置、存档-加载兼容。

对应实现：
- dolphin/surgery.py  grow_layer（深度轴，恒等映射插入新层）
- dolphin/life.py    _plan_grow_depth / _execute_grow_depth / configure_depth_grow
- feed.py            --depth-grow / --depth-grow-max-layers / --depth-grow-min-gap

红线自守：不触碰生产资产（probes/probe.txt、dolphin/fed_state.pt、记忆冷层）；
所有写操作落在 tempfile 临时目录。
"""
import os
import shutil
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.dolphin import Dolphin
from dolphin.life import (DEPTH_GROW_ENABLED, STATE_GROWN, STATE_GROW_PLAN)
from dolphin.model import Config, ByteTransformer
from dolphin.surgery import (apply_morphology, born_again_student, grow_layer,
                             morphology_of, shrink_config)

PASS = []


def check(name, cond, detail=""):
    PASS.append(cond)
    print(f"  {'✓' if cond else '✗'} {name}  {detail}")


def _tmpdir(prefix="dolphin_depth_"):
    return tempfile.mkdtemp(prefix=prefix)


def _make_dolphin(cfg=None, probe_path=None):
    cfg = cfg or Config(d_model=32, n_layers=2, n_heads=2, block_size=64)
    d = Dolphin(cfg=cfg, device="cpu", probe_path=probe_path)
    d.param_adaptive_enabled = False
    return d


# ==================== 单元测试：grow_layer ====================

def t_grow_layer_identity():
    """深度轴精确函数保持：插入新层后模型输出逐位不变（torch.equal）。"""
    cfg = Config(d_model=32, n_layers=2, n_heads=2, block_size=64)
    m = ByteTransformer(cfg)
    x = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        l0, _ = m(x)
    rec = grow_layer(m, at=1, seed=0)
    with torch.no_grad():
        l1, _ = m(x)
    check("grow_layer 插入后层数+1", len(m.blocks) == 3 and m.cfg.n_layers == 3,
          f"blocks={len(m.blocks)}")
    check("grow_layer 输出逐位不变", torch.equal(l0, l1),
          f"max|Δ|={(l0 - l1).abs().max().item():.2e}")
    check("grow_layer 返回记录", rec["axis"] == "depth" and rec["at"] == 1
          and rec["delta"] == 1, f"{rec}")


def t_grow_layer_positions():
    """不同插入位置（末尾/开头/中间）行为正确。"""
    cfg = Config(d_model=16, n_layers=2, n_heads=2, block_size=32)
    for at in (None, 0, 1, 2):
        m = ByteTransformer(cfg)
        before = len(m.blocks)
        rec = grow_layer(m, at=at, seed=1)
        expect = before if at is None else at
        check(f"grow_layer at={at} 插入位置正确",
              len(m.blocks) == before + 1 and rec["at"] == expect,
              f"at={rec['at']}")


def t_grow_layer_invalid_at():
    """非法插入位置抛 ValueError。"""
    m = ByteTransformer(Config(d_model=16, n_layers=1, n_heads=2, block_size=32))
    try:
        grow_layer(m, at=5, seed=0)
        check("grow_layer 非法位置抛错", False, "未抛错")
    except ValueError:
        check("grow_layer 非法位置抛错", True)


def t_grow_layer_trains():
    """新层可训练：一步优化后 proj 权重离开零点、loss 下降。"""
    cfg = Config(d_model=32, n_layers=1, n_heads=2, block_size=32)
    m = ByteTransformer(cfg)
    grow_layer(m, at=0, seed=0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (1, 32))
    y = torch.randint(0, 256, (1, 32))
    _, loss0 = m(x, y)
    new_proj = m.blocks[0].attn.proj.weight
    check("新层 proj 初始全零", new_proj.abs().sum().item() == 0.0)
    loss0.backward()
    opt.step()
    opt.zero_grad()
    _, loss1 = m(x, y)
    check("新层 proj 训练后离开零点", new_proj.abs().sum().item() > 0.0,
          f"sum|w|={new_proj.abs().sum().item():.4f}")
    check("训练一步 loss 下降", loss1.item() < loss0.item(),
          f"{loss0.item():.4f} -> {loss1.item():.4f}")


# ==================== 单元测试：形态持久化 ====================

def t_morphology_roundtrip():
    """grow_layer 后形态记录可重建同构模型，state_dict 可加载且输出一致。"""
    cfg = Config(d_model=32, n_layers=2, n_heads=2, block_size=64)
    m = ByteTransformer(cfg)
    rec = grow_layer(m, at=1, seed=0)
    morph = morphology_of(m)
    morph["layers"] = [rec["at"]]
    m2, real_cfg = apply_morphology(cfg, morph, "cpu")
    check("形态重建层数正确", real_cfg.n_layers == 3 and len(m2.blocks) == 3,
          f"n_layers={real_cfg.n_layers}")
    check("state_dict 键一致",
          set(m.state_dict().keys()) == set(m2.state_dict().keys()))
    m2.load_state_dict(m.state_dict())
    x = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        l1, _ = m(x)
        l2, _ = m2(x)
    check("形态重建后输出逐位一致", torch.equal(l1, l2))


def t_apply_morphology_layers_empty():
    """无 layers 记录（旧档）时重建层数=出生层数（兼容）。"""
    cfg = Config(d_model=16, n_layers=2, n_heads=2, block_size=32)
    m, real_cfg = apply_morphology(
        cfg, {"d_model_k": 1, "mlp": {}, "attn_v": {}}, "cpu")
    check("旧档无 layers 兼容", real_cfg.n_layers == 2 and len(m.blocks) == 2)


# ==================== 集成测试：life 控制器配置 ====================

def t_configure_depth_grow():
    """configure_depth_grow 参数调节接口。"""
    d = _make_dolphin(probe_path=os.path.join(_tmpdir(), "probe.txt"))
    life = d.life_ctl
    check("默认深度生长开关开启", life.depth_grow_enabled == DEPTH_GROW_ENABLED)
    cfg = life.configure_depth_grow(enabled=False, max_layers=5, min_gap=10)
    check("配置生效", cfg == {"depth_grow_enabled": False,
                              "depth_grow_max_layers": 5,
                              "depth_grow_min_gap": 10})
    check("配置后属性更新", (not life.depth_grow_enabled)
          and life.depth_grow_max_layers == 5 and life.depth_grow_min_gap == 10)
    try:
        life.configure_depth_grow(max_layers=0)
        check("非法 max_layers 抛错", False)
    except ValueError:
        check("非法 max_layers 抛错", True)


# ==================== 集成测试：触发与执行 ====================

def _stall_dolphin():
    """构造学习停滞 + 宽度轴全部冷却的 Dolphin。"""
    d = _make_dolphin(probe_path=os.path.join(_tmpdir(), "probe.txt"))
    d.cycle = 100
    d._last_sleep_wall = time.time()   # 睡眠债低（时间债务口径）
    d.last_margins = [0.01] * 10       # 均值 0.01 < STALL_MARGIN=0.02 → 停滞
    life = d.life_ctl
    # 宽度轴全部标记为刚动作过 → 冷却中（限频③）
    for key in ("b0.mlp_hidden", "b0.attn_out", "b1.mlp_hidden", "b1.attn_out"):
        life.site_last_action[key] = [d.cycle, "grow"]
    # 守卫④：让预算/供给比通过前置检查（budget_units ≥ 8）
    life.fresh_since_surgery = 409600.0
    life.grown_since_surgery = 0
    life.last_stall_grow_cycle = None
    life.last_depth_grow_cycle = None
    life.cooldown = 0
    life.consec_rollback = 0
    life.lr_window = 0
    return d


def t_plan_grow_depth_trigger():
    """学习停滞且宽度轴不可用 → 回退排程深度生长。"""
    d = _stall_dolphin()
    report = {}
    action = d.life_ctl._plan_stall_grow(
        d, report, d.life_ctl.vitals_for(d, d.sleeping().name), b"")
    check("宽度轴不可用回退深度生长", action.get("kind") == "grow_depth",
          f"kind={action.get('kind')}")
    check("排程 plan 为 grow_depth",
          d.life_ctl.plan == {"kind": "grow_depth", "at": 1,
                              "seed": 100, "source": "depth_grow"},
          f"plan={d.life_ctl.plan}")
    check("状态为 GROW_PLAN", d.life_ctl.state == STATE_GROW_PLAN)
    check("报告含 m4_depth_grow", "m4_depth_grow" in report)


def t_depth_grow_execute():
    """执行深度生长：层数+1、形态记录 layers、报告字段。"""
    d = _stall_dolphin()
    life = d.life_ctl
    report = {}
    life.plan = {"kind": "grow_depth", "at": 1, "seed": d.cycle,
                 "source": "depth_grow"}
    life.state = STATE_GROW_PLAN
    before_layers = len(d.sleeping().model.blocks)
    life.pre_train(d, report, b"")
    check("执行后层数+1", len(d.sleeping().model.blocks) == before_layers + 1,
          f"{before_layers} -> {len(d.sleeping().model.blocks)}")
    check("状态为 GROWN", life.state == STATE_GROWN)
    morph = life.morph_for(d, d.sleeping().name)
    check("形态记录 layers 追加", morph.get("layers") == [1],
          f"layers={morph.get('layers')}")
    check("报告含 m4_depth_grow", "m4_depth_grow" in report
          and report["m4_depth_grow"]["at"] == 1)
    check("报告含 m4_surgery axis=depth",
          report.get("m4_surgery", {}).get("axis") == "depth")


def t_depth_grow_disabled():
    """关闭深度生长后返回 hold。"""
    d = _stall_dolphin()
    d.life_ctl.depth_grow_enabled = False
    action = d.life_ctl._plan_grow_depth(
        d, {}, d.life_ctl.vitals_for(d, d.sleeping().name), b"")
    check("关闭后 hold", action.get("kind") == "hold"
          and action.get("class") == "深度生长关闭", f"{action}")


def t_depth_grow_max_layers():
    """层数达上限后返回 hold。"""
    d = _stall_dolphin()
    d.life_ctl.depth_grow_max_layers = len(d.sleeping().model.blocks)
    action = d.life_ctl._plan_grow_depth(
        d, {}, d.life_ctl.vitals_for(d, d.sleeping().name), b"")
    check("达上限后 hold", action.get("kind") == "hold"
          and action.get("class") == "层数已达上限", f"{action}")


def t_depth_grow_min_gap():
    """最小间隔未满返回 hold。"""
    d = _stall_dolphin()
    d.life_ctl.last_depth_grow_cycle = d.cycle   # 刚深度生长过
    action = d.life_ctl._plan_grow_depth(
        d, {}, d.life_ctl.vitals_for(d, d.sleeping().name), b"")
    check("间隔未满 hold", action.get("kind") == "hold"
          and "间隔" in action.get("reason", ""), f"{action}")


# ==================== 集成测试：存档-加载 ====================

def t_save_load_depth_grown():
    """深度生长后的模型 save → load 层数保持、输出一致。"""
    tmp = _tmpdir()
    probe = os.path.join(tmp, "probe.txt")
    with open(probe, "wb") as f:
        f.write(b"0123456789abcdef" * 8)
    d = _make_dolphin(probe_path=probe)
    life = d.life_ctl
    life.plan = {"kind": "grow_depth", "at": 1, "seed": d.cycle,
                 "source": "depth_grow"}
    life.state = STATE_GROW_PLAN
    d._last_sleep_wall = time.time()
    life.pre_train(d, {}, b"")
    grown_layers = len(d.sleeping().model.blocks)
    ckpt = os.path.join(tmp, "fed_state.pt")
    d.save(ckpt)
    d2 = _make_dolphin(probe_path=probe)
    d2.load(ckpt)
    check("存档-加载层数保持", len(d2.sleeping().model.blocks) == grown_layers,
          f"{grown_layers}")
    x = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        l1, _ = d.awake().model(x)
        l2, _ = d2.awake().model(x)
    check("存档-加载输出一致", torch.equal(l1, l2))
    shutil.rmtree(tmp, ignore_errors=True)


def t_concentrate_shrink():
    """浓缩精华（2026-10-17 用户定案）：缩减时缩浅层增深度——宽度缩 target、
    层数增 1/target，总参数仍下降，但模型更窄更深。"""
    cfg = Config(d_model=64, n_layers=4, n_heads=4, block_size=64)
    s_cfg, info = shrink_config(cfg, None, target=0.5, concentrate=True)
    check("浓缩精华：宽度缩 target", s_cfg.d_model == 32,
          f"d_model={s_cfg.d_model}")
    check("浓缩精华：深度增 1/target", s_cfg.n_layers == 8,
          f"n_layers={s_cfg.n_layers}")
    check("浓缩精华：不丢层", info["dropped_layers"] == []
          and info["kept_layers"] == [0, 1, 2, 3], f"{info['dropped_layers']}")
    check("浓缩精华：标志记录", info["concentrate"] is True)
    n_orig = sum(p.numel() for p in ByteTransformer(cfg).parameters())
    n_stud = sum(p.numel() for p in ByteTransformer(s_cfg).parameters())
    check("浓缩精华：总参数下降", n_stud < n_orig,
          f"{n_stud}/{n_orig}={n_stud / n_orig:.3f}")
    # born_again_student 透传 concentrate
    stud, _, _ = born_again_student(cfg, None, 0.5, "cpu", seed=1, concentrate=True)
    check("浓缩精华：born_again 学生透传", stud.cfg.n_layers == 8
          and stud.cfg.d_model == 32, f"{stud.cfg.d_model}×{stud.cfg.n_layers}")
    # 默认 concentrate=False 保持既有行为（不回归）
    s2, info2 = shrink_config(cfg, None, target=0.5)
    check("默认不浓缩：层数按 target 缩", s2.n_layers == 2
          and info2["concentrate"] is False,
          f"d_model={s2.d_model} n_layers={s2.n_layers}")


def main():
    print("== 深度轴单元测试 ==")
    t_grow_layer_identity()
    t_grow_layer_positions()
    t_grow_layer_invalid_at()
    t_grow_layer_trains()
    print("== 形态持久化 ==")
    t_morphology_roundtrip()
    t_apply_morphology_layers_empty()
    print("== 浓缩精华（缩浅层增深度） ==")
    t_concentrate_shrink()
    print("== life 控制器配置与触发 ==")
    t_configure_depth_grow()
    t_plan_grow_depth_trigger()
    t_depth_grow_execute()
    t_depth_grow_disabled()
    t_depth_grow_max_layers()
    t_depth_grow_min_gap()
    print("== 存档-加载 ==")
    t_save_load_depth_grown()
    n_fail = sum(1 for c in PASS if not c)
    print(f"\n通过 {len(PASS) - n_fail}/{len(PASS)}，失败 {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())