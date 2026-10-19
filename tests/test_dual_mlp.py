"""DualMLP（发达神经元：树突-胞体双通道门控 FFN）回归测试（2026-10-19）。

对应中科院脑科学发现（2026-10-16 Science）：神经元内部精密分工——树突编码
规则/慢特征，胞体表征输入/快特征，两类信号在胞体内做非线性整合后输出。

覆盖：
- 结构：Block 使用 DualMLP；rule/body/out 三线性层维度正确。
- 前向公式：out(gelu(rule(x)) * tanh(body(x))) 与手工计算逐位一致。
- 旧存档迁移：_migrate_mlp_state_dict 把 fc/proj → rule/body/out，
  迁移后模型输出与旧 MLP 模型接近（数学等价，浮点容差）。
- grow_layer 精确函数保持：插入恒等层后 logits 逐位不变（torch.equal）。
- widen_mlp 精确函数保持：旁路零输出权重掩蔽，宽化后 logits 逐位不变。
- tile_state_dict 新旧键兼容：旧键（fc/proj）输入 → 输出新键（rule/body/out）。
- ghost_probe 双通道幽灵：在 DualMLP 上返回 mlp_hidden 增益。

红线自守：只构造内存模型与随机张量，不读写任何生产存档/数据。
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.model import Block, ByteTransformer, Config, DualMLP, MLP
from dolphin.surgery import (WidenedMLP, ghost_probe, grow_layer,
                            tile_state_dict, widen_mlp)

PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"  {'✓' if cond else '✗'} {name}  {detail}")


def _make_model(cfg, seed=0):
    torch.manual_seed(seed)
    return ByteTransformer(cfg)


def _make_legacy_model(cfg, seed=0):
    """构造使用旧 MLP（fc/proj）的模型，产出旧格式 state_dict。"""
    torch.manual_seed(seed)
    m = ByteTransformer(cfg)
    for blk in m.blocks:
        blk.mlp = MLP(cfg)
    return m


# ==================== 1. 结构与前向公式 ====================

def t_structure():
    cfg = Config(d_model=16, n_layers=2, n_heads=2, block_size=32)
    m = _make_model(cfg)
    for li, blk in enumerate(m.blocks):
        check("Block 使用 DualMLP", isinstance(blk.mlp, DualMLP),
              f"layer {li}: {type(blk.mlp).__name__}")
    d = cfg.d_model
    mlp = m.blocks[0].mlp
    check("rule 维度 d→4d", mlp.rule.in_features == d and mlp.rule.out_features == 4 * d,
          f"{mlp.rule.in_features}→{mlp.rule.out_features}")
    check("body 维度 d→4d", mlp.body.in_features == d and mlp.body.out_features == 4 * d,
          f"{mlp.body.in_features}→{mlp.body.out_features}")
    check("out 维度 4d→d", mlp.out.in_features == 4 * d and mlp.out.out_features == d,
          f"{mlp.out.in_features}→{mlp.out.out_features}")


def t_forward_formula():
    """手工展开 DualMLP 前向，与 PyTorch 模块输出逐位一致。"""
    torch.manual_seed(7)
    cfg = Config(d_model=8, n_layers=1, n_heads=2, block_size=32)
    m = _make_model(cfg)
    mlp = m.blocks[0].mlp
    x = torch.randn(2, 6, 8)
    y = mlp(x)
    rule = F.gelu(F.linear(x, mlp.rule.weight, mlp.rule.bias))
    body = torch.tanh(F.linear(x, mlp.body.weight, mlp.body.bias))
    y_manual = F.linear(rule * body, mlp.out.weight, mlp.out.bias)
    check("DualMLP 前向与手工公式一致", torch.equal(y, y_manual),
          f"max|Δ|={(y - y_manual).abs().max().item():.3e}")


# ==================== 2. 旧存档迁移 ====================

def t_migrate_keys():
    """旧键 fc/proj → 新键 rule/body/out；已是新键则原样返回。"""
    from dolphin.dolphin import _migrate_mlp_state_dict
    cfg = Config(d_model=8, n_layers=1, n_heads=2, block_size=16)
    legacy = _make_legacy_model(cfg, seed=3)
    old_sd = legacy.state_dict()
    check("旧模型含 fc/proj 键",
          any(".mlp.fc.weight" in k for k in old_sd)
          and any(".mlp.proj.weight" in k for k in old_sd))
    migrated = _migrate_mlp_state_dict(old_sd)
    check("迁移后旧键全部消失",
          not any(".mlp.fc." in k or ".mlp.proj." in k for k in migrated))
    check("迁移后新键齐全",
          all(any(f".mlp.{mod}.{part}" in k for k in migrated)
              for mod in ("rule", "body", "out") for part in ("weight", "bias")))
    # body：权重全零、偏置全一（tanh(1)≈0.7616 常数调制）
    bw = migrated["blocks.0.mlp.body.weight"]
    bb = migrated["blocks.0.mlp.body.bias"]
    check("body 权重全零", torch.count_nonzero(bw) == 0)
    check("body 偏置全一", torch.all(bb == 1.0))
    # rule ← fc 原样；out ← proj × (1/tanh(1))
    scale = 1.0 / math.tanh(1.0)
    rule_w = migrated["blocks.0.mlp.rule.weight"]
    out_w = migrated["blocks.0.mlp.out.weight"]
    check("rule 继承 fc", torch.equal(rule_w, old_sd["blocks.0.mlp.fc.weight"]))
    check("out 权重 = proj × (1/tanh(1))",
          torch.allclose(out_w, old_sd["blocks.0.mlp.proj.weight"] * scale,
                         atol=1e-6, rtol=1e-6))
    check("out 偏置 = proj 偏置",
          torch.equal(migrated["blocks.0.mlp.out.bias"],
                      old_sd["blocks.0.mlp.proj.bias"]))
    # 已是新键 → 内容不变（返回 dict 拷贝，键值集合完全一致）
    new_sd = _make_model(cfg, seed=4).state_dict()
    back = _migrate_mlp_state_dict(new_sd)
    check("已是新键则内容不变", back == new_sd and set(back) == set(new_sd))


def t_migrate_bypass_bodies():
    """旧 WidenedMLP 存档（仅 bypass_fcs 旁路）迁移后补齐 bypass_bodies。"""
    from dolphin.dolphin import _migrate_mlp_state_dict
    cfg = Config(d_model=8, n_layers=1, n_heads=2, block_size=16)
    legacy = _make_legacy_model(cfg, seed=12)
    # 手工给旧模型附加旧版旁路（仅 bypass_fcs + proj_new_ws，无 bypass_bodies）
    for blk in legacy.blocks:
        blk.mlp.bypass_fcs = torch.nn.ModuleList(
            [torch.nn.Linear(cfg.d_model, 4) for _ in range(2)])
        blk.mlp.proj_new_ws = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(cfg.d_model, 4)) for _ in range(2)])
    old_sd = legacy.state_dict()
    migrated = _migrate_mlp_state_dict(old_sd)
    bb_keys = [k for k in migrated if ".mlp.bypass_bodies." in k]
    check("迁移后补齐 bypass_bodies 权重/偏置",
          len(bb_keys) == 4
          and all(k.endswith(".weight") or k.endswith(".bias") for k in bb_keys))
    bw0 = migrated["blocks.0.mlp.bypass_bodies.0.weight"]
    bb0 = migrated["blocks.0.mlp.bypass_bodies.0.bias"]
    check("bypass_bodies 权重全零", torch.count_nonzero(bw0) == 0)
    check("bypass_bodies 偏置全一", torch.all(bb0 == 1.0))


def t_migrate_output_equivalence():
    """旧模型 vs 迁移后新模型：相同输入下输出接近（数学等价）。"""
    from dolphin.dolphin import _migrate_mlp_state_dict
    cfg = Config(d_model=8, n_layers=1, n_heads=2, block_size=16)
    legacy = _make_legacy_model(cfg, seed=11)
    old_sd = legacy.state_dict()
    new = _make_model(cfg, seed=99)
    new.load_state_dict(_migrate_mlp_state_dict(old_sd))
    legacy.eval(); new.eval()
    x = torch.randint(0, 256, (1, 16))
    xin, y = x[:, :-1], x[:, 1:]
    with torch.no_grad():
        l0, _ = legacy(xin, y)
        l1, _ = new(xin, y)
    d = (l0 - l1).abs().max().item()
    check("迁移后输出等价（allclose 1e-4）",
          torch.allclose(l0, l1, atol=1e-4, rtol=1e-4), f"max|Δlogit|={d:.3e}")


# ==================== 3. 精确函数保持 ====================

def t_grow_layer_bitwise():
    """grow_layer 插入恒等层：任意输入 logits 逐位不变。"""
    torch.manual_seed(0)
    cfg = Config(d_model=8, n_layers=1, n_heads=2, block_size=16)
    m = _make_model(cfg, seed=5)
    m.eval()
    x = torch.randint(0, 256, (1, 16))
    xin, y = x[:, :-1], x[:, 1:]
    with torch.no_grad():
        l0, _ = m(xin, y)
    grow_layer(m, at=0, seed=1)
    with torch.no_grad():
        l1, _ = m(xin, y)
    check("grow_layer 后模型层数 +1", len(m.blocks) == 2, f"n_layers={len(m.blocks)}")
    check("grow_layer 逐位不变（torch.equal）", torch.equal(l0, l1),
          f"max|Δ|={(l0 - l1).abs().max().item():.3e}")


def t_widen_mlp_bitwise():
    """widen_mlp 旁路零输出权重：宽化后 logits 逐位不变。"""
    torch.manual_seed(0)
    cfg = Config(d_model=8, n_layers=1, n_heads=2, block_size=16)
    m = _make_model(cfg, seed=6)
    m.eval()
    x = torch.randint(0, 256, (1, 16))
    xin, y = x[:, :-1], x[:, 1:]
    with torch.no_grad():
        l0, _ = m(xin, y)
    rec = widen_mlp(m, 0, delta=4, seed=2)
    check("widen_mlp 返回记录", rec["axis"] == "mlp" and rec["delta"] == 4,
          f"{rec['hidden']}")
    check("widen_mlp 后为 WidenedMLP 移植体",
          isinstance(m.blocks[0].mlp, WidenedMLP))
    with torch.no_grad():
        l1, _ = m(xin, y)
    check("widen_mlp 逐位不变（torch.equal）", torch.equal(l0, l1),
          f"max|Δ|={(l0 - l1).abs().max().item():.3e}")
    # 旁路结构：bypass_fcs / bypass_bodies / proj_new_ws 齐备
    mlp = m.blocks[0].mlp
    check("旁路三件套齐备",
          len(mlp.bypass_fcs) == 1 and len(mlp.bypass_bodies) == 1
          and len(mlp.proj_new_ws) == 1
          and mlp.bypass_fcs[0].out_features == 4
          and mlp.bypass_bodies[0].out_features == 4)
    check("旁路输出权重全零", torch.count_nonzero(mlp.proj_new_ws[0]) == 0)


# ==================== 4. tile_state_dict 新旧键兼容 ====================

def t_tile_legacy_keys():
    """旧格式（fc/proj）state_dict 平铺后输出新键（rule/body/out）。"""
    cfg = Config(d_model=8, n_layers=1, n_heads=2, block_size=16)
    legacy = _make_legacy_model(cfg, seed=8)
    old_sd = legacy.state_dict()
    tiled = tile_state_dict(old_sd, 2)
    p = "blocks.0.mlp."
    has = lambda suffix: any(k.endswith(suffix) for k in tiled)
    check("平铺输出 rule 键", has("mlp.rule.weight") and has("mlp.rule.bias"))
    check("平铺输出 body 键", has("mlp.body.weight") and has("mlp.body.bias"))
    check("平铺输出 out 键", has("mlp.out.weight") and has("mlp.out.bias"))
    check("平铺后不再含旧键",
          not any(".mlp.fc." in k or ".mlp.proj." in k for k in tiled))
    # d_model ×k：新 d_model=16，rule/body 隐层 = 4×16=64；out 投影回 16。
    check("rule 平铺形状 (4·d·k, d·k)",
          tiled["blocks.0.mlp.rule.weight"].shape == (64, 16))
    check("out 平铺形状 (d·k, 4·d·k)",
          tiled["blocks.0.mlp.out.weight"].shape == (16, 64))


def t_tile_new_keys():
    """新格式（rule/body/out）state_dict 平铺后键名与形状保持新体系。"""
    cfg = Config(d_model=8, n_layers=1, n_heads=2, block_size=16)
    m = _make_model(cfg, seed=9)
    sd = m.state_dict()
    tiled = tile_state_dict(sd, 2)
    p = "blocks.0.mlp."
    check("新键平铺 rule 形状",
          tiled[p + "rule.weight"].shape == (64, 16))
    check("新键平铺 body 形状",
          tiled[p + "body.weight"].shape == (64, 16))
    check("新键平铺 out 形状",
          tiled[p + "out.weight"].shape == (16, 64))


# ==================== 5. ghost_probe 双通道 ====================

def t_ghost_probe_dual():
    """ghost_probe 在 DualMLP 上挂 rule 输入 hook + 双通道幽灵，返回增益。"""
    torch.manual_seed(0)
    cfg = Config(d_model=8, n_layers=1, n_heads=2, block_size=16)
    m = _make_model(cfg, seed=10)
    data = ("幽灵探测数据：这段字节流需要足够长才能支撑反向传播的窗口切分，"
            "青雀从不畏惧风浪。").encode("utf-8")
    r = ghost_probe(m, 0, data, site="mlp_hidden", n_ghosts=4, seed=0)
    check("ghost_probe 返回 mlp_hidden 增益",
          r["mlp_hidden"] is not None and r["mlp_hidden"] > 0,
          f"gain={r['mlp_hidden']:.4e}")
    check("ghost_probe 返回 loss", r["loss"] is not None and torch.isfinite(torch.tensor(r["loss"])))
    check("ghost_probe 后梯度清零",
          all(p.grad is None or torch.count_nonzero(p.grad) == 0
              for p in m.parameters()))


if __name__ == "__main__":
    print("== DualMLP（发达神经元）回归测试 ==")
    t_structure()
    t_forward_formula()
    t_migrate_keys()
    t_migrate_bypass_bodies()
    t_migrate_output_equivalence()
    t_grow_layer_bitwise()
    t_widen_mlp_bitwise()
    t_tile_legacy_keys()
    t_tile_new_keys()
    t_ghost_probe_dual()
    n_ok = sum(PASS)
    print(f"\n== 结论：{n_ok}/{len(PASS)} 绿 ==")
    sys.exit(0 if n_ok == len(PASS) else 1)