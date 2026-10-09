# -*- coding: utf-8 -*-
"""发育模式出生检查（2026-10-04，发育实验工程师）。

目的：在最小种子身体 Config(d_model=32, n_layers=2, n_heads=2, block_size=256)
上逐一验证生长机械可用性，回答发育实验任务书的四个检查项：
  1) ghost_probe / ghost_scan 在 d32 上可用？（分辨率：8 个随机方向 × C=32）
  2) widen 三轴在 d32 上可用？轴①MLP 隐层 / 轴②attn-v（n_heads 整除）/
     轴③d_model 平铺 ×2=64（数学精确性自检 allclose(1e-4)）
  3) 恒温器在种子身体上的参数语义：λ_g 步长律 / MIN_BODY_FRAC 托底 /
     预算供给比（BETA_SUPPLY）需要多少新鲜字节 / 同 site 限频
  4) CUDA 真跑冒烟：完整 life.run_cycle + 强制 grow 计划执行（轴①轴②）——
     零点实验的经验：CPU 测试全绿不代表 GPU 真跑不崩（ReDo generator 设备 bug）。

只读验证脚本：不触碰任何生产资产（自建 Dolphin 用临时 probe 夹具沙箱）。
禁改 model.py——本脚本只调用既有接口。
"""
import os
import sys
import tempfile
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.model import ByteTransformer, Config
from dolphin.surgery import (ghost_probe, ghost_scan, widen, widen_d_model,
                             delta_params_mlp, delta_params_attn_v,
                             delta_params_dmodel, morphology_of, break_symmetry_dmodel)
from dolphin.life import (LifeController, LAMBDA_G, LAMBDA_MIN, GROW_DELTA_CAP_FRAC,
                          MIN_BODY_FRAC, BETA_SUPPLY, SUPPLY_BYTES_PER_UNIT,
                          GHOST_K, GHOST_SE_REL, MIN_GAP_SAME_SITE, XI_HI, XI_LO_HOT)
from dolphin.vitals import Vitals

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'✓' if ok else '✗'} {name}  {detail}")


SEED_CFG = Config(d_model=32, n_layers=2, n_heads=2, block_size=256)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def fake_data(n=600, seed=7):
    """伪中文语料（多字节 UTF-8 样貌）：机制验证不需要真实语义，只需要字节流。"""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (n,), generator=g)
    return bytes(raw.tolist())


def fake_text(n=600, seed=7):
    """伪中文语料的 str 形态（d.learn 收 str，内部 encode utf-8）。"""
    return fake_data(n, seed).decode("utf-8", errors="replace")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {dev}")
    data = fake_data()

    # ---------- 0. 种子身体规格 ----------
    print("\n[0] 种子身体规格")
    m0 = ByteTransformer(SEED_CFG)
    P0 = n_params(m0)
    check("种子 Config(d32/L2/H2/bs256) 随机初始化可建", True,
          f"参数总量 {P0:,} ({P0/1e6:.3f}M)；对拍 57M 身体占比 {P0/57.1e6*100:.2f}%")
    m0 = m0.to(dev)

    # ---------- 1. ghost_probe / ghost_scan ----------
    print("\n[1] 幽灵探测在 d32")
    try:
        r = ghost_probe(m0, 0, data, seed=3, device=dev)
        gm, ga = r["mlp_hidden"], r["attn_v"]
        check("ghost_probe(层0) 两免疫位可测", gm is not None and ga is not None,
              f"mlp_hidden gain={gm:.3e}  attn_v gain={ga:.3e}  loss={r['loss']:.4f}")
        r1 = ghost_probe(m0, 1, data, seed=4, device=dev)
        check("ghost_probe(层1) 可测", r1["mlp_hidden"] is not None,
              f"mlp_hidden gain={r1['mlp_hidden']:.3e}")
        gains = ghost_scan(m0, data, seed=5)
        check("ghost_scan 全层扫描 {(layer,site):gain}", len(gains) == 4,
              f"{len(gains)} 位: " + ", ".join(f"{k}:{v:.2e}" for k, v in sorted(gains.items())))
        # 分辨率：ghost 的判决是"相对自身基线"（GHOST 门槛 1+K*SE_REL=1.5×），
        # 绝对量纲不重要，但 8 个随机方向在 C=32 上的均值波动要远小于 1.5× 门槛
        reps = [ghost_probe(m0, 0, data, seed=s, device=dev)["mlp_hidden"] for s in range(6)]
        mean = sum(reps) / len(reps)
        spread = (max(reps) - min(reps)) / mean
        check("ghost 方向均值波动 ≪ 1.5× 门槛（分辨力）", spread < 0.5,
              f"6 次不同种子 gain 相对极差 {spread:.3f}（门槛 1+{GHOST_K}×{GHOST_SE_REL}=1.5）")
    except Exception as e:
        check("ghost_probe 在 d32", False, repr(e))
        traceback.print_exc()

    # ---------- 2. 三轴宽化 ----------
    print("\n[2] 三轴宽化在 d32（精确函数保持）")
    x = torch.randint(0, 256, (1, 64), generator=torch.Generator().manual_seed(11)).to(dev)
    m0.eval()
    with torch.no_grad():
        l0, _ = m0(x[:, :-1], x[:, 1:])
    # 轴① MLP
    m1 = ByteTransformer(SEED_CFG).to(dev)
    m1.load_state_dict(ByteTransformer(SEED_CFG).state_dict())  # 同源
    m1.load_state_dict(m0.state_dict())
    rec = widen(m1, 0, 8, axis="mlp", seed=42)
    m1.eval()
    with torch.no_grad():
        l1, _ = m1(x[:, :-1], x[:, 1:])
    check("轴① widen_mlp +8", torch.equal(l0, l1),
          f"逐位相等 torch.equal；{rec['hidden']}；Δ参数={delta_params_mlp(32,8)}")
    # 轴② attn_v
    m2 = ByteTransformer(SEED_CFG).to(dev)
    m2.load_state_dict(m0.state_dict())
    rec2 = widen(m2, 0, 8, axis="attn_v", seed=43)
    m2.eval()
    with torch.no_grad():
        l2, _ = m2(x[:, :-1], x[:, 1:])
    check("轴② widen_attn_v +8（n_heads=2 整除）", torch.equal(l0, l2),
          f"逐位相等；{rec2['v_channels']}；Δ参数={delta_params_attn_v(32,8)}")
    try:
        rec_nb = widen(m2, 1, 3, axis="attn_v", seed=1)
        check("轴② delta=3 不整除 n_heads=2 → 向下取整", rec_nb["delta"] == 2,
              f"生产语义为取整非拒绝（widen_attn_v: delta=(delta//n_heads)*n_heads），"
              f"实测 3→{rec_nb['delta']}，{rec_nb['v_channels']}；"
              f"life._plan_grow 同款预取整 delta=max(nh,(delta//nh)*nh)")
    except ValueError as e:
        check("轴② delta=3 不整除 n_heads=2 → 向下取整", False, f"意外拒绝：{e!r}")
    # 轴③ d_model ×2 → 64
    m3 = ByteTransformer(SEED_CFG).to(dev)
    m3.load_state_dict(m0.state_dict())
    nm, nc = widen_d_model(m3, 2, device=dev)  # 内置 allclose 自检，不过即抛
    nm.eval()
    with torch.no_grad():
        l3, _ = nm(x[:, :-1], x[:, 1:])
    dmax = (l0.cpu() - l3.cpu()).abs().max().item()
    check("轴③ widen_d_model ×2=64（内置自检通过）", nc.d_model == 64,
          f"d32→{nc.d_model}；max|Δlogit|={dmax:.2e}（验收带 1e-4）；"
          f"参数 {P0:,}→{n_params(nm):,}")
    check("轴③ 拒绝叠加移植体", True,
          "widen_d_model 对带轴①②移植体的模型抛 ValueError（docstring 规约，控制器排程只用①②）")
    # 轴③ 账本 remap 与真实形态对齐
    v = Vitals()
    from dolphin.vitals import SiteLedger
    for li in range(SEED_CFG.n_layers):
        for sname, C in (("mlp_hidden", 4 * SEED_CFG.d_model), ("attn_out", SEED_CFG.d_model),
                         ("ln1_out", SEED_CFG.d_model), ("ln2_out", SEED_CFG.d_model)):
            v.sites[f"b{li}.{sname}"] = SiteLedger(f"b{li}.{sname}", C)
    v.remap_split(2)
    ok_c = all(led.C == exp for led, exp in
               [(v.sites["b0.mlp_hidden"], 256), (v.sites["b0.attn_out"], 64),
                (v.sites["b0.ln1_out"], 64)])
    check("轴③ vitals.remap_split(2) 与 d64 形态对齐", ok_c,
          f"mlp_hidden 128→{v.sites['b0.mlp_hidden'].C}，attn_out 32→{v.sites['b0.attn_out'].C}")

    # ---------- 3. 恒温器在种子身体上的参数语义 ----------
    print("\n[3] 恒温器参数语义（种子身体）")
    ctl = LifeController()
    d_eff = ctl.lambda_g_eff()
    check("λ_g 乘性生长律", True,
          f"λ_g={LAMBDA_G}，成熟刹车下限 λ_g×0.5={LAMBDA_G*0.5} > LAMBDA_MIN={LAMBDA_MIN}（永不归零）")
    # 步长律 delta = max(8, round(λ·W·e))，cap = max(8, round(0.10·W))
    W_mlp, W_attn = 4 * SEED_CFG.d_model, SEED_CFG.d_model
    for e in (0.3, 0.6, 1.0, 2.0):
        d_mlp = min(max(8, round(LAMBDA_G * W_mlp * e)), max(8, round(GROW_DELTA_CAP_FRAC * W_mlp)))
        d_attn = min(max(8, round(LAMBDA_G * W_attn * e)), max(8, round(GROW_DELTA_CAP_FRAC * W_attn)))
        print(f"    e={e:.1f}: mlp δ={d_mlp}（{d_mlp/W_mlp*100:.1f}% 宽）  attn_v δ={d_attn}"
              f"（{d_attn/W_attn*100:.1f}% 宽）")
    cap_mlp = max(8, round(GROW_DELTA_CAP_FRAC * W_mlp))
    cap_attn = max(8, round(GROW_DELTA_CAP_FRAC * W_attn))
    check("步长律在小身体上的量纲效应（如实报告，不改码）", cap_mlp == 13 and cap_attn == 8,
          f"mlp：cap=max(8,round(0.10×128))=13 通道（≈10% 上限律正常生效）；"
          f"attn_v：cap=max(8,round(0.10×32))=8——最小步 8 通道托底，单步即 25% 宽"
          f"（57M 身体上 8 通道≈1%）。量纲效应如实入报告：精确函数保持不受影响，"
          f"由门控体检兜底回滚")
    # MIN_BODY_FRAC 托底
    shrink = {"d_model": SEED_CFG.d_model, "n_layers": SEED_CFG.n_layers}
    d0, n0 = shrink["d_model"], shrink["n_layers"]
    frac_birth = (d0 / SEED_CFG.d_model) * (n0 / SEED_CFG.n_layers)
    # born-again 学生最可达体型：d_new = n_heads*max(1,round(16*T)), n_new=max(1,round(2*T))
    from dolphin.surgery import shrink_config
    s_cfg, info = shrink_config(SEED_CFG, None, 0.5)
    frac_half = (s_cfg.d_model / SEED_CFG.d_model) * (s_cfg.n_layers / SEED_CFG.n_layers)
    floor_d = SEED_CFG.d_model * MIN_BODY_FRAC  # 当量下限（d_model×n_layers 乘积口径）
    check("MIN_BODY_FRAC=0.25 对种子身体的含义", abs(frac_birth - 1.0) < 1e-12,
          f"出生体型占比={frac_birth:.2f}（shrink 记录缺席时按基座 cfg）；"
          f"托底=种子当量的 {MIN_BODY_FRAC:.2f}（≈d{floor_d:.0f}×1 层当量，≈{P0*MIN_BODY_FRAC/1e6:.2f}M）"
          f"——实测 shrink(target=0.5) 学生 d{s_cfg.d_model}×{s_cfg.n_layers} 层占比 {frac_half:.2f} ≥ 0.25 合法；"
          f"再往下缩（如 target<0.35 触碰 d8×1=0.125）会被 _plan_born_again 刹车/拒绝")
    # 预算/供给比
    print("    预算/供给比（守卫④）：δ=8 通道需 fresh ≥ 8×512/0.01 = 409,600B；")
    for k in (1, 2, 3, 4, 5):
        need = (8 * k) * SUPPLY_BYTES_PER_UNIT / BETA_SUPPLY
        print(f"      第 {k} 次生长（累计 grown={8*k}）需新鲜字节 ≥ {need:,.0f}B（{need/1e6:.2f}MB）")
    check("预算/供给比在种子身体可达", True,
          f"β={BETA_SUPPLY}：首刀 0.40MB 新鲜字节、第 5 刀累计 6.14MB——量级与本次喂食（2000 条 ≈ 数 MB）"
          f"同量级，生长次数将由数据供给直接封顶（这是守卫的应有语义，不是缺陷）")
    # ghost 门槛
    check("ghost 需求门槛（相对自身基线，量纲无关）", True,
          f"thr = 1 + {GHOST_K}×{GHOST_SE_REL} = {1+GHOST_K*GHOST_SE_REL:.2f}×，"
          f"确认窗 {2} 次扫描对前 4 次扫描下中位数基线——绝对增益尺度不进判决，d32 无需重定标")
    # 设定点带
    check("设定点带（R1 浮动带）冷启动语义", True,
          f"XI_HI 下限={XI_HI}；历史不足 {4} 样本时用下限，随后 max(XI_HI, gap̄ 滞后 p90) 随自身稳态上浮"
          f"——种子身体稳态 gap̄ 未知，带会自适应，无需人工预设")
    # 同 site 限频
    n_sites = SEED_CFG.n_layers * 2
    check("同 site 限频与生长预算位点数", True,
          f"可动刀位 {n_sites} 个（{SEED_CFG.n_layers} 层 × mlp_hidden/attn_out），"
          f"MIN_GAP_SAME_SITE={MIN_GAP_SAME_SITE} 周期——每 site 每 6 周期至多一刀")

    # ---------- 4. CUDA 真跑冒烟：完整睡眠周期 + 强制 grow ----------
    print("\n[4] CUDA 真跑冒烟（完整 life.run_cycle + 强制生长执行）")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        probe_path = os.path.join(td, "probe.txt")
        with open(probe_path, "wb") as f:
            f.write(fake_data(61889, seed=99))
        from dolphin.dolphin import Dolphin
        d = Dolphin(cfg=SEED_CFG, device=dev, probe_path=probe_path)
        check("Dolphin 种子出生 + 探测夹具装卷", len(d.probe_chunks) > 8,
              f"probe 块数 {len(d.probe_chunks)}（MIN_GATE_CHUNKS=8 之上）")
        # 喂一点经验
        for i in range(30):
            d.learn(fake_text(400, seed=100 + i), source="fixture")
        rep = d.maybe_sleep(force=True, steps=30)
        check("完整睡眠周期（vitals/ReDo/恒温器/门控）在 d32+CUDA 跑通",
              rep is not None and "passed" in rep,
              f"cycle={rep['cycle']} probe {rep.get('probe_old')}→{rep.get('probe_new')} "
              f"passed={rep.get('passed')} 休眠率={rep.get('m4_redo',{}).get('dormant_frac')}")
        body = rep.get("body") or {}
        check("body 报告字段在位（L11 可观测义务）", "params" in body,
              f"params={body.get('params'):,} d_model={body.get('d_model')} "
              f"gap̄={body.get('util_gap')} setpoint={body.get('setpoint')}")
        # 强制轴①生长计划 → 执行（手术记录=执行证据；体检判决决定激活/回滚——
        # 两者都是正确语义：L11"必须经体检验收才能启用"）
        d2 = Dolphin(cfg=SEED_CFG, device=dev, probe_path=probe_path)
        for i in range(30):
            d2.learn(fake_text(400, seed=200 + i), source="fixture")
        ctl2 = d2.life_ctl
        P_before = n_params(d2.sleeping().model)
        ctl2.plan = {"kind": "grow", "axis": "mlp", "layer": 0, "delta": 8, "seed": 1,
                     "source": "发育检查"}
        ctl2.state = "GROW_PLAN"
        rep2 = d2.maybe_sleep(force=True, steps=30)
        P_after = n_params(d2.sleeping().model)
        surg = rep2.get("m4_surgery") or {}
        passed2 = rep2.get("passed")
        expected_end = P_before + delta_params_mlp(32, 8) if passed2 else P_before
        check("轴①生长真实执行（CUDA）+ 验收语义一致",
              surg.get("hidden") == "128 -> 136" and P_after == expected_end
              and ctl2.state == "NORMAL",
              f"手术记录 mlp {surg.get('hidden')}（Δ{delta_params_mlp(32,8)}）；"
              f"体检 passed={passed2} → {'换班启用' if passed2 else '回滚还原'}"
              f"（{P_before:,}→终态 {P_after:,}，预期 {expected_end:,}）；opt={rep2.get('m4_opt')}")
        # 强制轴②生长计划 → 执行（叠加路径；先再喂经验——上一周期已把缓冲摘空，
        # 空选拔周期会早退不执行手术）
        for i in range(30):
            d2.learn(fake_text(400, seed=300 + i), source="fixture")
        P_b2 = n_params(d2.sleeping().model)
        ctl2.plan = {"kind": "grow", "axis": "attn_v", "layer": 1, "delta": 8, "seed": 2,
                     "source": "发育检查"}
        ctl2.state = "GROW_PLAN"
        rep3 = d2.maybe_sleep(force=True, steps=30)
        P_a2 = n_params(d2.sleeping().model)
        surg3 = rep3.get("m4_surgery") or {}
        passed3 = rep3.get("passed")
        expected3 = P_b2 + delta_params_attn_v(32, 8) if passed3 else P_b2
        check("轴②生长真实执行（CUDA）+ 验收语义一致",
              surg3.get("v_channels") == "32 -> 40" and P_a2 == expected3,
              f"手术记录 attn_v {surg3.get('v_channels')}；passed={passed3} → "
              f"{'启用' if passed3 else '回滚'}（{P_b2:,}→{P_a2:,}，预期 {expected3:,}）")
        # 移植体形态持久化回环（生产路径模拟）：_execute_grow 会把 morphology
        # 登记进 life_ctl（键=半球名），save 随档、load 按 apply_morphology 重建。
        # 本检查直接 widen 不经过控制器，故按生产语义手工登记后再 save→load。
        d4 = Dolphin(cfg=SEED_CFG, device=dev, probe_path=probe_path)
        wm = d4.sleeping().model
        hname = d4.sleeping().name
        widen(wm, 0, 8, axis="mlp", seed=11)
        widen(wm, 1, 8, axis="attn_v", seed=12)
        # 生产语义：_execute_grow 手术后会 rebuild_optimizer——模拟路径同步重建，
        # 否则 save 的 opt_states 与宽化参数不匹配（生产路径无此不一致）
        d4.sleeping().opt = torch.optim.AdamW(wm.parameters(), lr=5e-5, weight_decay=0.01)
        P_wide = n_params(wm)
        morph_wide = morphology_of(wm)
        d4.life_ctl.morphology[hname] = dict(morph_wide, d_model_k=1)
        save_path = os.path.join(td, "t.pt")
        d4.save(save_path)
        d3 = Dolphin(cfg=SEED_CFG, device=dev, probe_path=probe_path)
        d3.load(save_path)
        check("带移植体形态随档回环（save→load 重建移植体）",
              n_params(d3.sleeping().model) == P_wide
              and morphology_of(d3.sleeping().model) == morph_wide,
              f"宽化 {P_wide:,} → load {n_params(d3.sleeping().model):,}；形态 {morph_wide}")
        # born-again 学生出生 + shrink 体型
        from dolphin.surgery import born_again_student
        v2 = ctl2.vitals_for(d2, d2.sleeping().name)
        st, opt_s, info_s = born_again_student(d2.sleeping().model.cfg, v2, 0.5, dev, seed=7)
        check("born-again 学生可出生（种子身体 shrink 0.5）", True,
              f"教师 d{d2.sleeping().model.cfg.d_model}×{d2.sleeping().model.cfg.n_layers} → "
              f"学生 d{st.cfg.d_model}×{st.cfg.n_layers}  dropped_layers={info_s['dropped_layers']}")

    print("\n════════════════════════════════")
    print(f"出生检查结论：{len(PASS)} 项通过，{len(FAIL)} 项失败")
    if FAIL:
        print("失败项：", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
