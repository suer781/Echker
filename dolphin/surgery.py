"""结构手术（M4 自生长/自凋零的执行层）。

三条**精确函数保持**变宽轴（规格 B，按序优先）：
  ① MLP 隐层    ——旁路新单元（fc 新行 = 对称破缺噪声，输出权重零初始化掩蔽）
  ② attn-v 路径 ——旁路新 v 通道（qkv 旁路新行 + 零输出权重掩蔽；delta % n_heads == 0）
  ③ d_model 复制平铺 ——W⊗I_k 式状态字典变换（q/k/v 副本尺度 k^(-1/4)、残差流
     副本尺度 1/√k，由绑定头约束 k·t_r²=1 唯一确定；LN(平铺)=平铺(LN) 精确成立）

精确性的实现形态（前置浮点实验定标，2026-10-04）：
  逐位相等的敌人不是数学而是 GEMM 的 K 维变化——CPU/CUDA 内核按形状选择，
  即使追加纯零列也会改变老输出的舍入次序（实测 max|Δ|≈4e-7）。因此轴①②做成
  **移植体**：老 fc/qkv/proj 模块原封不动（GEMM 形状与算子次序逐位不变），
  新单元走独立旁路 GEMM，其输出权重零初始化——y + F.linear(·, 0) = y + 0.0
  在 IEEE754 下精确。实测 CPU 与 CUDA 上宽化前后 logits 逐位相等（torch.equal）。
  轴③的 K 维随副本数增长且各项非零，逐位不可达；数学上精确，浮点实测相对误差
  ~3e-6，用 allclose(atol=1e-4, rtol=1e-4) 验收，并在 widen_d_model 内置自检。

对称破缺（规格 B）：所有新副本/新单元必须带噪声，否则梯度逐位相同永不分化
（dropout=0 的本项目尤其如此）。轴①②的噪声藏在零输出权重后面——宽化瞬间仍
逐位保持；轴③的副本是活的（直接参与输出），破缺噪声由 break_symmetry_dmodel
在**训练恢复前**单独施加（那一刻起放弃逐位性——这正是噪声的目的）。

幽灵探测 ghost_probe（规格 B）：只挂 **LN 免疫位**（MLP 隐层单元、attn-v 通道）
——残差流挂幽灵会被 √(1+Δ/d) 污染测量（对质已证伪），残差流位**拒绝挂载**。
一阶口径：新单元（方向 u，输出列 w）的损失梯度增益 = ‖Σ_pos g·a_ghost‖₂，
g = 该位输出侧梯度；等价于"若插入此幽灵，其输出参数将收到的梯度范数"。

born-again 蒸馏（规格 B）：教师=现脑（醒脑，L3 冻结），学生=shrink_config 的
从头初始化；kd_alpha 提到 0.7–0.8（仅限手术验收周期，取 0.75【定标带中值】）；
训练数据与常规周期同一条真实回放管线（记忆库 dream+复活+缓冲，L5 锚定；变异
重放照 L6 律执行——"真实回放"指数据来源真实，非自生成内容）。

显存预算（规格 B）：变宽前核算新足迹（参数×16B：fp32 权重+梯度+Adam m,v，
架构设计 §6），超 1.9GB 推迟（WDDM 倒页 6 倍减速的教训）。

形态持久化：morphology（纯 list/dict）随档保存；load 时 apply_morphology 先
重建同构模型再 load_state_dict——宽化后的脑可跨重启复原。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ByteTransformer, Config

INIT_STD = 0.02          # 【律定承袭】model._init 的出生初始化 std，对称破缺噪声同尺度
MEM_BUDGET = int(1.9 * 2 ** 30)   # 【律定】GTX1060 3GB 训练足迹上限（交接文档 §2）
MEM_MARGIN = int(256 * 2 ** 20)   # 【值自成】激活+碎片裕量
WIDEN_AXIS_PRIORITY = ("mlp", "attn_v", "d_model")  # 【律定承袭规格 B】按序优先


# ============================================================================
# 移植体（轴①②）：老模块原封不动 + 独立旁路（零输出权重掩蔽）
# ============================================================================

class WidenedMLP(nn.Module):
    """轴①：MLP 隐层宽化移植体。

    老路径 proj(gelu(fc(x))) 与原实现同形同序（逐位）；每个旁路 = 新隐层单元组
    （fc 新行带对称破缺噪声 + 零输出权重列）。重复宽化时旁路表合并（老旁路
    原样保留，老路径 GEMM 形状依旧不变）。
    """

    def __init__(self, old, m, gen=None, noise_std=INIT_STD):
        super().__init__()
        C = old.fc.in_features
        self.fc = old.fc       # 原封：老路径逐位的根基
        self.proj = old.proj   # 原封
        bypass = list(getattr(old, "bypass_fcs", []) or [])
        pws = list(getattr(old, "proj_new_ws", []) or [])
        if m > 0:
            lin = nn.Linear(C, m)
            with torch.no_grad():
                if gen is not None:
                    lin.weight.copy_(torch.randn(m, C, generator=gen) * noise_std)
                else:  # 形态重放路径：数值将被 load_state_dict 覆盖
                    lin.weight.normal_(0.0, noise_std)
                lin.bias.zero_()
            bypass.append(lin)
            pws.append(nn.Parameter(torch.zeros(self.proj.out_features, m)))
        self.bypass_fcs = nn.ModuleList(bypass)
        self.proj_new_ws = nn.ParameterList(pws)

    def forward(self, x):
        y = self.proj(F.gelu(self.fc(x)))  # 老路径：与 model.MLP 同形同序
        for lin, w in zip(self.bypass_fcs, self.proj_new_ws):
            # w 零初始化 → 该项为精确 +0（IEEE754：有限值 + 0.0 逐位不变）；
            # 训练后 w 离开零点，新单元开始参与（对称破缺噪声已备好方向）。
            y = y + F.linear(F.gelu(lin(x)), w)
        return y

    def bypass_width(self):
        return sum(lin.out_features for lin in self.bypass_fcs)


class WidenedAttn(nn.Module):
    """轴②：attn-v 路径宽化移植体。

    老路径与 model.CausalSelfAttention.forward 逐算子同形同序（逐位）；旁路 =
    新 v 通道组（qkv 旁路新行 + 零输出权重列）。注意力权重 att 不受旁路影响
    （新通道只加宽 v/y 的通道维，不进 q·k 点积）——_softmax 后的判决逐位不变。
    要求 m % n_heads == 0（规格 B 整除保持）。
    """

    def __init__(self, old, m, gen=None, noise_std=INIT_STD):
        super().__init__()
        self.n_heads = old.n_heads
        self.qkv = old.qkv     # 原封
        self.proj = old.proj   # 原封
        self.register_buffer("mask", old.mask.clone())
        bq = list(getattr(old, "bypass_qkvs", []) or [])
        pws = list(getattr(old, "proj_new_ws", []) or [])
        if m > 0:
            if m % self.n_heads != 0:
                raise ValueError(f"attn_v 宽化量 {m} 必须被 n_heads={self.n_heads} 整除"
                                 f"（规格 B：n_heads 整除保持）")
            lin = nn.Linear(old.qkv.in_features, m)
            with torch.no_grad():
                if gen is not None:
                    lin.weight.copy_(torch.randn(m, old.qkv.in_features,
                                                 generator=gen) * noise_std)
                else:
                    lin.weight.normal_(0.0, noise_std)
                lin.bias.zero_()
            bq.append(lin)
            pws.append(nn.Parameter(torch.zeros(self.proj.out_features, m)))
        self.bypass_qkvs = nn.ModuleList(bq)
        self.proj_new_ws = nn.ParameterList(pws)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = lambda t, d: t.view(B, T, self.n_heads, d).transpose(1, 2)
        hd = C // self.n_heads
        q, k, v = shape(q, hd), shape(k, hd), shape(v, hd)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        z = self.proj(y)  # 老路径：与 model.CausalSelfAttention 同形同序
        for lin, w in zip(self.bypass_qkvs, self.proj_new_ws):
            mh = lin.out_features // self.n_heads
            yn = (att @ shape(lin(x), mh)).transpose(1, 2).contiguous().view(B, T, lin.out_features)
            z = z + F.linear(yn, w)  # w 零初始化 → 精确 +0
        return z

    def bypass_width(self):
        return sum(lin.out_features for lin in self.bypass_qkvs)


def has_transplants(model):
    """模型是否带轴①②移植体（轴③拒绝叠加——见 widen_d_model docstring）。"""
    for blk in model.blocks:
        if isinstance(blk.mlp, WidenedMLP) or isinstance(blk.attn, WidenedAttn):
            return True
    return False


# ============================================================================
# 幽灵探测（LN 免疫位）
# ============================================================================

IMMUNE_SITES = ("mlp_hidden", "attn_v")     # 【律定承袭规格 B】LN 免疫位
RESIDUAL_SITES = ("residual", "residual_stream", "ln1_out", "ln2_out", "attn_out")


def _ce_backward(model, data, device):
    """对一段字节流做一次 forward+backward，返回 (loss)。梯度留在参数上由调用方
    处置（probe 结束后 zero_grad）。"""
    blk = model.cfg.block_size
    b = torch.tensor(list(data[: blk * 2]), dtype=torch.long, device=device)
    if b.numel() < blk + 2:
        raise ValueError("ghost_probe 数据过短（需 ≥ block_size+2 字节）")
    x, y = b[None, :blk], b[None, 1:blk + 1]
    model.zero_grad(set_to_none=True)
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab), y.reshape(-1))
    loss.backward()
    return loss


def ghost_probe(model, layer, data, site=None, n_ghosts=8, seed=0, device=None):
    """LN 免疫位的容量增益探测（规格 B）。

    返回 {"mlp_hidden": gain 或 None, "attn_v": gain 或 None,
          "attn_v_per_head": [...], "loss": float}；
    gain = 幽灵输出参数将收到的梯度范数（按位置数归一），对 n_ghosts 个随机
    方向取均值（随机方向=对"残差梯度可利用成分"的草图估计）。

    site="residual_stream"（或任何残差流位）→ ValueError 拒绝挂载：
    残差流挂幽灵会被 √(1+Δ/d) 污染测量（对质已证伪，规格 B）。
    """
    if site in RESIDUAL_SITES:
        raise ValueError(
            "残差流位拒绝挂幽灵（√(1+Δ/d) 缝隙污染测量，对质已证伪；"
            f"LN 免疫位仅有 {IMMUNE_SITES}）")
    if site is not None and site not in IMMUNE_SITES:
        raise ValueError(f"未知探测位 {site}（免疫位：{IMMUNE_SITES}）")
    device = device or next(model.parameters()).device
    blk = model.blocks[layer]
    # 2026-10-05 呼吸实验发现并修复：与 life._redo 同款设备 bug——torch.Generator()
    # 默认 CPU，而 CUDA 上 randn(device=cuda, generator=cpu_gen) 抛 RuntimeError。
    # 幽灵探测此前从未在 CUDA 真实运行过（只在平台期第三阶梯排程时执行），
    # 且 life._schedule 的 except (ValueError, RuntimeError) 会把该异常静默吞成
    # "无增益"——即阶梯哪天打开，生长判决也会在 GPU 上静默失效。本修复只动
    # generator 设备侧，CPU 路径语义不变（种子口径原样保留）。
    gen = torch.Generator(device=device).manual_seed(int(seed))  # 探测不碰全局随机源（M0 等价性）
    out = {"mlp_hidden": None, "attn_v": None, "attn_v_per_head": [], "loss": None}
    T = min(model.cfg.block_size, len(data) - 1)

    if site in (None, "mlp_hidden"):
        box = {}
        h1 = blk.mlp.fc.register_forward_hook(lambda m, i, o: box.__setitem__("x", i[0]))
        h2 = blk.mlp.register_full_backward_hook(
            lambda m, gi, go: box.__setitem__("g", go[0]))
        loss = _ce_backward(model, data, device)
        h1.remove(); h2.remove()
        out["loss"] = float(loss)
        x, g = box["x"].detach()[0], box["g"].detach()[0]  # (T, C)
        gains = []
        for _ in range(n_ghosts):
            u = torch.randn(x.shape[1], generator=gen, device=device) * INIT_STD
            h_ghost = F.gelu(x @ u)                       # (T,)
            G = torch.einsum("tc,t->c", g, h_ghost)       # 幽灵输出列的梯度
            gains.append(float(G.norm() / max(1, T)))
        out["mlp_hidden"] = sum(gains) / len(gains)

    if site in (None, "attn_v"):
        box = {}
        h1 = blk.attn.qkv.register_forward_hook(
            lambda m, i, o: box.__setitem__("qkv", (i[0], o)))
        h2 = blk.attn.register_full_backward_hook(
            lambda m, gi, go: box.__setitem__("g", go[0]))
        loss = _ce_backward(model, data, device)
        h1.remove(); h2.remove()
        out["loss"] = float(loss)
        x_qkv, qkv_out = box["qkv"]
        x_qkv = x_qkv.detach()[0]
        g_attn = box["g"].detach()[0]                     # (T, C) dL/d attn 输出
        qkv_out = qkv_out.detach()[0]
        C = x_qkv.shape[1]
        q, k, _ = qkv_out.split(C, dim=1)
        H = blk.attn.n_heads
        hd = C // H
        shape = lambda t: t.view(T, H, hd).transpose(0, 1)  # (H, T, hd)
        q, k = shape(q), shape(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = blk.attn.mask[0, 0, :T, :T]   # (T,T)：探测路径无 batch 维，att 是
        att = att.masked_fill(mask == 0, float("-inf"))   # (H,T,T)——带 (1,1,T,T)
        att = F.softmax(att, dim=-1)                      # 的原 mask 广播会撑出 4 维
        per_head = []
        for h in range(H):
            gh = []
            for _ in range(n_ghosts):
                u = torch.randn(C, generator=gen, device=device) * INIT_STD
                v_ghost = x_qkv @ u                       # (T,)
                y_ghost = att[h] @ v_ghost                # (T,)
                G = torch.einsum("tc,t->c", g_attn, y_ghost)
                gh.append(float(G.norm() / max(1, T)))
            per_head.append(sum(gh) / len(gh))
        out["attn_v_per_head"] = per_head
        out["attn_v"] = max(per_head) if per_head else None
    model.zero_grad(set_to_none=True)
    return out


def ghost_scan(model, data, layers=None, **kw):
    """全层免疫位扫描（控制器用）：{(layer, site): gain}。"""
    res = {}
    for li in range(len(model.blocks)) if layers is None else layers:
        r = ghost_probe(model, li, data, **kw)
        if r.get("mlp_hidden") is not None:
            res[(li, "mlp_hidden")] = r["mlp_hidden"]
        if r.get("attn_v") is not None:
            res[(li, "attn_v")] = r["attn_v"]
    return res


# ============================================================================
# 变宽三轴
# ============================================================================

def widen_mlp(model, layer, delta, seed=0, noise_std=INIT_STD):
    """轴①：MLP 隐层 +delta 单元（精确函数保持，逐位）。"""
    gen = torch.Generator().manual_seed(int(seed))
    blk = model.blocks[layer]
    dev = blk.mlp.fc.weight.device  # 新模块必须跟上脑所在设备（CUDA 脑上新建
    before = blk.mlp.fc.out_features  # 的 nn.Linear 默认 CPU——不同步就是部署炸弹）
    blk.mlp = WidenedMLP(blk.mlp, int(delta), gen, noise_std).to(dev)
    return {"axis": "mlp", "layer": layer, "delta": int(delta),
            "hidden": f"{before} -> {before + int(delta)}"}


def widen_attn_v(model, layer, delta, seed=0, noise_std=INIT_STD):
    """轴②：attn-v 路径 +delta 通道（精确函数保持，逐位；delta % n_heads == 0）。"""
    gen = torch.Generator().manual_seed(int(seed))
    blk = model.blocks[layer]
    dev = blk.attn.qkv.weight.device
    delta = (int(delta) // blk.attn.n_heads) * blk.attn.n_heads  # 向下取整到整除
    before = blk.attn.qkv.out_features - 2 * blk.attn.qkv.in_features
    blk.attn = WidenedAttn(blk.attn, delta, gen, noise_std).to(dev)
    return {"axis": "attn_v", "layer": layer, "delta": int(delta),
            "v_channels": f"{before} -> {before + delta}"}


def _ri(t, k, dim, scale=1.0):
    u = torch.repeat_interleave(t, k, dim=dim)
    return u if scale == 1.0 else u * scale


def tile_state_dict(sd, k):
    """轴③状态字典变换：d_model ×k（数学精确；q/k/v 副本尺度 k^(-1/4)、残差流
    副本尺度 1/√k——由绑定头 k·t_r²=1 与注意力尺度 t²√k=1 联立确定）。"""
    t_r = k ** -0.5
    t_qk = k ** -0.25
    o = {}
    C = sd["wte.weight"].shape[1]
    o["wte.weight"] = _ri(sd["wte.weight"], k, 1, t_r)
    o["wpe.weight"] = _ri(sd["wpe.weight"], k, 1, t_r)
    for key, v in sd.items():
        if key.endswith(".mask"):
            o[key] = v.clone()
    n_layers = sum(1 for key in sd if key.startswith("blocks.") and key.endswith(".ln1.weight"))
    for li in range(n_layers):
        p = f"blocks.{li}."
        for ln in ("ln1", "ln2"):
            o[p + ln + ".weight"] = _ri(sd[p + ln + ".weight"], k, 0, t_r)
            o[p + ln + ".bias"] = _ri(sd[p + ln + ".bias"], k, 0, t_r)
        W = sd[p + "attn.qkv.weight"]
        b = sd[p + "attn.qkv.bias"]
        qw, kw, vw = W[:C], W[C:2 * C], W[2 * C:]
        qb, kb, vb = b[:C], b[C:2 * C], b[2 * C:]
        s_q = t_qk / (k * t_r)
        s_v = 1.0 / k
        o[p + "attn.qkv.weight"] = torch.cat([
            _ri(qw, k, 1, s_q).repeat_interleave(k, 0),
            _ri(kw, k, 1, s_q).repeat_interleave(k, 0),
            _ri(vw, k, 1, s_v).repeat_interleave(k, 0)], 0)
        o[p + "attn.qkv.bias"] = torch.cat([
            _ri(qb, k, 0, t_qk), _ri(kb, k, 0, t_qk), _ri(vb, k, 0, t_r)], 0)
        o[p + "attn.proj.weight"] = _ri(_ri(sd[p + "attn.proj.weight"], k, 0), k, 1, 1.0 / k)
        o[p + "attn.proj.bias"] = _ri(sd[p + "attn.proj.bias"], k, 0, t_r)
        o[p + "mlp.fc.weight"] = _ri(_ri(sd[p + "mlp.fc.weight"], k, 0), k, 1, t_r)
        o[p + "mlp.fc.bias"] = sd[p + "mlp.fc.bias"].repeat_interleave(k, 0)
        o[p + "mlp.proj.weight"] = _ri(_ri(sd[p + "mlp.proj.weight"], k, 0), k, 1, t_r / k)
        o[p + "mlp.proj.bias"] = _ri(sd[p + "mlp.proj.bias"], k, 0, t_r)
    o["ln_f.weight"] = _ri(sd["ln_f.weight"], k, 0, t_r)
    o["ln_f.bias"] = _ri(sd["ln_f.bias"], k, 0, t_r)
    o["head.weight"] = o["wte.weight"]  # 绑定头随 wte 平铺（k·t_r²=1 保证 logits 不变）
    return o


def widen_d_model(model, k, device=None, verify=True):
    """轴③：d_model 复制平铺 ×k（返回**新模型**与新 cfg——d_model 变了）。

    数学上精确（LN(平铺)=平铺(LN)——新脑 LayerNorm 的 eps 同步除以 k，见下方
    注释；注意力尺度由 q/k 副本尺度 k^(-1/4) 精确补偿；绑定头由残差流副本尺度
    1/√k 精确补偿）。浮点上 K 维随副本增长且各项非零，逐位不可达（前置实验
    实测相对误差 ~3e-6），用 allclose(1e-4,1e-4) 验收，并在 widen_d_model 内
    置自检。**拒绝叠加**：模型带轴①②移植体时抛错——移植体参数的平铺折叠
    留待下一棒。（措辞修正，监督审计 P1-2：控制器的手术排程从不停靠轴③
    ——life._schedule 的轴映射只有 ①/②，并无"自动改轴"动作；需轴③时由
    调用方自行改轴。）
    """
    if has_transplants(model):
        raise ValueError("轴③拒绝叠加：模型带轴①②移植体（折叠平铺留待下一棒）；"
                         "需变宽请改用 mlp/attn_v 轴（控制器排程本就只用这两轴）")
    if k < 2:
        raise ValueError("轴③倍数 k 必须 ≥2")
    device = device or next(model.parameters()).device
    old_cfg = model.cfg
    new_cfg = Config(d_model=old_cfg.d_model * k, n_layers=old_cfg.n_layers,
                     n_heads=old_cfg.n_heads, block_size=old_cfg.block_size,
                     dropout=old_cfg.dropout, vocab=old_cfg.vocab)
    new_model = ByteTransformer(new_cfg).to(device)
    # 【精确性命门】绑定头强制残差流副本尺度 t_r=k^{-1/2}（k·t_r²=1），而
    # LN(a·x)≠a·LN(x)——eps 项不随 a 缩放。把新脑全部 LayerNorm 的 eps 设为
    # eps_old/k 后，sqrt(t_r²σ²+eps')=t_r·sqrt(σ²+eps_old) 对任意 σ² 逐字成立，
    # LN(平铺)=平铺(LN) 才在含 eps 的定义下严格精确。不修此项：随机初始化权重
    # 的残差方差 (~4e-4) 与 eps(1e-5) 同量级，实测 |Δlogit|~4e-3；训练后期
    # σ²≫eps 时误差隐没——恰好骗过"只在训练后的脑上验"的侥幸。
    eps_old = model.blocks[0].ln1.eps
    for mod in new_model.modules():
        if isinstance(mod, nn.LayerNorm):
            mod.eps = eps_old / k
    new_model.load_state_dict(tile_state_dict(model.state_dict(), k))
    if verify:
        g = torch.Generator().manual_seed(1234)
        # 探测输入先用 CPU generator 生成再上目标设备（randint 的 generator 必须与
        # device 同侧，CUDA 直造会抛错——内置自检此前只在 CPU 上跑过）
        x = torch.randint(0, 256, (1, min(64, old_cfg.block_size)),
                          generator=g).to(device)
        model.eval(); new_model.eval()
        with torch.no_grad():
            l0, _ = model(x[:, :-1], x[:, 1:])
            l1, _ = new_model(x[:, :-1], x[:, 1:])
        if not torch.allclose(l0, l1, atol=1e-4, rtol=1e-4):
            raise RuntimeError(f"轴③平铺自检失败：max|Δlogit|="
                               f"{(l0 - l1).abs().max().item():.3e}")
    return new_model, new_cfg


def break_symmetry_dmodel(model, k, seed=0, noise_std=INIT_STD):
    """轴③对称破缺（训练恢复前单独调用）：对每块 attn.proj / mlp.proj 的副本行
    （通道 c 的第 j≥1 份）加 N(0, noise_std)。

    副本是活的（直接参与输出），此调用起输出改变——这正是噪声的目的：不加噪声
    的副本梯度逐位相同永不分化（规格 B）。调用后逐位性不再成立（属预期）。
    """
    gen = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for blk in model.blocks:
            for w_name in ("attn.proj.weight", "mlp.proj.weight"):
                w = dict(blk.named_parameters())[w_name]
                C_out = w.shape[0]
                for c in range(C_out // k):
                    for j in range(1, k):
                        w[c * k + j].add_(
                            torch.randn(w.shape[1], generator=gen,
                                        device=w.device) * noise_std)
    return {"axis": "d_model", "broken": True}


def widen(model, layer, delta, axis="auto", vitals_snap=None, seed=0, device=None):
    """变宽入口（规格 B，按序优先 ①mlp → ②attn_v → ③d_model）。

    axis="auto"：按优先级取第一条可行轴（attn_v 的 delta 自动向下取整到 n_heads
    倍数；d_model 要求无移植体且 delta 解释为倍数 k）。返回 record dict。
    轴①②原地改模型；轴③返回 (new_model, new_cfg) 放在 record["new_model"]。
    """
    axis = axis or "auto"
    if axis == "auto":
        for cand in WIDEN_AXIS_PRIORITY:
            try:
                return widen(model, layer, delta, axis=cand, seed=seed, device=device)
            except ValueError:
                continue
        raise ValueError("无可用变宽轴")
    if axis == "mlp":
        return widen_mlp(model, layer, delta, seed=seed)
    if axis == "attn_v":
        return widen_attn_v(model, layer, delta, seed=seed)
    if axis == "d_model":
        k = max(2, int(delta))
        nm, nc = widen_d_model(model, k, device=device)
        return {"axis": "d_model", "layer": "all", "delta": k,
                "new_model": nm, "new_cfg": nc}
    raise ValueError(f"未知变宽轴 {axis}")


# ============================================================================
# 收缩与 born-again
# ============================================================================

def shrink_config(cfg, vitals=None, target=0.7):
    """born-again 学生的小身体（规格 B）。

    - n_heads 整除保持：d_model' = n_heads × max(1, round(head_dim·target))；
    - 深度按账本逐层平均 stable-Taylor 重要性排序，保留最重要的
      round(n_layers·target) 层（学生从头初始化，层数=保留数；重要性排序决定
      收缩幅度并向记录交代"丢了谁"）；
    - vitals 缺席时按均匀重要性处理。
    返回 (new_cfg, info)。
    """
    target = float(target)
    if not 0.1 <= target <= 0.95:
        raise ValueError(f"shrink target={target} 出域 [0.1, 0.95]")
    head_dim = max(1, cfg.d_model // cfg.n_heads)
    d_new = cfg.n_heads * max(1, int(round(head_dim * target)))
    n_new = max(1, int(round(cfg.n_layers * target)))
    kept = list(range(cfg.n_layers))
    dropped = []
    if vitals is not None and vitals.er:
        # 逐层重要性 = 平均 stable-Taylor 份额（有效秩作次序参考：ER 低=谱更收敛）
        imp = {}
        for li in range(cfg.n_layers):
            vals = [sum(led.stable_tay) / max(1, led.C)
                    for k, led in vitals.sites.items()
                    if k.startswith(f"b{li}.") and ".new." not in k]
            imp[li] = sum(vals) / len(vals) if vals else 0.0
        ranked = sorted(range(cfg.n_layers), key=lambda li: imp[li])  # 最不重要在前
        dropped = sorted(ranked[: cfg.n_layers - n_new])
        kept = [li for li in range(cfg.n_layers) if li not in set(dropped)]
    elif n_new < cfg.n_layers:
        # 无账本：去尾（确定性），info 如实交代"丢了谁"——否则 kept/dropped 与
        # 实际 n_layers 自相矛盾（记录谎言比不记录更糟）
        dropped = list(range(cfg.n_layers - n_new, cfg.n_layers))
        kept = list(range(cfg.n_layers - n_new))
    new_cfg = Config(d_model=d_new, n_layers=n_new, n_heads=cfg.n_heads,
                     block_size=cfg.block_size, dropout=cfg.dropout, vocab=cfg.vocab)
    info = {"d_model": f"{cfg.d_model}->{d_new}", "n_layers": f"{cfg.n_layers}->{n_new}",
            "dropped_layers": dropped, "kept_layers": kept,
            "target": target, "n_heads_divisible": d_new % cfg.n_heads == 0}
    return new_cfg, info


def born_again_student(cfg, vitals, target, device, seed=0):
    """born-again 学生：shrink_config 的从头初始化（规格 B）。

    返回 (student_model, student_opt, info)。训练由 life.run_cycle 的 WITHERING
    战役承担（kd_alpha 提到 0.75，仅限手术验收周期；训练数据=记忆库 dream+复活+
    缓冲真实回放，L5 锚定）。
    """
    s_cfg, info = shrink_config(cfg, vitals, target)
    torch.manual_seed(int(seed))  # 学生出生种子（从头初始化，与教师无关）
    student = ByteTransformer(s_cfg).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=0.01)
    return student, opt, info


def rebuild_optimizer(old_model, old_opt, new_model, lr):
    """手术后优化器重建（规格 B：受影响参数优化器状态重置 + LR 重启）。

    同名同形状的参数**迁移** Adam 动量（未受手术影响的脑区不丢巩固进度）；
    新生/变形参数状态重置。weight_decay 与 Hemisphere 惯例一致（0.01）。
    """
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=lr, weight_decay=0.01)
    old_named = dict(old_model.named_parameters())
    migrated = reset = 0
    for name, p in new_model.named_parameters():
        op = old_named.get(name)
        st = old_opt.state.get(op) if op is not None else None
        if st and st.get("exp_avg") is not None and st["exp_avg"].shape == p.shape:
            new_opt.state[p] = {
                "step": st["step"].clone(),
                "exp_avg": st["exp_avg"].clone().to(p.device),
                "exp_avg_sq": st["exp_avg_sq"].clone().to(p.device),
            }
            migrated += 1
        else:
            reset += 1
    return new_opt, {"migrated": migrated, "reset": reset}


# ============================================================================
# 显存预算与形态持久化
# ============================================================================

def delta_params_mlp(C, delta):
    """轴①新增参数量：fc (delta×C 权重 + delta 偏置) + 旁路输出列 (C×delta)。"""
    return delta * C + delta + C * delta


def delta_params_attn_v(C, delta):
    """轴②新增参数量：旁路 qkv (delta×C + delta) + 输出列 (C×delta)。"""
    return delta * C + delta + C * delta


def delta_params_dmodel(cfg, k):
    """轴③新增参数量 ≈ 总参数×k − 总参数（粗账：全部权重随 d_model 线性增长，
    嵌入/头占大头）。"""
    per_layer = (4 * cfg.d_model * cfg.d_model      # qkv
                 + cfg.d_model * cfg.d_model        # attn proj
                 + 4 * cfg.d_model * cfg.d_model    # mlp fc
                 + 4 * cfg.d_model * cfg.d_model    # mlp proj
                 + 2 * cfg.d_model)                 # 两个 LN
    base = cfg.vocab * cfg.d_model + cfg.block_size * cfg.d_model
    total1 = base + cfg.n_layers * per_layer
    totalk = base * k + cfg.n_layers * per_layer * k * k
    return totalk - total1


def mem_budget_ok(device, delta_params, reserved=None):
    """显存预算检查（规格 B）：预计足迹 = 当前预留 + Δ参数×16B + 裕量 ≤ 1.9GB。

    超预算 → 推迟变宽（WDDM 倒页 6 倍减速的教训）。CPU 恒放行（无倒页问题）。
    """
    if device != "cuda":
        return True, "cpu 无显存预算约束"
    used = torch.cuda.memory_reserved() if reserved is None else reserved
    projected = used + delta_params * 16 + MEM_MARGIN
    return projected <= MEM_BUDGET, (f"预计 {projected / 2**20:.0f}MB / 预算 "
                                     f"{MEM_BUDGET / 2**20:.0f}MB"
                                     f"（当前预留 {used / 2**20:.0f}MB）")


def empty_morphology():
    return {"d_model_k": 1, "mlp": {}, "attn_v": {}}


def morphology_of(model):
    """从模型实例读出形态记录（save 用）。"""
    morph = empty_morphology()
    for li, blk in enumerate(model.blocks):
        if isinstance(blk.mlp, WidenedMLP):
            morph["mlp"][str(li)] = [lin.out_features for lin in blk.mlp.bypass_fcs]
        if isinstance(blk.attn, WidenedAttn):
            morph["attn_v"][str(li)] = [lin.out_features for lin in blk.attn.bypass_qkvs]
    return morph


def apply_morphology(base_cfg, morph, device):
    """按形态记录重建模型（load 用）：先建平铺基座，再重放移植体结构
    （参数数值由随后的 load_state_dict 填入）。返回 (model, real_cfg)。

    - d_model_k：轴③平铺倍数（真实 d_model = 学生/基座 d_model × k）；
    - shrink：born-again 学生体型（d_model/n_layers 直接取学生出生值——
      平铺史已折进该值；无记录则用基座）。没有它，凋零换装后的脑存档
      load 时会按基座 cfg 重建 → state_dict 形状不匹配 → 加载崩溃。
    - eps：k>1 时 LayerNorm eps 除以 k（与 widen_d_model 同一精确性约定，
      state_dict 不含 eps，只能按形态重放——漏掉则平铺脑重启后函数静默漂移）。"""
    morph = morph or empty_morphology()
    k = int(morph.get("d_model_k", 1) or 1)
    shrink = morph.get("shrink") or {}
    d0 = int(shrink.get("d_model", base_cfg.d_model) or base_cfg.d_model)
    n0 = int(shrink.get("n_layers", base_cfg.n_layers) or base_cfg.n_layers)
    real_cfg = Config(d_model=d0 * k, n_layers=n0,
                      n_heads=base_cfg.n_heads, block_size=base_cfg.block_size,
                      dropout=base_cfg.dropout, vocab=base_cfg.vocab)
    m = ByteTransformer(real_cfg).to(device)
    if k > 1:
        for mod in m.modules():
            if isinstance(mod, nn.LayerNorm):
                mod.eps = mod.eps / k
    for li_s, widths in (morph.get("mlp") or {}).items():
        blk = m.blocks[int(li_s)]
        blk.mlp = WidenedMLP(blk.mlp, 0)  # 空壳；旁路结构由下面的重放补齐
        lins = [nn.Linear(real_cfg.d_model, int(w)) for w in (widths or [])]
        blk.mlp.bypass_fcs = nn.ModuleList(lins)
        blk.mlp.proj_new_ws = nn.ParameterList(
            [nn.Parameter(torch.zeros(real_cfg.d_model, int(w))) for w in (widths or [])])
    for li_s, widths in (morph.get("attn_v") or {}).items():
        blk = m.blocks[int(li_s)]
        blk.attn = WidenedAttn(blk.attn, 0)
        lins = [nn.Linear(real_cfg.d_model, int(w)) for w in (widths or [])]
        blk.attn.bypass_qkvs = nn.ModuleList(lins)
        blk.attn.proj_new_ws = nn.ParameterList(
            [nn.Parameter(torch.zeros(real_cfg.d_model, int(w))) for w in (widths or [])])
    # 新建的移植体模块默认 CPU——整体跟上目标设备（load_state_dict 的 copy_ 不会
    # 改变参数自身设备，漏掉这步会造出 CUDA/CPU 混血模型）
    m = m.to(device)
    return m, real_cfg
