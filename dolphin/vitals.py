"""营养因子账本（M4 自生长/自凋零的测量层）。

逐通道记录四类营养因子，供手术与回收决策：
  1. 激活范数 EMA（利用率原料）——tag（快）/ stable（慢）双时间尺度；
  2. 一阶 Taylor 重要性 EMA——**只用 |g×x| 绝对值口径**（Q6 定标证实：带符号 g×x
     的符号分量在 token 间随机抵消，半侧均值符号相反的通道过半，是纯噪声）；
  3. 每层残差流有效秩（RankMe 式奇异值熵，Gram 精确累计）；
  4. 休眠占比（层内分位法）。

通道口径与 Q6 定标一致 = 各 Linear 的**输入激活维度**（手术对象"单元"）：
  b{i}.ln1_out / b{i}.attn_out / b{i}.ln2_out / b{i}.mlp_hidden
手术旁路新生单元（轴①②移植体的 bypass）单独成账本（键含 ".new.{i}"）——
规格 A 明文：新老通道（手术后新生的）分开归一化。

全部关键数值的来源标注（律固定，值自成）：
  - TAG_HALF_LIFE = 20 步        【定标】Q6 建议带 15–30，带内取 20；实测 h=15 相对
                                 h=∞ 仅增 1–2 个百分点波动。
  - STABLE_HALF_LIFE = 240 步    【定标】Q6 建议 ≥240（2 周期），iid 外推把通道
                                 噪声从 ~3% 压到 ~1.5%。
  - DORMANT_QUANTILE = 0.05      【定标】层内分位 bottom-5%（两半 Jaccard
                                 0.32–0.45，显著高于随机基线 ~0.03）；bottom-1%
                                 名单洗牌（~0.2）不得作为手术对象。
  - DORMANT_CONFIRM = 2          【定标】连续 2 周期确认——把 ~0.4 的单周期集合
                                 稳定性复利成跨周期确认。
  - stable-Taylor 周期中位数重整  【定标】Taylor 有 trend=-0.27 的系统性量级衰减
                                 （跨周期漂移 ~35% 主导）；stable 口径不重整会把
                                 梯度量级衰减误读成"全员凋零"。
  - PROBATION_CYCLES = 3         【值自成】新生/回收通道观察期，期间豁免休眠判决。

采集安全（Q6 定标脚本用显存换来的教训，此处照方抓药）：
  - 禁止张量钩子 x.register_hook(闭包捕获 x)——x→钩子表→闭包→x 引用环滞留整段
    反向子图，120 步即溢出 3GB 显存进 WDDM（6 倍减速）。
  - 正确做法：模块级 register_full_backward_hook（每模块注册一次），forward 钩子
    只把输入暂存进暂存表，backward 钩子用完即 pop——零滞留、零引用环。
  - Gram 全程 no_grad，绝不并进自动微分图。
  - 只在睡脑训练时 enable（训练线程内），评估/服务路径零足迹。

通道身份管理（手术后必须重映射，规格 A）：
  - 复制分裂（轴③ d_model 平铺）：子通道继承父账本值/k（k 份合起来守恒，
    "各半"的 k 份推广）；
  - 轴①②旁路新生单元：独立账本，初始化即 probation，与老通道分开归一；
  - ReDo 回收：账本重置 + probation 重新生效；
  - 回滚：widen 前的 to_state 快照原样恢复。
"""
import math

import torch
import torch.nn.functional as F

# —— 常量（来源身份见模块 docstring）——
TAG_HALF_LIFE = 20.0        # 【定标】Q6 建议带 15–30 步，带内取 20
STABLE_HALF_LIFE = 240.0    # 【定标】Q6 建议 ≥240 步（2 个 120 步周期）
DORMANT_QUANTILE = 0.05     # 【定标】层内分位 bottom-5%
DORMANT_CONFIRM = 2         # 【定标】连续 2 周期确认
PROBATION_CYCLES = 3        # 【值自成】新生/回收通道观察期
SITE_NAMES = ("ln1_out", "attn_out", "ln2_out", "mlp_hidden")


def _vec_mean(t):
    """(B,T,C) 或 (T,C) → (C,) 逐通道均值（CPU list）。"""
    return t.detach().abs().mean(dim=tuple(range(t.dim() - 1))).cpu().tolist()


class SiteLedger:
    """单个 site 的逐通道账本。数值用 CPU list（序列化 = weights_only 安全）。"""

    def __init__(self, name, C, values=None):
        self.name = name
        self.C = int(C)
        v = values or {}
        n = self.C
        self.tag_act = list(v.get("tag_act", [0.0] * n))
        self.tag_tay = list(v.get("tag_tay", [0.0] * n))
        self.stable_act = list(v.get("stable_act", [0.0] * n))
        self.stable_tay = list(v.get("stable_tay", [0.0] * n))
        self.dorm_cycles = list(v.get("dorm_cycles", [0] * n))
        self.is_new = list(v.get("is_new", [False] * n))
        self.probation = list(v.get("probation", [0] * n))
        self.sum_act = [0.0] * n
        self.sum_tay = [0.0] * n
        self.n_steps = 0

    def to_state(self):
        return {"C": self.C, "tag_act": self.tag_act, "tag_tay": self.tag_tay,
                "stable_act": self.stable_act, "stable_tay": self.stable_tay,
                "dorm_cycles": self.dorm_cycles, "is_new": self.is_new,
                "probation": self.probation}

    @classmethod
    def from_state(cls, name, st):
        return cls(name, st["C"], st)

    # —— 每步采集（hook 调用；vec 为本步逐通道均值 list）——

    def observe(self, act_means, tay_means):
        if len(act_means) != self.C:
            return  # 形态与账本不符且未经 remap：丢弃本步（回滚/换脑竞态兜底）
        a = 1.0 - 0.5 ** (1.0 / TAG_HALF_LIFE)  # 【定标】tag EMA 步级系数
        for c in range(self.C):
            self.tag_act[c] += a * (act_means[c] - self.tag_act[c])
            self.tag_tay[c] += a * (tay_means[c] - self.tag_tay[c])
            self.sum_act[c] += act_means[c]
            self.sum_tay[c] += tay_means[c]
        self.n_steps += 1

    # —— 周期折叠 ——

    def end_cycle(self, steps_in_cycle):
        if self.n_steps == 0:
            return
        mean_act = [s / self.n_steps for s in self.sum_act]
        mean_tay = [s / self.n_steps for s in self.sum_tay]
        # 【定标】stable-Taylor 周期中位数重整：除掉梯度量级系统性衰减（trend=-0.27），
        # stable 口径只保留相对份额。
        med = sorted(mean_tay)[len(mean_tay) // 2]
        denom = med if med > 1e-12 else 1.0
        renorm = [t / denom for t in mean_tay]
        alpha = 1.0 - 0.5 ** (steps_in_cycle / STABLE_HALF_LIFE)  # 【定标】周期级折叠
        for c in range(self.C):
            self.stable_act[c] += alpha * (mean_act[c] - self.stable_act[c])
            self.stable_tay[c] += alpha * (renorm[c] - self.stable_tay[c])
        # 休眠判决：层内分位 bottom-5%（【定标】bottom-1% 洗牌不可用），连续 2 周期确认；
        # probation 通道豁免（规格 D：新生/回收通道不在判决域）。
        pool = [c for c in range(self.C) if self.probation[c] == 0]
        if pool:
            vals = sorted((mean_act[c], c) for c in pool)
            k = max(1, int(round(DORMANT_QUANTILE * len(pool))))
            bottom = {c for _, c in vals[:k]}
            for c in pool:
                self.dorm_cycles[c] = self.dorm_cycles[c] + 1 if c in bottom else 0
        for c in range(self.C):  # probation 递减；期满摘新生帽
            if self.probation[c] > 0:
                self.probation[c] -= 1
                if self.probation[c] == 0:
                    self.is_new[c] = False
        self.sum_act = [0.0] * self.C
        self.sum_tay = [0.0] * self.C
        self.n_steps = 0

    # —— 查询 ——

    def dormant(self):
        """确认休眠通道：连续 2 周期 bottom-5% 且不在观察期。"""
        return [c for c in range(self.C)
                if self.probation[c] == 0 and self.dorm_cycles[c] >= DORMANT_CONFIRM]

    def shares(self):
        """层内分位数归一化份额（【预留接口，现状无生产消费者】——预留给战斗
        决策（如 M4 手术层选址）使用；休眠判决**不经过它**，end_cycle 直接按
        原始激活值排序取 bottom-5%。措辞修正（监督审计 P2-1）：旧注释称本函数
        影响"休眠判决"，系前瞻性辩护冒充现状（用途漂移），已删）。

        禁全局绝对阈值：营养竞争=相对份额。返回 {通道: [0,1]}，1=site 内相对
        最营养。份额 = 激活分位与 Taylor 分位的均值（双因子等权）。老账本与
        新账本天然分离（分开归一）。
        """
        if self.C == 0:
            return {}
        def pct(vals):
            # 并列值取平均秩：全等 → 全 0.5 中性。分位数必须只由数值决定——
            # 若按索引洗牌打破并列，全等的 Taylor 账本会给"索引靠后"的通道虚高
            # 份额（实测能把最沉睡通道推到 1.0）。此修正属于本函数自身的正确性
            # 要求，与休眠判决无关（判决不走 shares()）。
            n = len(vals)
            if n <= 1:
                return [0.5] * n
            order = sorted(range(n), key=lambda i: vals[i])
            out = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                    j += 1
                avg_rank = (i + j) / 2.0
                for t in range(i, j + 1):
                    out[order[t]] = avg_rank / (n - 1)
                i = j + 1
            return out
        pa = pct(self.stable_act)
        pt = pct(self.stable_tay)
        return {c: 0.5 * (pa[c] + pt[c]) for c in range(self.C)}

    def utilization_indices(self, q=DORMANT_QUANTILE, p=0.10):
        """连续利用率指数（研究报告_自适应原理 §3.1：恒温器的"钙"）。

        返回 (utilization_gap, headroom_hot)，均从 **stable 激活**（禁止 tag——
        测量阻尼第一重）构造，分位池与休眠判决同域（probation 豁免）：
          gap = (med − bottom-q 均值) / med ∈ [0,1]——尾部落后量（过剩信号，
                连续化的休眠占比；0=尾部齐平，1=尾部死透）；
          hot = top-p 均值 / med——头部过热程度（1=完全齐平，>1 有过热）。
        med ≤ 1e-12（冷账本/全零）→ (0.0, 1.0)（既不过剩也不过热——无信号）。
        分位并列不再特殊处理：指数读的是均值距离不是名单，并列天然给中性值。
        """
        pool = [c for c in range(self.C) if self.probation[c] == 0]
        if not pool:
            return 0.0, 1.0
        vals = sorted(self.stable_act[c] for c in pool)
        n = len(vals)
        med = vals[n // 2]
        if med <= 1e-12:
            return 0.0, 1.0
        k = max(1, int(round(q * n)))
        tail = sum(vals[:k]) / k
        kp = max(1, int(round(p * n)))
        top = sum(vals[-kp:]) / kp
        return (med - tail) / med, top / med

    def median_share(self):
        sh = self.shares()
        return sorted(sh.values())[len(sh) // 2] if sh else 0.0

    # —— 手术后身份重映射 ——

    def remap_split(self, k):
        """复制分裂（轴③）：每通道 → k 份，各继承父值/k（合起来守恒）。"""
        if k <= 1:
            return
        def rep(vals, divide):
            out = []
            for v in vals:
                out.extend([v / k] * k if divide else [v] * k)
            return out
        self.tag_act = rep(self.tag_act, True)
        self.tag_tay = rep(self.tag_tay, True)
        self.stable_act = rep(self.stable_act, True)
        self.stable_tay = rep(self.stable_tay, True)
        self.dorm_cycles = rep(self.dorm_cycles, False)
        self.is_new = rep(self.is_new, False)
        self.probation = rep(self.probation, False)
        self.sum_act = [0.0] * (self.C * k)
        self.sum_tay = [0.0] * (self.C * k)
        self.C *= k

    def remap_reset(self, idxs):
        """ReDo 回收：账本重置为中位数 + probation（规格：回收通道账本重置）。"""
        idxs = [c for c in idxs if 0 <= c < self.C]
        if not idxs:
            return
        alive = [c for c in range(self.C) if c not in set(idxs)]
        if not alive:
            alive = list(range(self.C))
        med_act = sorted(self.stable_act[c] for c in alive)[len(alive) // 2]
        med_tay = sorted(self.stable_tay[c] for c in alive)[len(alive) // 2]
        for c in idxs:
            self.tag_act[c] = med_act; self.tag_tay[c] = med_tay
            self.stable_act[c] = med_act; self.stable_tay[c] = med_tay
            self.dorm_cycles[c] = 0; self.is_new[c] = True
            self.probation[c] = PROBATION_CYCLES
            self.sum_act[c] = 0.0; self.sum_tay[c] = 0.0


def _iter_bypasses(blk):
    """枚举块内手术旁路（新单元 Linear）：[("mlp_hidden", i, lin), ("attn_v", j, lin)]。"""
    out = []
    for i, lin in enumerate(getattr(blk.mlp, "bypass_fcs", []) or []):
        out.append(("mlp_hidden", i, lin))
    for i, lin in enumerate(getattr(blk.attn, "bypass_qkvs", []) or []):
        out.append(("attn_v", i, lin))
    return out


class Vitals:
    """一颞大脑的营养因子账本 + 训练期 hook 采集器。

    生命周期：life.run_cycle 每周期 attach(睡脑模型) → enable() → 训练循环 →
    disable() → detach() → end_cycle()。账本内容跨周期持久（随档保存）。
    """

    def __init__(self):
        self.sites = {}          # site 名 → SiteLedger
        self.er = {}             # 层 → 有效秩（每周期一个值）
        self.enabled = False
        self._handles = []
        self._stash = {}         # 定标教训：暂存/弹出，零引用环
        self._grams = {}         # 层 → 残差流 Gram
        self._steps_seen = 0

    # ---------- 持久化 ----------

    def to_state(self):
        return {"sites": {k: v.to_state() for k, v in self.sites.items()},
                "er": {str(k): v for k, v in self.er.items()}}

    def from_state(self, st):
        self.sites = {k: SiteLedger.from_state(k, v)
                      for k, v in (st.get("sites") or {}).items()}
        self.er = {int(k): v for k, v in (st.get("er") or {}).items()}
        return self

    # ---------- hook 装卸 ----------

    def attach(self, model):
        """注册训练期采集 hook（模块级、每模块一次；每周期重挂，重复调用安全）。"""
        self.detach()
        dev = next(model.parameters()).device
        for li, blk in enumerate(model.blocks):
            # 主 site：通道口径 = Linear 输入激活维度（Q6 定标口径）
            main = {"ln1_out": blk.attn.qkv, "attn_out": blk.attn.proj,
                    "ln2_out": blk.mlp.fc, "mlp_hidden": blk.mlp.proj}
            for sname, lin in main.items():
                key = f"b{li}.{sname}"
                self._ensure_site(key, lin.in_features)
                self._install_io_hook(key, lin, kind="in")
            # 旁路新生单元：独立账本（新老分开归一，规格 A）
            for kind, i, lin in _iter_bypasses(blk):
                key = f"b{li}.{kind}.new.{i}"
                self._ensure_site(key, lin.out_features)
                self._install_io_hook(key, lin, kind="bypass")
            # 每层残差流 Gram（有效秩原料）——全程 no_grad（定标教训：误入微分图
            # 会把测量链进反向子图，逐步滞留激活）
            self._ensure_gram(li, dev)

            def make_blk(li=li):
                def fwd(mod, inp, out):
                    if not self.enabled:
                        return
                    with torch.no_grad():
                        X = out[0] if isinstance(out, tuple) else out
                        X = X.detach().reshape(-1, X.shape[-1])
                        if self._grams.get(li) is None:  # 首步按真实尺寸建
                            self._grams[li] = torch.zeros(X.shape[1], X.shape[1],
                                                          device=X.device)
                        Xc = X - X.mean(dim=0, keepdim=True)
                        self._grams[li] += Xc.T @ Xc
                return fwd

            self._handles.append(blk.register_forward_hook(make_blk(li)))

    def _install_io_hook(self, key, lin, kind):
        """主 site：|x| 与 |g×x|（输入侧）；旁路代理：|gelu(o)| 与 |g·o|（输出侧，
        量纲不同的独立账本，不与主账本直接比较）。"""
        stash = self._stash

        def make_fwd():
            def fwd(mod, inp, out):
                if not self.enabled:
                    return
                if kind == "in":
                    stash[key] = inp[0]
                else:
                    stash[key] = out
            return fwd

        def make_bwd():
            def bwd(mod, grad_input, grad_output):
                if not self.enabled:
                    return
                t = stash.pop(key, None)
                if t is None:
                    return
                with torch.no_grad():
                    if kind == "in":
                        g = grad_input[0] if grad_input else None
                        if g is None:
                            return
                        act = t.detach().abs()
                        tay = (g.detach() * t.detach()).abs()
                    else:
                        g = grad_output[0] if grad_output else None
                        if g is None:
                            return
                        o = t.detach()
                        act = F.gelu(o).abs()
                        tay = (g.detach() * o).abs()
                    self.sites[key].observe(_vec_mean(act), _vec_mean(tay))
            return bwd

        self._handles.append(lin.register_forward_hook(make_fwd()))
        self._handles.append(lin.register_full_backward_hook(make_bwd()))

    def detach(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        self._stash = {}

    def _ensure_site(self, name, C):
        led = self.sites.get(name)
        if led is None or led.C != C:
            self.sites[name] = SiteLedger(name, C)  # 无 remap 的形态变化：重建（兜底）

    def _ensure_gram(self, li, dev):
        self._grams[li] = None  # 延迟到首步按真实尺寸/设备建（形态变化安全）

    # ---------- 采集开关与周期折叠 ----------

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def end_cycle(self, steps_in_cycle):
        """周期折叠：stable EMA 合并、Taylor 中位数重整、休眠确认、有效秩计算。"""
        for led in self.sites.values():
            led.end_cycle(steps_in_cycle)
        for li, G in list(self._grams.items()):
            if G is not None and G.abs().sum() > 0:
                w = torch.linalg.eigvalsh(G.double().cpu())
                s = torch.sqrt(torch.clamp(w, min=0.0))
                ss = s.sum()
                if ss > 0:
                    p = s / ss
                    p = p[p > 0]
                    self.er[li] = float(torch.exp(-(p * p.log()).sum()).item())
            self._grams[li] = None
        self._steps_seen = 0

    # ---------- 汇总查询（控制器用） ----------

    def dormant_report(self):
        """(确认休眠清单, 全模型休眠占比)。老账本与旁路新账本合并计占比。"""
        rep = {}
        n_dorm = n_all = 0
        for k, v in self.sites.items():
            d = v.dormant()
            if d:
                rep[k] = d
            n_dorm += len(d)
            n_all += v.C
        frac = n_dorm / n_all if n_all else 0.0
        return rep, frac

    def site_names(self):
        return sorted(self.sites)
