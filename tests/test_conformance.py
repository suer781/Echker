"""律符合度常备测试：每次改动后必须全绿。

来源：2026-10-03 实现符合度审计的 8 组实验固化。审计员发现的问题
（L9 蒸发、负奖励钳位、L6 算子缺失）由此类实验抓获——本文件是
"声明 vs 代码"的常设对账机制，不依赖任何人自觉。

用法：python tests/test_conformance.py   （纯 CPU，约 1 分钟）
"""
import atexit
import contextlib
import hashlib
import math
import os
import secrets
import shutil
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.dolphin import Dolphin
from dolphin.experience import ExperienceBuffer
from dolphin.memory import MemoryStore
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


# ============ 冷层隔离总闸（2026-10-04，生产事故后施工） ============
# 事故档案：上一次修复 resurrect() 后跑全量测试，t_g3_guard_ruminate 的
# guard_loop 死循环 15 分钟，期间不断从**真实生产冷层**做梦注入、再逐回落盘，
# 把 dolphin/memory_cold.jsonl 从 15498 行撑到 16316 行（+818 行垃圾）。
# 该文件不在 git 版本控制内（.gitignore 第 7 行）→ 损坏不可逆、无法 checkout 恢复。
#
# 为什么用「全局改写 MemoryStore 默认值」而不是「每个测试包 try/finally」：
#   逐测试 try/finally 依赖每个测试作者都记得包一层——而本文件历史上正是
#   「19 处构造忘了包」才出的事故。忘了包不会有任何报错，只会静默污染生产数据，
#   且由于文件不受 git 保护，事后再想补救已经晚了。改写成全局默认后，
#   「忘记隔离」从「静默污染」变成「不可能发生」——这是安全等级的量级差别。
#   代价：本文件再新增测试时，冷层自动落在临时目录，无需（也不应）再手动包
#   try/finally；临时目录的生命周期由模块级 atexit 统一收口。
#
# 三道防线（互相独立，任何一道单独失效都不会导致生产数据被改）：
#   ① 默认值改写：MemoryStore(cold_path=None) 解析出的若是生产路径，一律换成
#      隔离沙箱内的路径。覆盖本文件全部构造点，且对**将来**新增的构造点自动生效。
#   ② 显式路径也拦：即使有人显式传入 cold_path=生产路径，仍被改写——
#      意图不明的显式传参不能绕过隔离（这是 ① 单独做不到的）。
#   ③ 指纹哨兵：模块导入时记录生产冷层的 sha256/行数/mtime/尺寸，测试全部跑完
#      后比对。任一项变化 → 显式报错（红线断言），不静默通过。
#      ①② 是「让它碰不到」，③ 是「万一碰到了必须喊出来」——互不替代。
#
# 生产代码零改动：dolphin/memory.py 的 cold_path 默认语义、dolphin/dolphin.py 的
# MemoryStore() 调用全部原样保留，改写只发生在本测试模块的进程内命名空间里。

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROD_COLD = os.path.realpath(os.path.join(_ROOT, "dolphin", "memory_cold.jsonl"))
_PROD_PROBE = os.path.realpath(os.path.join(_ROOT, "probes", "probe.txt"))

# 隔离沙箱：整个测试进程共用一个根目录，每个 MemoryStore 实例在其下独占一个文件。
# 模块导入时创建一次，atexit 统一 rmtree（ignore_errors：Windows 上文件句柄
# 可能延迟释放，残留一个临时目录无害，生产数据绝不能残留）。
_SANDBOX = tempfile.mkdtemp(prefix="dolphin_cold_sandbox_")
atexit.register(shutil.rmtree, _SANDBOX, True)

_sandbox_seq = 0


def _sandbox_cold_path():
    """为隔离沙箱内的每个冷层实例分配独占路径（不复用，避免实例间互相看见）。"""
    global _sandbox_seq
    _sandbox_seq += 1
    return os.path.join(_SANDBOX, f"cold_{_sandbox_seq:04d}.jsonl")


# ---- 防线①②：改写 MemoryStore 默认冷层解析 ----
_orig_ms_init = MemoryStore.__init__


def _quarantined_ms_init(self, cap_entries=500, promote_hits=3, cold_path=None):
    """MemoryStore.__init__ 的隔离版：解析结果若指向生产冷层，换成沙箱路径。

    只在「解析结果 == 生产冷层」时改写：显式传入的临时路径（t_g1_cold_layer、
    t_g10_* 等自带 mkdtemp 的用例）原样保留，那些用例断言的就是自己的沙箱文件。
    """
    resolved = os.path.realpath(cold_path) if cold_path else _PROD_COLD
    if resolved == _PROD_COLD:
        cold_path = _sandbox_cold_path()
    _orig_ms_init(self, cap_entries, promote_hits, cold_path)


MemoryStore.__init__ = _quarantined_ms_init


# ---- 防线③：指纹哨兵 ----


def _fingerprint(path):
    """生产资产的 (行数, sha256, mtime_ns, 字节数)。

    行数与 sha256 抓内容变化；mtime_ns 与字节数抓「内容碰巧相同但被重写过」
    （flush_cold 是全量重写：内容一致时行数与 sha256 都不变，只有 mtime 变——
    只比内容会漏掉这种写入）。mtime_ns 用 ns 精度，避免秒级精度撞时间戳的漏检。
    """
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    h = hashlib.sha256()
    lines = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            lines += chunk.count(b"\n")
    return (lines, h.hexdigest(), st.st_mtime_ns, st.st_size)


_BEFORE_COLD = _fingerprint(_PROD_COLD)
_BEFORE_PROBE = _fingerprint(_PROD_PROBE)


# 测试特征串（2026-10-08 audit4/audit5 防御纵深）：三道防线已让测试不可能写
# 生产文件，这条哨兵防的是「外部进程恰好把测试数据写进生产冷层」的极端串扰。
# 出现任一标记 → 即使方向是追加/重写也判红（堵住「尾部追加测试数据」假阴性盲区）。
_TEST_MARKERS = (
    "预热记录：让经验编号从 1 开始",
    "游标验证题目",
    "循环喂食验证题目",
    "隔离自检条目",
    "恒温器测试回放流",
    "回滚验证记录",
    "门控验证记录",
    "双通道验证记录",
    "持久化回环测试记录",
    "并发喂食记录",
    "原子写验证记录",
    "滚动备份验证记录",
    "哈希校验验证记录",
    "门控失效验证记录",
    "阈值钳位验证记录",
    "复活回归",
    "重启失聪回归",
    "凋零战役验证记录",
    "等价性验证记录",
    "回收验证记录",
    "锚持久化验证记录",
    "状态机验证记录",
    "恒温器集成验证",
    "反刍旧事",
    "自杀回归旧条目",
    "部署侧新经验流入",
)


def _contains_test_marker(path):
    """文件内容里是否出现测试特征串（出现则视为测试污染，判红）。"""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return False
    return any(m.encode("utf-8") in data for m in _TEST_MARKERS)


def _is_append_only(path, before, after):
    """判断 after 是否可解释为 before 的「纯追加/仅重写」。

    2026-10-07 修复（审计B）：生产进程 feed.py 会在测试期间并发写入
    memory_cold.jsonl——_demote 是追加新行，flush_cold 是同内容全量重写。
    这两类都是「外部进程的正常写入」，不是测试污染，应降级为警告。

    2026-10-08 修复（audit4/audit5）：flush_cold 按影子索引全量重写文件，
    条目元数据（hits/score/created）会更新 → 文件头 b_size 字节 sha 必变，
    原「头部 sha 不匹配 → 判红」在生产进程正常写冷层时恒假红。采用方案A：
    对 memory_cold.jsonl 放宽为「非删减方向即视为外部正常写入」——行数/字节数
    增长或不变（不是删减）→ 降级警告；只有删减（行数/字节数变少）才判红。
    probe.txt 仍严格（任何变化判红，由调用方 allow_append=False 保证）。

    同时补上审计指出的假阴性盲区：若 after 中出现测试特征串（测试独有文本），
    即使方向是追加/重写也判红——防止「尾部追加测试数据」被误降级为警告。

    判定基于「内容变化方向」而非 mtime（外部写入时 mtime_ns 必然变化）：
    - before 为 None（测试前文件不存在）→ 测试后出现文件 = 外部首次创建，算追加
      （但含测试特征串则判红）；
    - after 为 None（文件被删除）→ 删减，判红；
    - 行数/字节数变少 → 删减/重写，判红；
    - 行数/字节数不变或增长 → 非删减方向：外部进程正常写入（flush_cold 会重写
      头部元数据，无法用「头部一致性」判断），降级为警告——但含测试特征串则判红。
    """
    if before is None:
        return after is not None and not _contains_test_marker(path)
    if after is None:
        return False  # 文件被删除 → 判红
    b_lines, _, _, b_size = before
    a_lines, _, _, a_size = after
    if a_lines < b_lines or a_size < b_size:
        return False  # 变少：删减/重写 → 判红
    if _contains_test_marker(path):
        return False  # 测试特征串出现 → 测试污染，判红（堵假阴性盲区）
    return True  # 非删减方向 → 外部进程正常写入（flush_cold 重写/追加），降级警告


def _judge_asset_change(name, path, before, after, allow_append):
    """判断单个生产资产的变化。返回 (是否完好, 明细, 警告文本或 None)。

    allow_append=True：memory_cold.jsonl 允许「外部进程纯追加/仅重写」→ 降级警告。
    allow_append=False：probe.txt 是评测基准，任何变化都判红。
    """
    if before == after:
        return True, f"{name} 零改动（{after[0]} 行 sha256={after[1][:12]}…）", None
    # 有差异：先构造可读的指纹描述
    def _desc(fp):
        if fp is None:
            return "不存在"
        return f"{fp[0]} 行/{fp[3]}B sha256={fp[1][:12]}…"
    diff = f"{name} 已被改动！测试前={_desc(before)} 测试后={_desc(after)}"
    if allow_append and _is_append_only(path, before, after):
        warning = (f"{name} 在测试期间被外部进程写入（追加方向，测试前 "
                    f"{_desc(before)} → 测试后 {_desc(after)}）。"
                    f"红线降级为警告：测试自身未污染生产资产。")
        return True, diff + "（判定：外部进程正常写入，非测试污染）", warning
    return False, diff + "（判定：非追加方向的改动，判红）", None


def verify_cold_quarantine():
    """防线③：比对测试前后生产资产的指纹。返回 (是否完好, 警告列表, 明细)。

    在全部测试跑完之后调用。2026-10-07 修复（审计B）：原先「任何差异都判红」，
    但生产进程 feed.py 会在测试期间并发写入 memory_cold.jsonl（正常冷层日志
    追加/重写），导致红线误报。现区分两类变化：

    - 外部进程正常写入：memory_cold.jsonl 呈「非删减方向」——行数/字节数不变
      或增长（flush_cold 全量重写/追加，头部元数据会变，无法用头部一致性判断；
      2026-10-08 audit4 方案A）。这类变化不是测试造成的，降级为黄色警告，
      ok 仍为 True。
    - 测试自身污染/异常改动：行数/字节数变少（删减/重写）、内容含测试特征串
      （防御纵深，堵「尾部追加测试数据」假阴性盲区）、或 probe.txt 变化
      （评测基准被污染）——一律判红。

    判定基于「内容变化方向」而非 mtime（外部写入时 mtime_ns 必然变化）。
    """
    after_cold = _fingerprint(_PROD_COLD)
    after_probe = _fingerprint(_PROD_PROBE)
    warnings = []
    ok = True
    detail = []

    cold_ok, cold_detail, cold_warning = _judge_asset_change(
        "memory_cold.jsonl", _PROD_COLD, _BEFORE_COLD, after_cold,
        allow_append=True)
    if cold_warning:
        warnings.append(cold_warning)
    if not cold_ok:
        ok = False
    detail.append(cold_detail)

    probe_ok, probe_detail, probe_warning = _judge_asset_change(
        "probe.txt", _PROD_PROBE, _BEFORE_PROBE, after_probe,
        allow_append=False)
    if probe_warning:
        warnings.append(probe_warning)
    if not probe_ok:
        ok = False
    detail.append(probe_detail)

    return ok, warnings, "  ".join(detail)


# ---- 显式工厂：让「本测试的 Dolphin 一律用临时冷层」在代码里看得见 ----


def make_dolphin(cfg=None, device="cpu", probe_path=None):
    """构造 Dolphin，并把 d.memory 换到隔离沙箱的独立冷层。

    有了防线①②，本工厂在**正确性**上已是冗余的（默认路径已被改写到沙箱）；
    保留它是为了让「本文件不碰生产冷层」这件事在源码里自证，而非依赖读者
    记得模块顶部那段 monkeypatch。两者同时存在时：改写管兜底，工厂管可读性。

    必须在构造后**立刻**替换、且替换前不做任何 add/retrieve——MemoryStore
    构造本身只填字段不落盘（Dolphin.__init__ 也不会读写冷层），所以此处替换
    是安全的；但若将来 Dolphin.__init__ 改成预加载冷层，这个顺序就会失效，
    故在此注明。
    """
    d = Dolphin(cfg=cfg, device=device, probe_path=probe_path)
    # H3：生产默认 param_adaptive_enabled=True；测试统一在此关闭，确保参数
    # 自适应控制律（压力三段式下跌/微调）不在测试中触发，保持 M4 判决主路径
    # 精确可控（test_growth_support 的 ward 场景不跑 run_cycle，无需额外隔离）。
    d.param_adaptive_enabled = False
    d.memory = MemoryStore(cold_path=_sandbox_cold_path())
    return d


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
    """L5：全部训练路径（sleep/life 与 feed 驱动器）无 generate 调用；回滚逐张量。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("dolphin/sleep.py", "dolphin/life.py"):  # 审计④：两条训练路径都要扫
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        check(f"L5 训练路径无自生成（{rel}）", "generate(" not in src)
        check(f"L5 训练用真实字节流（{rel}）", "varied_replay" in src)
    src = open(os.path.join(root, "feed.py"), encoding="utf-8").read()
    check("L5 驱动器无自生成（feed.py）", "generate(" not in src)

    with _probe_fixture("dolphin_l5l8_") as probe:
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
        d2 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
    from dolphin.life import run_cycle
    # 冷层落盘与探测集都隔离到临时目录：吃生产默认 probes/probe.txt 会让
    # 探测集一挪走就G8 fail-fast 拒绝开睡，做梦注入照常但断言全灭。
    with _probe_fixture("dolphin_g1c_") as probe:
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                         probe_path=probe, device="cpu")
        d.memory = MemoryStore(cold_path=os.path.join(os.path.dirname(probe), "c.jsonl"))
        cue = "做梦联想线索条目：海豚用回声定位寻找沙丁鱼群，记忆库旧事。"
        d.memory.add(cue.encode(), 3.0, 0, "residue")
        d.learn(f"喂食记录携带与旧事重叠的联想线索：{cue}", source="test")
        hits0 = d.memory.entries[0].hits
        # M4 取代：feed.trainer_cycle 已删，喂食语义由 life.run_cycle(feeding=True) 唯一实现
        report = run_cycle(d, steps=2, feeding=True)
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

    # 2026-10-04 回归：len(data) % bs == 1 时旧实现会丢最后 1 字节
        # （range(0, len-1, bs) 的 -1 使尾字节永不进块）。修复后覆盖恒等仍成立。
        with _probe_fixture("dolphin_g2a_mod1_") as mod1:
            _write_synthetic_volume(mod1, 256 + 1)  # 257 % 256 == 1
            total_m = os.path.getsize(mod1)
            cm = load_chunks(mod1, 256)
            check("G2-a 修复：len%bs==1 覆盖字节 == 全卷字节（不丢尾字节）",
                  sum(len(c) for c in cm) == total_m,
                  f"{sum(len(c) for c in cm)}/{total_m}")
            check("G2-a 修复：len%bs==1 尾块仍可评分（≥2 字节）",
                  all(len(c) >= 2 for c in cm),
                  f"尾块 {len(cm[-1])}B")

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
    # 2026-10-04 回归：n 小于 MIN_GATE_CHUNKS 时判决带消失（旧实现 n=1 时 SE=0、
    # eps=EPS_FLOOR → 任何 >1e-6 的"改善"即放行 = 无统计依据却换班）。
    # 修复后 n 不足应判否，且报告里带 gate_reason。
    from dolphin.probe import MIN_GATE_CHUNKS
    p_small, d_small = gate_decision([5.0], [5.0 - 1e-5])
    check("G2-b n<MIN_GATE_CHUNKS 时拒绝换班（防判决带消失）",
          p_small is False and bool(d_small.get("gate_reason")),
          f"n=1 passed={p_small} reason={d_small.get('gate_reason')}")
    # 恰好在下限 n=MIN_GATE_CHUNKS 时统计判决仍应工作（非退化）
    x = [5.0] * MIN_GATE_CHUNKS
    p_min, d_min = gate_decision(x, [v - 1e-3 for v in x])
    check("G2-b n==MIN_GATE_CHUNKS 真实改善仍放行",
          p_min is True and d_min["chunks"] == MIN_GATE_CHUNKS,
          f"n={d_min['chunks']} passed={p_min}")
    # 同步守卫（M4 漂移警告）：两条训练路径都必须走公共判决，掷硬币判决不得复活
    # （M4 取代：feed.trainer_cycle 已删，喂食路径在 dolphin/life.py）
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel, call in (("dolphin/sleep.py", "dolphin.gate("), ("dolphin/life.py", "dolphin.gate(")):
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

        # 2026-10-04 回归：中途新增字典序更小的候选文件，指针不得错位——
        # 旧实现用 file_idx（排序下标），池文件集合一变就错位到新文件，
        # 导致候选题静默丢失 + 已消费区间重复进卷。
        probe3 = os.path.join(tmp, "probe3.txt")
        pool3 = os.path.join(tmp, "pool3")
        ret3 = os.path.join(tmp, "retired3")
        os.makedirs(pool3)
        open(probe3, "wb").write(b"S" * 300)
        open(os.path.join(pool3, "m.txt"), "wb").write(b"M" * 100 + b"N" * 100)
        open(os.path.join(pool3, "z.txt"), "wb").write(b"Z" * 100)
        rotate(probe_path=probe3, pool_dir=pool3, retired_dir=ret3, section_bytes=100)
        rotate(probe_path=probe3, pool_dir=pool3, retired_dir=ret3, section_bytes=100)
        # 此时已消费 m.txt 两段（共 200B），指针应在 m.txt 的 offset=200（已消费完）
        # 中途插入字典序更小的 0_new.txt —— 旧实现会把它误判为"已消费 200B"
        open(os.path.join(pool3, "0_new.txt"), "wb").write(b"X" * 100 + b"Y" * 100)
        r6 = rotate(probe_path=probe3, pool_dir=pool3, retired_dir=ret3, section_bytes=100)
        # 应继续消费 z.txt 的第一段（100B），而不是跳过/错位到 0_new.txt 的中间
        c6 = open(probe3, "rb").read()
        check("G2-c 修复：中途新增文件指针不错位（不丢候选不重复）",
              r6.get("source") == "z.txt"
              and c6[-100:] == b"Z" * 100,
              f"source={r6.get('source')} 卷尾={c6[-100:][:4]!r}")
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
    from dolphin.memory import MemoryStore

    # 测试夹具自造探测集，不绑死生产 probes/probe.txt（与 t_g8_gate_failfast 同一套做法）：
    # 本组要验的是反刍自续，不是探测集。若吃生产默认路径，探测卷一旦被挪走/改名/清空，
    # G8 fail-fast 会让睡眠周期（life.run_cycle，监督审计 P1-2 名词修正：trainer_cycle
    # 已删）拒绝开睡 → 反刍注入照常但周期恒为 0，
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
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                         probe_path=probe, device="cpu")
        # 冷层必须隔离到临时目录（同本文件 t_g1_dream_feed 的写法）：Dolphin() 默认
        # MemoryStore() 指向生产 dolphin/memory_cold.jsonl。本组断言的前提是「记忆库
        # 可控地空/非空」，若读生产冷层，场景 2 的 max-idle 收工断言就永不成立
        # （ruminate 会从真实冷层持续做梦注入 → idle_cycles 永不达标 → 死循环），
        # 且反刍注入会污染生产冷层。
        d.memory = MemoryStore(cold_path=os.path.join(tmp, "g3a.jsonl"))
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
        d2 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                          probe_path=probe, device="cpu")
        d2.memory = MemoryStore(cold_path=os.path.join(tmp, "g3a2.jsonl"))  # 同上：冷层隔离
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
            d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
            d2 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                              probe_path=probe, device="cpu")
            d2.load(path)
            os.remove(path)
            check("G3-b 游标随档存读一致", d2.feed_cursor == cursor, f"{d2.feed_cursor}")
            # 续喂：游标 (0,2) → 跳过 fakeA 前两条
            d3 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                              probe_path=probe, device="cpu")
            cursor3 = {"sources": list(names), "source_idx": 0, "record_idx": 2}
            d3.feed_cursor = cursor3
            fed3, _, _ = feed_from(d3, names, cursor3, {})
            first = d3.buffer.items[0].data.decode("utf-8", errors="replace")
            check("G3-b 续喂从游标起（跳过已吃记录）",
                  fed3 == 2 + 3 and first == feed_mod.serialize(src_a[2]), f"fed3={fed3}")
    finally:
        feed_mod.iter_source = orig_iter


def t_loop_feed():
    """无限循环喂食（--loop-feed，2026-10-07）：数据源喂尽自动轮转回第一个源。

    loop=True 时游标 round 递增、数据永远滋长（quota 人工限次生效，不无限空转）；
    空源/全碎片零产出不崩溃、round 不无限递增、游标不越界；非 loop 模式保持旧
    有限批次语义（不写 round 字段）。全部落在隔离沙箱，不碰生产资产。
    """
    import feed as feed_mod
    from feed import feed_from
    orig_iter = feed_mod.iter_source
    try:
        def mk(i):
            return {"prompt": f"循环喂食验证题目第{i}号：甲乙丙丁戊己庚辛壬癸顺序编号。",
                    "answer": f"标准答案正文第{i}号，长度足以通过碎片过滤线。", "source": "fake"}
        src_a = [mk(i) for i in range(3)]
        src_b = [mk(10 + i) for i in range(2)]
        feed_mod.iter_source = lambda name: iter(src_a if name == "fakeA" else src_b)
        with _probe_fixture("dolphin_loopfeed_") as probe:
            # a) 两源轮转：quota=6 > 两源总长 5 → 进入第二轮，round 递增
            names = ["fakeA", "fakeB"]
            d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                             probe_path=probe, device="cpu")
            cursor = {"sources": list(names), "source_idx": 0, "record_idx": 0}
            d.feed_cursor = cursor
            fed, hit, _ = feed_from(d, names, cursor, {}, loop=True, quota=6)
            check("loop-feed 两源轮转后 fed=6 且限次命中",
                  fed == 6 and hit is True, f"fed={fed} hit={hit}")
            check("loop-feed 轮转后 round 递增为 1",
                  cursor.get("round", 0) == 1, f"cursor={cursor}")
            check("loop-feed 游标位置指向第二轮中（source_idx/record_idx 正确）",
                  cursor["source_idx"] == 0 and cursor["record_idx"] == 1,
                  f"cursor={cursor}")

            # b) 空源 names=[]：返回 (0, False, 0) 不崩溃
            d2 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                              probe_path=probe, device="cpu")
            cursor2 = {"sources": [], "source_idx": 0, "record_idx": 0}
            fed2, hit2, skip2 = feed_from(d2, [], cursor2, {}, loop=True, quota=10)
            check("loop-feed 空源列表返回 (0, False, 0) 不崩溃",
                  (fed2, hit2, skip2) == (0, False, 0), f"{(fed2, hit2, skip2)}")

            # c) 空源（有源名但源为空）：零产出返回，round 不无限递增、游标不越界
            feed_mod.iter_source = lambda name: iter([])  # 全部源为空
            d3 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                              probe_path=probe, device="cpu")
            cursor3 = {"sources": ["fakeA", "fakeB"], "source_idx": 0, "record_idx": 0}
            fed3, hit3, skip3 = feed_from(d3, ["fakeA", "fakeB"], cursor3, {},
                                          loop=True, quota=10)
            check("loop-feed 空源零产出返回 (0, False, 0) 不崩溃",
                  (fed3, hit3, skip3) == (0, False, 0), f"{(fed3, hit3, skip3)}")
            check("loop-feed 空源 round 不无限递增（≤1，轮转尝试标记）",
                  cursor3.get("round", 0) <= 1, f"cursor3={cursor3}")
            check("loop-feed 空源游标不越界",
                  cursor3["source_idx"] <= len(cursor3["sources"])
                  and cursor3["record_idx"] == 0, f"cursor3={cursor3}")

            # d) 非 loop 模式不写 round 字段（旧有限批次语义不变）
            feed_mod.iter_source = orig_iter
            feed_mod.iter_source = lambda name: iter(src_a if name == "fakeA" else src_b)
            d4 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                              probe_path=probe, device="cpu")
            cursor4 = {"sources": list(names), "source_idx": 0, "record_idx": 0}
            fed4, _, _ = feed_from(d4, names, cursor4, {}, loop=False)
            check("loop-feed 非 loop 模式不写 round 字段",
                  "round" not in cursor4 and fed4 == 5,
                  f"fed4={fed4} cursor4={cursor4}")
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
            d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
        d2 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                         probe_path=probe, device="cpu")
        d.learn("哈希校验验证记录，内容足够长。", source="test")
        path = os.path.join(tmp, "state.pt")
        d.save(path)
        with open(probe, "ab") as f:  # 篡改：追加一字节即改变卷面语义
            f.write(b"X")
        d2 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                          probe_path=probe, device="cpu")
        rejected = False
        try:
            d2.load(path)
        except RuntimeError as e:
            rejected = "探测" in str(e) or "sha256" in str(e)
        check("G7-d 篡改 probe 后 load 拒绝", rejected)
        open(probe, "wb").write(body)  # 恢复原文
        d3 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
    from dolphin.life import run_cycle
    from feed import SleepTrainer
    tmp = tempfile.mkdtemp(prefix="dolphin_g8_")
    try:
        probe = os.path.join(tmp, "probe.txt")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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
            run_cycle(d, steps=2, feeding=True)  # 喂食训练路径（M4 唯一实现）
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
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
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


# ============ 2026-10-04 resurrect() 两缺陷回归（间隔重复吞吐 / 冷层重启失聪） ============

def t_g10_resurrect_all():
    """缺陷 A：resurrect() 遍历中删除 → 隔一条漏一条（吞吐减半 + 状态不一致）。

    原实现 `for en in self.cold_entries: ... self.cold_entries.remove(en)`，
    remove 让索引左移、下一条被跳过：N 条全达标只复活 N/2 条，残留条目 hits
    已≥阈值却没被消费、一直占冷层位。取偶数 N 让"隔一条漏一条"无可遁形。
    """
    from dolphin.memory import MemoryStore
    for n in (6, 8):
        tmp = tempfile.mkdtemp(prefix="dolphin_g10a_")
        try:
            cold = os.path.join(tmp, "c.jsonl")
            # cap=2：每 add 一条挤出一条，add N+2 条正好落盘 N 条冷层
            src = MemoryStore(cap_entries=2, promote_hits=3, cold_path=cold)
            for i in range(n + 2):
                src.add(f"复活回归第{i:03d}号，内容各异可逐字核对。{i}".encode(), 0.0, 0, "residue")
            texts = [e.text for e in src.cold_entries]
            for e in src.cold_entries:
                e.hits = 3  # 全部达标
            src.flush_cold()
            check(f"G10-a 夹具前置：{n} 条冷层已落盘且全部达标",
                  len(texts) == n and all(e.hits >= 3 for e in src.cold_entries),
                  f"冷层{len(texts)}条")

            m = MemoryStore(cap_entries=1000, promote_hits=3, cold_path=cold)
            m._ensure_cold()
            hot_before = len(m.entries)
            got = m.resurrect()
            check(f"G10-a {n} 条全达标必须全部复活（原实现只回 {n // 2}）",
                  len(got) == n, f"返回{len(got)}条")
            check(f"G10-a {n} 条冷层残留必须为 0（漏网条目 hits 已达标却占位）",
                  len(m.cold_entries) == 0, f"残留{len(m.cold_entries)}条")
            check(f"G10-a {n} 条热层增量 == {n}",
                  len(m.entries) - hot_before == n, f"热层{len(m.entries)}")
            check(f"G10-a {n} 条返回 bytes 集合与原集合一致",
                  set(got) == {t.encode("utf-8") for t in texts})
            check(f"G10-a {n} 条升回热层后 hits 已复位为 0",
                  all(e.hits == 0 for e in m.entries if e.text in set(texts)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def t_g10_resurrect_cold_load():
    """缺陷 B：resurrect() 不调 _ensure_cold() → 断电重启后冷层失聪（违反 L9）。

    部署喂食路径 feed.py --resume 走 life.run_cycle(feeding=True) → resurrect()
    （监督审计 P1-2 名词修正：trainer_cycle 已删），不调 serve()
    也就不会触发 retrieve()，于是上一进程逐出的知识在新进程完全不可见，
    记忆写了落盘却在重启后失聪 = 静默丢弃。本测试用新建实例（不经 retrieve）
    复现该路径。冷层一律落临时目录，绝不碰生产 dolphin/memory_cold.jsonl。
    """
    from dolphin.memory import MemoryStore
    tmp = tempfile.mkdtemp(prefix="dolphin_g10b_")
    try:
        cold = os.path.join(tmp, "memory_cold.jsonl")
        src = MemoryStore(cap_entries=2, promote_hits=3, cold_path=cold)
        for i in range(6):  # cap=2 → 落盘 4 条
            src.add(f"重启失聪回归第{i:03d}号，内容可核对。{i}".encode(), 0.0, 0, "residue")
        texts = [e.text for e in src.cold_entries]
        for e in src.cold_entries:
            e.hits = 3
        src.flush_cold()
        n_cold = sum(1 for l in open(cold, encoding="utf-8") if l.strip())
        check("G10-b 夹具前置：4 条冷层已落盘", n_cold == 4 and len(texts) == 4,
              f"文件{n_cold}行")

        # 关键：全新实例，不经 retrieve()/dream()，直接 resurrect()（= 喂食路径）
        m2 = MemoryStore(cap_entries=1000, promote_hits=3, cold_path=cold)
        check("G10-b 全新实例冷层尚未加载（复现前提）",
              m2._cold_loaded is False and len(m2.cold_entries) == 0)
        got = m2.resurrect()
        check("G10-b 新实例直接 resurrect 须读到 4 条（原实现返回 0 条 = 重启失聪）",
              len(got) == 4, f"返回{len(got)}条")
        check("G10-b 新实例 resurrect 后冷层清空、4 条升回热层",
              len(m2.cold_entries) == 0
              and {e.text for e in m2.entries} == set(texts),
              f"冷层{len(m2.cold_entries)} 热层{len(m2.entries)}")

        # 闭环：复活后 flush → 第三个实例重启仍能读到（落盘一致性未被 _ensure_cold 破坏）
        m2.flush_cold()
        m3 = MemoryStore(cap_entries=1000, promote_hits=3, cold_path=cold)
        check("G10-b 复活并回写后重启冷层为空（升回热层者不该留在冷层）",
              m3.resurrect() == [] and len(m3.cold_entries) == 0)
        check("G10-b 生产冷层文件未被本测试触碰",
              not cold.startswith(os.path.dirname(os.path.dirname(
                  os.path.abspath(__file__)))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==================== 2026-10-04 M4 自生长/自凋零施工新增（交接 §6 规格 A-E） ====================


def t_m4_vitals():
    """M4-A 账本：双时间尺度、stable-Taylor 周期中位数重整（量级衰减≠全员凋零）、
    休眠=bottom-5%+连续 2 周期确认、复制分裂守恒继承、probation 豁免、随档回环。"""
    from dolphin.vitals import (DORMANT_CONFIRM, PROBATION_CYCLES, SiteLedger, Vitals)
    led = SiteLedger("t.mlp_hidden", 8)
    # ① stable-Taylor 重整：整层等比例衰减（梯度量级系统性衰减，Q6 trend=-0.27）
    #    → 归一化份额与相对距离必须不变（否则会把全局衰减误读成"全员凋零"）
    led.observe([1.0] * 8, [4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    led.end_cycle(10)
    r1 = (led.stable_tay[0] / led.stable_tay[1], led.stable_act[0])
    led.observe([0.5] * 8, [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 全体减半
    led.end_cycle(10)
    r2 = led.stable_tay[0] / led.stable_tay[1]
    check("M4-A stable-Taylor 中位数重整抗量级衰减", abs(r1[0] - 2.0) < 1e-6
          and abs(r2 - 2.0) < 1e-6, f"份额比 c1={r1[0]:.4f} c2={r2:.4f}（应恒为 2）")
    # ② 休眠判决：bottom-5% 单周期不计入，连续 2 周期确认（定标 D）
    led2 = SiteLedger("t2.mlp_hidden", 8)
    led2.observe([5.0] * 7 + [0.01], [1.0] * 8)
    led2.end_cycle(10)
    check("M4-A 单周期 bottom 不判休眠（需连续确认）",
          7 not in led2.dormant() and led2.dorm_cycles[7] == 1,
          f"dorm={led2.dormant()} 计数={led2.dorm_cycles[7]}")
    led2.observe([5.0] * 7 + [0.01], [1.0] * 8)
    led2.end_cycle(10)
    check("M4-A 连续 2 周期确认休眠", led2.dormant() == [7],
          f"确认={DORMANT_CONFIRM} dorm={led2.dormant()}")
    # ③ 复制分裂（轴③）：值/k 继承、合起来守恒（规格 A"各半"的 k 份推广）
    led3 = SiteLedger("t3.ln1_out", 4)
    led3.stable_act = [8.0, 4.0, 2.0, 1.0]
    led3.remap_split(2)
    check("M4-A 复制分裂继承父账本各半（守恒）",
          led3.C == 8 and led3.stable_act == [4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 0.5, 0.5],
          f"{led3.stable_act}")
    # ④ ReDo 账本重置 + probation 豁免（规格：回收通道账本重置、probation 生效）
    led3.remap_reset([0, 1])
    check("M4-A 回收通道账本重置+probation 生效",
          led3.probation[0] == PROBATION_CYCLES and led3.is_new[0]
          and led3.dorm_cycles[0] == 0 and 0 not in led3.dormant(),
          f"probation={led3.probation[0]}")
    # ⑤ 逐层分位归一化：份额在 [0,1] 且单调对应原值（禁全局绝对阈值）
    sh = led2.shares()
    ok = all(0.0 <= v <= 1.0 for v in sh.values()) and \
        sh[0] > sh[7]  # 激活 5.0 的通道份额高于 0.01 的通道
    check("M4-A 逐层分位数归一化（相对份额）", ok, f"shares={ {k: round(v,2) for k,v in sh.items()} }")
    # ⑥ 随档回环（weights_only 安全反序列化兼容：纯 list/dict）
    v = Vitals()
    v.sites = {"a.mlp_hidden": led2, "a.ln1_out": led3}
    st = v.to_state()
    import json
    json.dumps(st)  # 必须可 JSON 化（torch.save weights_only 的载荷纪律）
    v2 = Vitals().from_state(st)
    check("M4-A 账本随档回环一致",
          v2.sites["a.mlp_hidden"].stable_act == led2.stable_act
          and v2.sites["a.ln1_out"].C == led3.C)


def t_m4_dev_birth():
    """发育模式出生检查钉子（2026-10-05）：① Vitals.remap_split 全账本委托——
    life._execute_grow 轴③路径调用 v.remap_split(...)，而委托方法此前只存在于
    SiteLedger，轴③生长真实执行即 AttributeError（恒温器排程从不停靠轴③，该路径
    仅手工计划可达，故既有断言从未踩到；发育出生检查首次真实演练轴③时暴露）；
    ② 钉住发育种子身体规格：d32×2 层×2 头 = 41,856 参数（发育实验的出生体重）。"""
    from dolphin.vitals import Vitals, SiteLedger
    v = Vitals()
    led_a = SiteLedger("a.ln1_out", 4)
    led_a.stable_act = [8.0, 4.0, 2.0, 1.0]
    led_a.dorm_cycles = [1, 0, 2, 0]
    led_b = SiteLedger("a.mlp_hidden", 2)
    led_b.stable_act = [3.0, 5.0]
    v.sites = {"a.ln1_out": led_a, "a.mlp_hidden": led_b}
    v.remap_split(2)
    check("M4-发育 Vitals.remap_split 全账本委托（轴③执行路径可达）",
          led_a.C == 8 and led_a.stable_act == [4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 0.5, 0.5]
          and led_a.dorm_cycles == [1, 1, 0, 0, 2, 2, 0, 0]
          and led_b.C == 4 and led_b.stable_act == [1.5, 1.5, 2.5, 2.5],
          f"ln1_out C={led_a.C} {led_a.stable_act}；mlp_hidden C={led_b.C} {led_b.stable_act}")
    from dolphin.model import Config, ByteTransformer
    cfg = Config(d_model=32, n_layers=2, n_heads=2, block_size=256)
    m = ByteTransformer(cfg)
    n = sum(p.numel() for p in m.parameters())
    check("M4-发育 种子身体规格 d32/L2/H2 随机初始化 41,856 参数", n == 41856,
          f"{n:,}（57M 身体的 0.07%；发育模式的出生体重）")
    # ③ 轴②移植体的捕获对账（2026-10-05 发育实验发现并修复：父账本须经
    #    _AXIS_LEDGER 映射到 attn_out 位——旧实现直接查 "b0.attn_v" 得 None →
    #    对账单被无声清除 → 轴②移植体的 λ_g 捕获结算结构性死。轴①前缀
    #    mlp_hidden 恰与主账本同名，故既有 T8 测试从未踩到。）
    from dolphin.life import LifeController, LAMBDA_G
    ctl = LifeController()
    ctl.vitals["B"] = Vitals()
    led_new = SiteLedger("b0.attn_v.new.0", 8, {"is_new": [True] * 8, "probation": [3] * 8})
    led_new.stable_act = [0.9] * 8
    led_old = SiteLedger("b0.attn_out", 32)
    led_old.stable_act = [0.1] * 32
    ctl.vitals["B"].sites["b0.attn_v.new.0"] = led_new
    ctl.vitals["B"].sites["b0.attn_out"] = led_old
    ctl.pending_capture = {"due_cycle": 0, "keys": ["b0.attn_v.new.0"], "hname": "B"}

    class _D:
        cycle = 5
    rep = {}
    ctl._capture_check(_D(), rep, ctl.vitals["B"])
    check("M4-发育 轴②捕获对账父账本经 _AXIS_LEDGER 解析（attn_v→attn_out）",
          abs((rep.get("m4_capture") or {}).get("rate", -1) - 1.0) < 1e-9
          and abs(ctl.lambda_g - LAMBDA_G * 1.1) < 1e-9
          and ctl.pending_capture is None,
          f"rate={(rep.get('m4_capture') or {}).get('rate')} "
          f"λ_g={ctl.lambda_g}（修复前 rate 缺失、对账单无声清除）")


def t_m4_widen_exact():
    """M4-B 宽化精确保持（本工程命门）：随机输入下三轴分别验证——轴①②逐位相等
    （torch.equal），轴③数学精确（allclose，浮点 K 维变化的物理极限 ~3e-6 相对）。
    另验：对称破缺、训练可跑、轴③拒绝对带移植体模型叠加、attn_v 整除保持。"""
    from dolphin.surgery import (break_symmetry_dmodel, has_transplants, widen_attn_v,
                                 widen_d_model, widen_mlp)
    cfg = Config(d_model=64, n_layers=2, n_heads=4, block_size=64)
    torch.manual_seed(11)
    m = ByteTransformer(cfg)
    m.eval()
    x = torch.randint(0, 256, (1, 48))
    with torch.no_grad():
        l0, _ = m(x[:, :-1], x[:, 1:])

    # 轴①：MLP 隐层（fc 新行=对称破缺噪声，被零输出权重掩蔽 → 逐位）
    rec1 = widen_mlp(m, 0, 16, seed=5)
    with torch.no_grad():
        l1, _ = m(x[:, :-1], x[:, 1:])
    check("M4-B 轴① MLP 隐层宽化逐位相等", torch.equal(l0, l1),
          f"max|Δ|={(l0 - l1).abs().max().item():.3e}")
    # 移植体形态（精确性的根基=老 GEMM 原封不动：GEMM 内核按形状选择，原地
    # 加行/列会改变 K 维归约次序 → 老输出 ~4e-7 抖动，逐位相等即破——前置浮点
    # 实验定标，见 surgery.py 模块 docstring）。新单元走旁路：零输出权重掩蔽
    # （精确 +0），对称破缺噪声藏在掩蔽后面就位。
    check("M4-B 轴① 移植体就位（老 GEMM 原封+旁路新单元+零输出掩蔽+破缺噪声）",
          m.blocks[0].mlp.fc.out_features == 4 * 64
          and m.blocks[0].mlp.bypass_width() == 16
          and float(m.blocks[0].mlp.proj_new_ws[0].abs().sum()) == 0.0
          and float(m.blocks[0].mlp.bypass_fcs[0].weight.abs().sum()) > 0.0)
    # 宽化后训练可跑（新单元梯度通路活着：proj_new_w 有梯度）
    m.train()
    tgt = torch.randint(0, 256, (1, 48))
    _, loss = m(x[:, :-1], x[:, 1:])
    loss.backward()
    gw = m.blocks[0].mlp.proj_new_ws[0].grad
    check("M4-B 轴① 新单元梯度通路活着", gw is not None and float(gw.abs().sum()) > 0,
          f"|g|={float(gw.abs().sum()):.3e}" if gw is not None else "无梯度")
    m.eval()

    # 轴②：attn-v 路径（delta 自动向下取整到 n_heads 倍数）
    rec2 = widen_attn_v(m, 1, 7, seed=6)  # 7 不是 4 的倍数 → 取整为 4
    with torch.no_grad():
        l2, _ = m(x[:, :-1], x[:, 1:])
    check("M4-B 轴② attn-v 宽化逐位相等", torch.equal(l1, l2),
          f"max|Δ|={(l1 - l2).abs().max().item():.3e}")
    check("M4-B 轴② delta 整除取整（n_heads 整除保持）",
          rec2["delta"] == 4 and m.blocks[1].attn.bypass_width() == 4)
    check("M4-B 注意力权重不受旁路影响（免疫位成立）",
          float(m.blocks[1].attn.proj_new_ws[0].abs().sum()) == 0.0)
    # 监督审计 P0-1（变异 A2 漏抓）：轴②旁路新行的**破缺噪声必须在位**。审计实验
    # 证明：把 surgery.WidenedAttn 的噪声行清零后全套件 174/174 依然绿——未来重构
    # 悄悄删掉这行噪声，自生长会静默退化成"精确但无用"（新通道梯度逐位相同永不
    # 分化），测试网防不住。本条补上轴②在位断言；对偶现状：轴①噪声在位已由上方
    # "移植体就位"断言钉死，轴③噪声由下方"副本分化"断言钉死——三轴自此各自有网。
    check("M4-B 轴② 破缺噪声在位（旁路 qkv 新行权重非零——零输出掩蔽下的分化根基）",
          float(m.blocks[1].attn.bypass_qkvs[0].weight.abs().sum()) > 0.0,
          f"|W|_1={float(m.blocks[1].attn.bypass_qkvs[0].weight.abs().sum()):.3e}")

    # 轴③：d_model 复制平铺（需要无移植体模型）——数学精确 + allclose 自检
    torch.manual_seed(12)
    m2 = ByteTransformer(cfg)
    m2.eval()
    with torch.no_grad():
        g0, _ = m2(x[:, :-1], x[:, 1:])
    m3, cfg3 = widen_d_model(m2, 2, device="cpu")
    m3.eval()
    with torch.no_grad():
        g1, _ = m3(x[:, :-1], x[:, 1:])
    rel = float((g0 - g1).abs().max() / g0.abs().max())
    check("M4-B 轴③ d_model 平铺数学精确（allclose）",
          torch.allclose(g0, g1, atol=1e-4, rtol=1e-4),
          f"相对误差={rel:.2e}（浮点 K 维物理极限，逐位不可达——见 surgery.py docstring）")
    check("M4-B 轴③ 形态倍增与 eps 同步（LN eps/k 是 LN(平铺)=平铺(LN) 的前提）",
          m3.cfg.d_model == 128
          and m3.blocks[0].mlp.fc.out_features == 512
          and abs(m3.blocks[0].ln1.eps - m2.blocks[0].ln1.eps / 2) < 1e-12)
    # 轴③对称破缺：副本行加噪后输出改变（不加则梯度逐位相同永不分化）
    with torch.no_grad():
        before, _ = m3(x[:, :-1], x[:, 1:])
    break_symmetry_dmodel(m3, 2, seed=9)
    with torch.no_grad():
        after, _ = m3(x[:, :-1], x[:, 1:])
    check("M4-B 轴③ 对称破缺噪声使副本分化", not torch.equal(before, after))
    # 轴③拒绝叠加：带移植体的模型（m 已有轴①②）必须 ValueError
    refused = False
    try:
        widen_d_model(m, 2, device="cpu")
    except ValueError:
        refused = True
    check("M4-B 轴③ 拒绝与移植体叠加（留待下一棒；控制器排程本就只用①/②轴，"
          "并无自动改轴动作——监督审计 P1-2 措辞修正）",
          refused and has_transplants(m))


def t_m4_ghost_probe():
    """M4-C 幽灵探测：LN 免疫位可测出容量增益（有限、正、可复现）；残差流位
    拒绝挂载（√(1+Δ/d) 污染测量，对质已证伪）。"""
    from dolphin.surgery import ghost_probe
    cfg = Config(d_model=64, n_layers=2, n_heads=4, block_size=64)
    torch.manual_seed(21)
    model = ByteTransformer(cfg)
    data = ("逻辑推理探测片段：若甲高于乙，乙高于丙，则甲高于丙；所有金属都导电，"
            "铁是金属，故铁导电。此段仅供幽灵探测使用。".encode() * 8)
    r1 = ghost_probe(model, 0, data, n_ghosts=6, seed=3)
    check("M4-C 免疫位（MLP 隐层）测出容量增益",
          r1["mlp_hidden"] is not None and r1["mlp_hidden"] > 0
          and math.isfinite(r1["mlp_hidden"]), f"gain={r1['mlp_hidden']:.3e}")
    check("M4-C 免疫位（attn-v）测出容量增益",
          r1["attn_v"] is not None and r1["attn_v"] > 0
          and math.isfinite(r1["attn_v"]) and len(r1["attn_v_per_head"]) == 4,
          f"gain={r1['attn_v']:.3e} per_head={[round(v, 4) for v in r1['attn_v_per_head']]}")
    r2 = ghost_probe(model, 0, data, n_ghosts=6, seed=3)
    check("M4-C 探测可复现（自带种子，不碰全局随机源）",
          abs(r1["mlp_hidden"] - r2["mlp_hidden"]) < 1e-12
          and abs(r1["attn_v"] - r2["attn_v"]) < 1e-12)
    refused = []
    for bad in ("residual_stream", "ln1_out", "ln2_out", "attn_out"):
        try:
            ghost_probe(model, 0, data, site=bad)
        except ValueError:
            refused.append(bad)
    check("M4-C 残差流位拒绝挂载（对质已证伪）", len(refused) == 4, f"拒绝={refused}")


def t_m4_shrink_born_again():
    """M4-D 收缩与 born-again：学生 config 更小且 n_heads 整除；born-again 战役
    （kd_alpha 手术验收档）+ 体检门控验收流程端到端可跑通。"""
    from dolphin.life import STATE_WITHERING, STATE_WITHER_PLAN, WITHER_KD_ALPHA, run_cycle
    from dolphin.surgery import born_again_student, shrink_config
    cfg = Config(d_model=64, n_layers=4, n_heads=4, block_size=64)
    from dolphin.vitals import Vitals
    v = Vitals()
    s_cfg, info = shrink_config(cfg, v, target=0.5)
    check("M4-D shrink_config 更小", s_cfg.n_layers == 2 and s_cfg.d_model == 32,
          f"{info['d_model']} {info['n_layers']}")
    check("M4-D n_heads 整除保持", s_cfg.d_model % s_cfg.n_heads == 0
          and info["n_heads_divisible"])
    student, opt, _ = born_again_student(cfg, v, 0.5, "cpu", seed=1)
    n_teach = sum(p.numel() for p in ByteTransformer(cfg).parameters())
    n_stud = sum(p.numel() for p in student.parameters())
    check("M4-D born-again 学生从头初始化且更小", n_stud < n_teach,
          f"{n_stud / 1e6:.3f}M < {n_teach / 1e6:.3f}M")

    with _probe_fixture("dolphin_m4d_") as probe:
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=4, block_size=256),
                         probe_path=probe, device="cpu")
        for i in range(12):
            d.learn(f"凋零战役验证记录第{i}条，内容足够长以稳定通过选拔训练。{i}", source="test")
        ctl = d.life_ctl
        # 直接排程 born-again 战役（绕过平台期阶梯——阶梯由 t_m4_state_machine 验）
        ctl.plan = {"kind": "wither", "mode": "born_again"}
        ctl.state = STATE_WITHER_PLAN
        r1 = run_cycle(d, steps=3, verbose=False)
        # 战役开打是周期内事实（学生换装 + 手术验收 kd 档）；周期末门控二选一：
        # 通过 → 学生上岗（awake=学生），未过 → 学生保留续训（sleeping=学生）。
        # 不赌门控抛硬币——3 步学生与教师的 probe 差在 ±1e-4 量级，换班与否随
        # 权重初始化摇摆，两种结局都是合法战役状态。
        check("M4-D born-again 战役开打（学生换装睡脑）",
              r1.get("m4_wither_start", {}).get("mode") == "born_again"
              and ((r1.get("swapped") and d.awake().model.cfg.d_model == 44
                    and d.awake().model.cfg.n_layers == 1)
                   or (ctl.state == STATE_WITHERING
                       and d.sleeping().model.cfg.d_model == 44
                       and d.sleeping().model.cfg.n_layers == 1)),
              f"swapped={r1.get('swapped')} state={ctl.state} "
              f"sleeping={d.sleeping().model.cfg.d_model}x{d.sleeping().model.cfg.n_layers}")
        check("M4-D 手术验收周期 kd_alpha 提到定标档",
              r1.get("m4_kd_alpha") == WITHER_KD_ALPHA,
              f"kd_alpha={r1.get('m4_kd_alpha')}（0.7–0.8 带中值）")
        for i in range(12):  # r1 已把缓冲清空：不补喂则 r2 空选拔早退、体检键缺席
            d.learn(f"凋零战役续训记录第{i}条，内容足够长以稳定通过选拔训练。{i}", source="test")
        r2 = run_cycle(d, steps=3, verbose=False)  # 验收 or 学生保留续训
        ok = (ctl.state == STATE_WITHERING and r2.get("m4_wither_keep") is not None) or \
             (ctl.state == "NORMAL" and "swapped" in r2)
        check("M4-D 体检门控验收流程跑通（验收换班 or 学生保留续训）", ok,
              f"state={ctl.state} swapped={r2.get('swapped')} keep={r2.get('m4_wither_keep')}")
        check("M4-D 门控判决始终在场（体检未绕过）",
              "probe_new" in r1 and "probe_new" in r2 and "gate_margin" in r2)

def t_m4_life_equivalence():
    """M4-E M0 等价性（迁移安全网）：同种子同输入下 life.run_cycle 与
    sleep.run_cycle 行为一致——报告键、权重逐位、记忆库、缓冲、换班。"""
    import random as _random
    from dolphin.life import run_cycle as life_run
    from dolphin.sleep import run_cycle as sleep_run

    with _probe_fixture("dolphin_m4e_") as probe:
        outs = []
        for impl in (sleep_run, life_run):
            torch.manual_seed(4242)
            d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=256),
                             probe_path=probe, device="cpu")
            d.rng = _random.Random(1234)  # 生产为 SystemRandom；测试注入种子化随机源
            for i in range(12):
                d.learn(f"等价性验证记录第{i}条，内容足够长以稳定通过选拔训练。{i}",
                        source="test")
            r1 = impl(d, steps=5, verbose=False)
            r2 = impl(d, steps=5, verbose=False)  # 第二周期（账本/状态跨周期累积后仍等价）
            outs.append((d, r1, r2))
        (d0, a1, a2), (d1, b1, b2) = outs
        keys = ("selected", "residue", "passed", "swapped", "probe_old", "probe_new",
                "gate_margin", "lr", "cycle")
        same1 = all(a1.get(k) == b1.get(k) for k in keys)
        same2 = all(a2.get(k) == b2.get(k) for k in keys)
        check("M4-E 周期报告逐键一致（两周期）", same1 and same2,
              f"c1={ {k: (a1.get(k), b1.get(k)) for k in keys if a1.get(k) != b1.get(k)} }"
              f" c2 diff={[k for k in keys if a2.get(k) != b2.get(k)]}")
        w_same = all(torch.equal(pa, pb)
                     for h0, h2 in ((d0.h[0], d1.h[0]), (d0.h[1], d1.h[1]))
                     for pa, pb in zip(h0.model.parameters(), h2.model.parameters()))
        check("M4-E 双半球权重逐位一致（M4 钩子是纯观察者）", w_same)
        m_same = ([(e.text, e.score, e.hits, e.cycle, e.kind) for e in d0.memory.entries]
                  == [(e.text, e.score, e.hits, e.cycle, e.kind) for e in d1.memory.entries])
        check("M4-E 记忆库（滞留+笔记）逐条一致", m_same,
              f"{len(d0.memory.entries)} vs {len(d1.memory.entries)} 条")
        check("M4-E 缓冲/醒脑指针/预算一致",
              len(d0.buffer.items) == len(d1.buffer.items)
              and d0.awake_idx == d1.awake_idx
              and abs(d0.budget - d1.budget) < 1e-12
              and abs(d0.buffer.sleep_threshold - d1.buffer.sleep_threshold) < 1e-12)


def t_m4_redo_recycle():
    """M4-F ReDo 回收：休眠通道被重置（输入行重随机、输出列清零）、账本重置、
    probation 生效（观察期内豁免休眠判决）。"""
    import random as _random
    from dolphin.life import run_cycle  # noqa: F401  集成段用（缺导入=前任遗留 NameError）
    from dolphin.vitals import PROBATION_CYCLES, SiteLedger
    with _probe_fixture("dolphin_m4f_") as probe:
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=4, block_size=256),
                         probe_path=probe, device="cpu")
        for i in range(10):
            d.learn(f"回收验证记录第{i}条，内容足够长。{i}", source="test")
        ctl = d.life_ctl
        h = d.sleeping()
        v = ctl.vitals_for(d, h.name)
        C = h.model.blocks[0].mlp.fc.out_features
        led = SiteLedger("b0.mlp_hidden", C)
        for c in (0, 1, 2):  # 伪造连续 2 周期 bottom-5% 的休眠通道
            led.dorm_cycles[c] = 2
            led.stable_act[c] = 0.0
        v.sites["b0.mlp_hidden"] = led
        before_fc = h.model.blocks[0].mlp.fc.weight[0].clone()
        before_proj = h.model.blocks[0].mlp.proj.weight[:, 0].clone()
        # 单元级：直接调回收（权重断言与训练解耦）
        res = ctl._redo(d, h, v)
        picked = res["recycled"].get("b0.mlp_hidden", [])
        check("M4-F 确认休眠通道进入回收名单", 0 in picked and len(picked) == 1,
              f"picked={picked}（单周期上限 REDO_MAX_FRAC）")
        check("M4-F 输入行被重随机（不再是原权重）",
              not torch.equal(before_fc, h.model.blocks[0].mlp.fc.weight[0]))
        check("M4-F 输出列清零（ReDo 重生单元从零贡献起步）",
              float(h.model.blocks[0].mlp.proj.weight[:, 0].abs().sum()) == 0.0
              and float(before_proj.abs().sum()) > 0)
        led2 = v.sites["b0.mlp_hidden"]
        check("M4-F 账本重置 + probation 生效",
              led2.probation[0] == PROBATION_CYCLES and led2.is_new[0]
              and led2.dorm_cycles[0] == 0 and 0 not in led2.dormant())
        # probation 豁免：观察期内即使激活垫底也不判休眠、不重复回收
        led2.observe([0.001] * C, [0.001] * C)
        led2.end_cycle(10)
        check("M4-F 观察期豁免休眠判决（防回收抖动）", 0 not in led2.dormant())
        # 集成：run_cycle 的 REM 相位会真的执行回收钩子。
        # 注入种子化 rng（生产是 SystemRandom）+ 加长记录：否则选拔/回放流长
        # 非确定，可能撞上"样本过短"早退、REM 相位整段不执行（实测 1/4 概率红）
        d2 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=4, block_size=256),
                          probe_path=probe, device="cpu")
        d2.rng = _random.Random(4321)
        for i in range(10):
            d2.learn(f"回收集成验证记录第{i}条，内容足够长。{i}" * 6, source="test")
        v2 = d2.life_ctl.vitals_for(d2, d2.sleeping().name)
        led3 = SiteLedger("b1.mlp_hidden", d2.sleeping().model.blocks[1].mlp.fc.out_features)
        led3.dorm_cycles[5] = 2
        led3.stable_act[5] = 0.0
        v2.sites["b1.mlp_hidden"] = led3
        r = run_cycle(d2, steps=2, verbose=False)
        check("M4-F REM 相位集成：回收钩子在周期内发生",
              "b1.mlp_hidden" in r.get("m4_redo", {}).get("recycled", {}),
              f"recycled={list(r.get('m4_redo', {}).get('recycled', {}))}")


def t_m4_anchor_persistence():
    """M4-G 锚与形态持久化：save/load 后历史最优 probe 锚不丢；手术后形态与
    营养账本随档复原（规格 C：锚=历史最优 probe，持久化进 save/load）。"""
    from dolphin.life import run_cycle  # noqa: F401  产生真实账本用
    from dolphin.surgery import morphology_of, rebuild_optimizer, widen_mlp
    from dolphin.vitals import SiteLedger
    with _probe_fixture("dolphin_m4g_") as probe:
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=4, block_size=256),
                         probe_path=probe, device="cpu")
        for i in range(10):
            d.learn(f"锚持久化验证记录第{i}条，内容足够长。{i}", source="test")
        ctl = d.life_ctl
        r = run_cycle(d, steps=3, verbose=False)  # 产生真实账本（可能换班）
        # 人工注入确定值便于断言（真实锚由 post_exam 更新，已在 r 里验证机制）
        ctl.anchor_best = 5.4321
        ctl.plateau = 2
        v = ctl.vitals_for(d, d.sleeping().name)
        v.sites["b0.mlp_hidden"] = SiteLedger("b0.mlp_hidden", 256)
        v.sites["b0.mlp_hidden"].stable_act = [0.5] * 256
        widen_mlp(d.sleeping().model, 1, 8, seed=2)  # 手术：形态变化
        # 真实手术流程含优化器重建（规格 B）——缺了它，save 的 opt_states 参数组
        # 与移植体参数数目不符，load 时 opt.load_state_dict 必炸（前任漏步）
        _h = d.sleeping()
        _h.opt, _ = rebuild_optimizer(_h.model, _h.opt, _h.model,
                                      _h.opt.param_groups[0]["lr"])
        ctl.morphology[d.sleeping().name] = morphology_of(d.sleeping().model)
        path = os.path.join(tempfile.gettempdir(), "dolphin_m4g_state.pt")
        d.save(path)
        d2 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=4, block_size=256),
                          probe_path=probe, device="cpu")
        d2.load(path)
        os.remove(path)
        check("M4-G 历史最优 probe 锚随档不丢（规格 C）",
              d2.life_ctl.anchor_best == 5.4321 and d2.life_ctl.plateau == 2,
              f"anchor={d2.life_ctl.anchor_best}")
        check("M4-G 手术后形态随档复原（宽化脑跨重启）",
              morphology_of(d2.sleeping().model) == ctl.morphology[d.sleeping().name]
              and d2.sleeping().model.blocks[1].mlp.bypass_width() == 8,
              f"morph={morphology_of(d2.sleeping().model)}")
        check("M4-G 营养账本随档复原",
              d2.life_ctl.vitals.get(d.sleeping().name) is not None
              and d2.life_ctl.vitals[d.sleeping().name].sites["b0.mlp_hidden"].stable_act[0] == 0.5)
        w_same = all(torch.equal(pa, pb)
                     for h0, h2 in ((d.h[0], d2.h[0]), (d.h[1], d2.h[1]))
                     for pa, pb in zip(h0.model.parameters(), h2.model.parameters()))
        check("M4-G 权重逐位回环（含移植体参数）", w_same)


def t_m4_state_machine():
    """M4-H 状态机与钩子：GROW 战役（执行→验收→NORMAL / 回滚→形态还原）、
    WITHER 战役（keep 保留标志→放弃还原 / born-again 一锤定音）、睡眠债守卫、
    显存预算推迟、固定干预次序阶梯、总开关（L11 人工安全阀）。"""
    from dolphin import life as life_mod
    from dolphin.life import (STATE_GROWN, STATE_NORMAL, STATE_WITHERING,
                              STATE_WITHER_PLAN, LifeController, run_cycle)
    from dolphin.surgery import MEM_BUDGET, mem_budget_ok, widen_mlp
    with _probe_fixture("dolphin_m4h_") as probe:
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=4, block_size=256),
                         probe_path=probe, device="cpu")
        for i in range(10):
            d.learn(f"状态机验证记录第{i}条，内容足够长。{i}", source="test")
        ctl = d.life_ctl
        h = d.sleeping()

        # —— GROW 成功：GROWN → 体检通过 → swap + NORMAL + 锚更新
        ctl.state = STATE_GROWN
        ctl.campaign = {"kind": "grow", "gen": 1, "snap": ctl._snapshot(d, h)}
        rep = {"probe_new": 5.0, "gate_margin": 0.05}
        check("M4-H GROW 验收通过 → 换班",
              ctl.post_exam(d, rep, True) == "swap" and ctl.state == STATE_NORMAL
              and ctl.anchor_best == 5.0 and ctl.campaign is None)

        # —— GROW 失败：GROWN → ROLLBACK → 形态还原 + 与醒脑同源
        snap = ctl._snapshot(d, h)
        widen_mlp(h.model, 0, 8, seed=3)  # 模拟已手术
        ctl.state = STATE_GROWN
        ctl.campaign = {"kind": "grow", "gen": 1, "snap": snap}
        rep = {"probe_new": 6.0, "gate_margin": -0.01}
        check("M4-H GROW 验收失败 → 回滚指令",
              ctl.post_exam(d, rep, False) == "rollback" and ctl.state == STATE_NORMAL)
        check("M4-H 回滚后形态还原（移植体消失）",
              h.model.blocks[0].mlp.fc.out_features == 4 * 64
              and not hasattr(h.model.blocks[0].mlp, "bypass_fcs"))
        check("M4-H 回滚后与醒脑同源（M0 回滚语义）",
              all(torch.equal(pa, pb) for pa, pb in
                  zip(h.model.parameters(), d.awake().model.parameters())))

        # —— WITHER decay：失败不回滚（学生保留标志）→ 分代耗尽放弃还原
        snap2 = ctl._snapshot(d, h)
        ctl.state = STATE_WITHERING
        ctl.campaign = {"kind": "wither", "mode": "decay", "gen": 1, "max_gen": 3,
                        "snap": snap2, "targets": {}}
        rep = {"probe_new": 6.0, "gate_margin": -0.01}
        ctl.consec_rollback = 0  # 基线清零（前面 GROW 回滚已合法 +1）：本段只验 keep/放弃的增量归属
        check("M4-H WITHER 失败 → 学生保留（不回滚，规格 C）",
              ctl.post_exam(d, rep, False) == "keep" and ctl.state == STATE_WITHERING
              and ctl.campaign["gen"] == 2)
        # 监督审计 P1-1c：keep 是"学生保留续训"，不是回滚——误递增会让
        # ROLLBACK_GUARD 把正常的多周期战役读成"脑在挣扎"
        check("M4-H keep 不递增 consec_rollback（P1-1c：学生保留≠挣扎回滚）",
              ctl.consec_rollback == 0, f"consec_rollback={ctl.consec_rollback}")
        ctl.post_exam(d, rep, False)   # gen 3
        check("M4-H WITHER 分代耗尽 → 放弃并还原",
              ctl.post_exam(d, rep, False) == "rollback" and ctl.state == STATE_NORMAL
              and ctl.campaign is None)
        check("M4-H 放弃凋零战役计入一次回滚（与 keep 区分，P1-1c 对偶）",
              ctl.consec_rollback == 1, f"consec_rollback={ctl.consec_rollback}")
        # —— WITHER born-again：学生保留（规格 C 多周期分代——从头初始化的学生
        # 不可能一个周期内赢过教师，"失败即放弃"会让 born-again 永不收敛）；
        # 分代耗尽同样放弃还原
        ctl.state = STATE_WITHERING
        ctl.campaign = {"kind": "wither", "mode": "born_again", "gen": 1,
                        "max_gen": 3, "snap": snap2}
        check("M4-H born-again 失败 → 学生保留（规格 C，与 decay 同一保留标志）",
              ctl.post_exam(d, rep, False) == "keep" and ctl.state == STATE_WITHERING)
        check("M4-H born-again keep 同样不计入 consec_rollback（P1-1c）",
              ctl.consec_rollback == 1, f"consec_rollback={ctl.consec_rollback}")
        ctl.post_exam(d, rep, False)   # gen 3
        check("M4-H born-again 分代耗尽 → 放弃并还原",
              ctl.post_exam(d, rep, False) == "rollback" and ctl.state == STATE_NORMAL
              and ctl.campaign is None)
        check("M4-H born-again 放弃计入回滚（keep/放弃增量归属对偶闭环，P1-1c）",
              ctl.consec_rollback == 2, f"consec_rollback={ctl.consec_rollback}")

        # —— 守卫：睡眠债（Bellesi 2017）与冷却
        d._since_sleep = d.target_interval * (life_mod.SLEEP_DEBT_GUARD + 1)
        ok, why = ctl.surgery_allowed(d)
        check("M4-H 睡眠债守卫禁手术（Bellesi 2017）", not ok and "睡眠债" in why, why)
        d._since_sleep = 0
        ctl.cooldown = 2
        ok, why = ctl.surgery_allowed(d)
        check("M4-H 冷却期禁手术", not ok and "冷却" in why, why)
        ctl.cooldown = 0

        # —— 显存预算：超 1.9GB 推迟（规格 B；WDDM 倒页教训）
        ok, why = mem_budget_ok("cuda", 10 ** 9, reserved=MEM_BUDGET - 1024)
        check("M4-H 显存预算超 1.9GB → 拒绝变宽", not ok, why)
        ok, why = mem_budget_ok("cuda", 10 ** 6, reserved=0)
        check("M4-H 预算内放行", ok, why)

        # 第三阶梯 + 伪造幽灵增益 → GROW_PLAN（mlp 轴）
        ctl2 = LifeController()
        d.life_ctl = ctl2
        ctl2.plateau = life_mod.PLATEAU_CYCLES * 2  # 第二阶梯
        rep = {}
        ctl2._schedule(d, rep, ctl2.vitals_for(d, d.sleeping().name), b"")
        check("M4-H 第二阶梯先 LR 退火（不许跳级）",
              ctl2.anneal_pending is True and ctl2.plan is None
              and ctl2.state == STATE_NORMAL, f"rep={ {k: rep[k] for k in rep if k.startswith('m4')} }")
        # 第三阶梯 + 伪造幽灵增益 → GROW_PLAN（mlp 轴）——ghost_scan 测试桩（用后还原）
        ctl2.plateau = life_mod.PLATEAU_CYCLES * 3
        orig_scan = life_mod.ghost_scan
        life_mod.ghost_scan = lambda *a, **k: {(0, "mlp_hidden"): 0.01}
        try:
            rep = {}
            ctl2._schedule(d, rep, ctl2.vitals_for(d, d.sleeping().name),
                           b"x" * (d.cfg.block_size + 8))
            check("M4-H 第三阶梯排程手术：幽灵有增益 → GROW_PLAN（mlp 轴）",
                  ctl2.state == "GROW_PLAN" and ctl2.plan["axis"] == "mlp"
                  and ctl2.plan["layer"] == 0 and ctl2.plan["delta"] >= 8,
                  f"plan={ctl2.plan}")
        finally:
            life_mod.ghost_scan = orig_scan
        # 幽灵无增益 + 高休眠占比 → born-again 凋零排程。
        # ghost_scan 用测试桩钉死"无增益"前提——真扫描在随机初始化权重上会偶尔
        # 测出 >GHOST_GAIN_MIN 的增益，把排程抢到 grow 轴（检查的是凋零分支）
        ctl3 = LifeController()
        d.life_ctl = ctl3
        ctl3.plateau = life_mod.PLATEAU_CYCLES * 3
        from dolphin.vitals import SiteLedger
        C = d.sleeping().model.blocks[0].mlp.fc.out_features
        led = SiteLedger("b0.mlp_hidden", C)
        for c in range(int(0.3 * C)):
            led.dorm_cycles[c] = 2
        ctl3.vitals_for(d, d.sleeping().name).sites["b0.mlp_hidden"] = led
        orig_scan = life_mod.ghost_scan
        life_mod.ghost_scan = lambda *a, **k: {}
        try:
            rep = {}
            ctl3._schedule(d, rep, ctl3.vitals_for(d, d.sleeping().name),
                           b"x" * (d.cfg.block_size + 8))
            check("M4-H 幽灵无增益+高休眠占比 → born-again 凋零排程",
                  ctl3.state == "WITHER_PLAN" and ctl3.plan["mode"] == "born_again",
                  f"rep={ {k: rep[k] for k in rep if k.startswith('m4')} }")
        finally:
            life_mod.ghost_scan = orig_scan

        # —— 监督审计 P1-1b：WITHER 执行前守卫复查（与 _execute_grow 对称）：
        # 排程后守卫恶化 → 推迟一周期且计划保留；守卫解除 → 同计划正常执行
        ctl4 = LifeController()
        ctl4.plan = {"kind": "wither", "mode": "decay", "targets": {}}
        ctl4.state = STATE_WITHER_PLAN
        ctl4.cooldown = 2  # 制造守卫恶化：手术冷却中
        rep4 = {}
        ctl4.pre_train(d, rep4, b"")
        check("M4-H WITHER 执行前守卫复查：恶化 → 推迟且计划保留（与 GROW 对称，P1-1b）",
              ctl4.state == STATE_WITHER_PLAN and ctl4.plan is not None
              and "m4_defer" in rep4,
              f"state={ctl4.state} defer={rep4.get('m4_defer')}")
        ctl4.cooldown = 0  # 守卫解除
        rep4 = {}
        ctl4.pre_train(d, rep4, b"")
        check("M4-H 守卫解除后 WITHER 计划正常执行（战役开打）",
              ctl4.state == STATE_WITHERING and ctl4.campaign is not None
              and "m4_wither_start" in rep4,
              f"state={ctl4.state} start={rep4.get('m4_wither_start')}")

        # —— L11 人工总开关：life_enabled=False 时 M4 钩子全灭（纯 M0 语义）
        d.life_enabled = False
        r = run_cycle(d, steps=2, verbose=False)
        check("M4-H 总开关关闭：M4 钩子零输出（人工安全阀，律 L11）",
              not any(k.startswith("m4_") for k in r),
              f"m4 keys={[k for k in r if k.startswith('m4_')]}")
        d.life_enabled = True

        # —— 监督审计 P0-2：手术幅度必须按睡脑**真实**脑形（d.sleeping().model.cfg）
        # 计算，不能用基座 d.cfg——born-again 学生脑（morphology 含 shrink 记录）上
        # 基座口径会把名义 5% 排成 7%+。学生脑上换算回实际通道数的占比偏差 ≤ ε。
        from dolphin.surgery import born_again_student, morphology_of
        d5 = make_dolphin(cfg=Config(d_model=256, n_layers=2, n_heads=4, block_size=256),
                          probe_path=probe, device="cpu")
        h5 = d5.sleeping()
        ctl5 = d5.life_ctl
        stud, _opt5, _info5 = born_again_student(h5.model.cfg, None, 0.7, "cpu", seed=7)
        h5.model = stud  # 模拟 born-again 学生换装后的睡脑
        C_real = stud.cfg.d_model  # = 4×round(64×0.7)=180；基座 256
        ctl5.morphology[h5.name] = {**morphology_of(stud),
                                    "shrink": {"d_model": stud.cfg.d_model,
                                               "n_layers": stud.cfg.n_layers}}
        check("M4-H P0-2 场景前提：学生脑换装且形态含 shrink 记录",
              d5.sleeping().model.cfg.d_model == C_real
              and ctl5.morphology[h5.name]["shrink"]["d_model"] == C_real
              and d5.cfg.d_model == 256,
              f"真实 d_model={C_real} 基座={d5.cfg.d_model}")
        for axis, base in (("mlp", 4 * C_real), ("attn_v", C_real)):
            delta = ctl5._grow_delta(d5, axis)
            frac = delta / base
            base_stale = max(8, round(life_mod.GROW_DELTA_FRAC
                                      * (4 * d5.cfg.d_model if axis == "mlp"
                                         else d5.cfg.d_model)))  # 基座口径（旧缺陷）
            check(f"M4-H P0-2 学生脑上 {axis} 轴手术幅度守住名义 5%（偏差≤0.005）",
                  delta > 8  # 下限不绑定，占比检查才有意义
                  and abs(frac - life_mod.GROW_DELTA_FRAC) <= 0.005,
                  f"delta={delta} 真实宽度={base} 实际占比={frac:.4f}"
                  f"（基座口径会排 {base_stale}，占比 {base_stale / base:.4f}）")
        # _mem_ok 的 Δ参数核算同样按真实脑形通道数（基座 256 会虚报显存账）
        calls = {}
        orig_dp = life_mod.delta_params_mlp
        def _spy_dp(C, delta, _o=orig_dp):
            calls["C"] = C
            return _o(C, delta)
        life_mod.delta_params_mlp = _spy_dp
        try:
            okm, why = ctl5._mem_ok(d5, "mlp", ctl5._grow_delta(d5, "mlp"))
        finally:
            life_mod.delta_params_mlp = orig_dp
        check("M4-H P0-2 显存核算按真实脑形通道数（delta_params_mlp 收到 C_real）",
              okm and calls.get("C") == C_real,
              f"C={calls.get('C')} 真实={C_real} ok={okm}")


def t_m4_thermostat():
    """M4-T 连续结构恒温器（研究报告_自适应原理 §3；2026-10-05 呼吸实验触发链
    诊断吸收）：双信号钙公式、设定点带语义（带内持有/越带收缩/确认窗回滞）、
    乘性生长律 delta=λ_g·W·e、born-again 目标从 gap̄ 导出（废除写死 0.7）、
    五重阻尼（确认窗/限频/振荡熔断/λ_g 自适应/成熟刹车）、四条守卫（数据供给门/
    预算供给比/选址禁用探测集/容量下限非零）、需求波动跟随时间线（涨/持有/缩）、
    L5/L8 不破、随档回环、触发可达性（现实信号链在受限场景真实开刀——
    平台期阶梯 9 连击数学不可达的反面证据）。"""
    import json
    import random as _random

    from dolphin import life as life_mod
    from dolphin.experience import Experience
    from dolphin.life import (LAMBDA_G, LAMBDA_MIN, MATURITY_NARROW, LifeController,
                              OSC_FUSE_CYCLES, STATE_GROWN, STATE_GROW_PLAN,
                              STATE_NORMAL, STATE_WITHERING, STATE_WITHER_PLAN,
                              XI_HI, run_cycle)
    from dolphin.vitals import SiteLedger

    with _probe_fixture("dolphin_m4t_") as probe:
        d = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=4, block_size=256),
                         probe_path=probe, device="cpu")

        # —— 夹具：按目标 (gap, hot) 反解 stable_act 的账本注入器 ——
        # med=1.0（中带）、bottom-5% 均值=1−gap（gap 精确）、top-10% 均值=hot。
        def set_ledger(key, C, gap, hot=1.0, dorm=False):
            k = max(1, int(round(0.05 * C)))
            kp = max(1, int(round(0.10 * C)))
            led = SiteLedger(key, C)
            led.stable_act = [1.0 * hot] * kp + [1.0] * (C - k - kp) + [1.0 - gap] * k
            if dorm:
                led.dorm_cycles = [0] * (C - k) + [2] * k  # 尾部通道连续 2 周期确认
            return led

        def inject(v, gap, hot=1.0, dorm=False):
            v.sites["b0.mlp_hidden"] = set_ledger("b0.mlp_hidden", 256, gap, hot, dorm)
            v.sites["b1.mlp_hidden"] = set_ledger("b1.mlp_hidden", 256, gap, hot, dorm)
            v.sites["b0.attn_out"] = set_ledger("b0.attn_out", 64, gap, hot, dorm)
            v.sites["b1.attn_out"] = set_ledger("b1.attn_out", 64, gap, hot, dorm)

        STREAM = "恒温器测试回放流".encode() * 30  # ≥ block_size+2，供幽灵扫描

        def fresh_ctl():
            c = LifeController()
            d.life_ctl = c
            return c, c.vitals_for(d, d.sleeping().name)

        def thermo(ctl, v, rep=None):
            d.cycle += 1
            rep = {} if rep is None else rep
            ctl.thermo_cycle(d, rep, v, STREAM, sel=None)
            # run_cycle 的 LR 段每周期对手术 LR 重启窗递减（life.py 周期末）；
            # 本尺级驱动不经 run_cycle，镜像该递减以保持限频③语义真实。
            if ctl.lr_window > 0:
                ctl.lr_window -= 1
            return rep

        def resolve(ctl):  # 测试桩：战役按"已验收/已放弃"收场，回 NORMAL 续演
            ctl.state = STATE_NORMAL
            ctl.campaign = None
            ctl.plan = None

        # ================= T1 利用率指数公式（双信号"钙"的测量层） =================
        led = set_ledger("t.mlp_hidden", 256, 0.30, hot=1.5)
        gap, hot = led.utilization_indices(p=0.10)
        check("M4-T 利用率指数公式（gap=(med−tail)/med、hot=top10%/med）",
              abs(gap - 0.30) < 1e-9 and abs(hot - 1.5) < 1e-9,
              f"gap={gap:.6f} hot={hot:.6f}")
        g0, h0 = SiteLedger("t.zero", 8).utilization_indices()
        check("M4-T 冷账本无信号（gap=0, hot=1——既不过剩也不过热）",
              g0 == 0.0 and h0 == 1.0)

        # ================= T2 设定点带语义（带内持有/确认窗/越带收缩） =================
        ctl, v = fresh_ctl()
        inject(v, 0.10)  # 带内（XI_HI=0.20 + 死区之下）
        rep = thermo(ctl, v)
        check("M4-T 带内 → 持有（设定点带语义）",
              rep["body"]["action"]["kind"] == "hold" and ctl.plan is None
              and ctl.state == STATE_NORMAL, rep["body"]["action"])
        inject(v, 0.35, dorm=True)  # 越带第 1 窗口
        thermo(ctl, v)
        check("M4-T 越带 1 窗口不动作（Schmitt 确认窗，阻尼②）",
              ctl.plan is None and ctl.state == STATE_NORMAL)
        inject(v, 0.35, dorm=True)  # 连续第 2 窗口
        rep = thermo(ctl, v)
        check("M4-T 连续 2 窗口越带 → 收缩计划（衰减，目标=确认休眠名单）",
              ctl.state == STATE_WITHER_PLAN and ctl.plan
              and ctl.plan["mode"] == "decay" and len(ctl.plan["targets"]) > 0,
              f"plan={ctl.plan and ctl.plan['mode']}")

        # ================= T3 born-again 目标从 gap̄ 导出 + 连环缩阻尼 =================
        ctl, v = fresh_ctl()
        inject(v, 0.60, dorm=True)  # gap̄=0.6 → T=1−0.6×0.6=0.64（非写死 0.7）
        thermo(ctl, v)
        rep = thermo(ctl, v)
        check("M4-T 结构级错配 → born-again，目标 T 从 gap̄ 导出（0.64≠0.7）",
              ctl.plan and ctl.plan["mode"] == "born_again"
              and abs(ctl.plan["target"] - 0.64) < 1e-6,
              f"target={ctl.plan and ctl.plan.get('target')}")
        ctl.last_born_again = d.cycle  # 模拟刚 born-again 过（3 周期内再缩）
        resolve(ctl)
        inject(v, 0.60, dorm=True)
        thermo(ctl, v)
        thermo(ctl, v)
        check("M4-T 连环缩阻尼：3 周期内再 born-again → 收缩系数减半（T=0.82）",
              ctl.plan and ctl.plan["mode"] == "born_again"
              and abs(ctl.plan["target"] - 0.82) < 1e-6,
              f"target={ctl.plan and ctl.plan.get('target')}")

        # ================= T4 乘性生长律 + ghost 需求门槛 + 确认窗（公式级） =================
        ctl, v = fresh_ctl()
        ghost_now = {(0, "mlp_hidden"): 1e-4, (0, "attn_v"): 1e-4,
                     (1, "mlp_hidden"): 1e-4, (1, "attn_v"): 1e-4}
        orig_scan = life_mod.ghost_scan
        life_mod.ghost_scan = lambda model, data, **kw: dict(ghost_now)
        try:
            inject(v, 0.10, hot=1.0)  # 带内：收缩不抢跑
            thermo(ctl, v)
            inject(v, 0.10, hot=1.5)  # 需求上升：b0.mlp_hidden 过热
            thermo(ctl, v)            # c4：过热第 1 窗
            inject(v, 0.10, hot=1.5)
            rep = thermo(ctl, v)      # c5：过热确认 → 首次幽灵扫描（基线）
            check("M4-T 幽灵基线未建立 → 不排程（自身基线门槛，防漂移误判）",
                  ctl.plan is None and ctl.state == STATE_NORMAL,
                  rep["body"]["action"].get("reason"))
            ctl.fresh_since_surgery = 10 ** 7  # 守卫④：充足证据流（预算放行）
            inject(v, 0.10, hot=1.5)
            ghost_now[(0, "mlp_hidden")] = 2e-4  # 需求跳变（对自身基线 2×）
            thermo(ctl, v)            # c6：扫描 #2（确认窗 1/2）
            inject(v, 0.10, hot=1.5)
            rep = thermo(ctl, v)      # c7：扫描 #3 → 确认窗满 → 排程
            W = 4 * 64  # b0.mlp_hidden 真实宽度（无移植体基座）
            exp_delta = max(8, round(ctl.lambda_g_eff() * W
                                     * max(0.0, min(2.0, 2.0 - 1.0))))
            check("M4-T ghost 需求 2× 自身基线 + 过热确认 → GROW_PLAN（mlp 轴）",
                  ctl.state == STATE_GROW_PLAN and ctl.plan
                  and ctl.plan["axis"] == "mlp" and ctl.plan["layer"] == 0,
                  f"plan={ctl.plan}")
            check("M4-T 乘性生长律 delta=λ_g·W·e（公式值断言，研究报告 §3.3）",
                  ctl.plan and ctl.plan["delta"] == exp_delta,
                  f"delta={ctl.plan and ctl.plan['delta']} 期望 {exp_delta}"
                  f"（λ_g_eff={ctl.lambda_g_eff():.4f} W={W} e=1.0）")
            # —— 限频③闭环：生长计划刚记录，同 site 冷却期内不得再排 ——
            resolve(ctl)
            inject(v, 0.10)  # 其余轴位撤除过热，只留 b0.mlp_hidden 过热
            v.sites["b0.mlp_hidden"] = set_ledger("b0.mlp_hidden", 256, 0.10, 1.5)
            rep = thermo(ctl, v)      # c8：同 site 间隔未满
            check("M4-T 同 site 动作间隔 ≥6 周期（限频③，冷却让位收缩前先挡生长）",
                  ctl.plan is None and "限频" == rep["body"]["action"].get("class"),
                  rep["body"]["action"])
        finally:
            life_mod.ghost_scan = orig_scan

        # ================= T5 需求波动跟随：涨 / 持有 / 缩 时间线（核心证据） =================
        ctl, v = fresh_ctl()
        kinds, params_t = [], []
        params = lambda: sum(p.numel() for p in d.sleeping().model.parameters())
        snap_base = ctl._snapshot(d, d.sleeping())  # 收尾还原点（born-again 异构兜底）
        ghost_now = {(0, "mlp_hidden"): 1e-4, (0, "attn_v"): 1e-4,
                     (1, "mlp_hidden"): 1e-4, (1, "attn_v"): 1e-4}
        orig_scan = life_mod.ghost_scan
        life_mod.ghost_scan = lambda model, data, **kw: dict(ghost_now)
        try:
            inject(v, 0.10)                       # 需求常态
            thermo(ctl, v); kinds.append("hold")  # 周期 1
            thermo(ctl, v); kinds.append("hold")  # 周期 2（持有）
            P0 = params()
            ctl.fresh_since_surgery = 10 ** 7     # 新切片带来的充足证据流（守卫④放行）
            inject(v, 0.10, hot=1.5)              # 需求上升（新数据切片）
            thermo(ctl, v)                        # 周期 3：过热第 1 窗（未确认，无扫描）
            thermo(ctl, v)                        # 周期 4：过热确认 → 首次扫描=基线（低）
            ghost_now[(0, "mlp_hidden")] = 2e-4   # 需求跳变（对自身基线 2×）
            thermo(ctl, v)                        # 周期 5：扫描 #2（确认窗 1/2）
            rep = thermo(ctl, v)                  # 周期 6：扫描 #3 → 确认窗满 → 排程
            kinds += ["hold", "hold"]
            kinds.append(rep["body"]["action"]["kind"])
            check("M4-T 需求上升 → 恒温器排出生长计划",
                  ctl.state == STATE_GROW_PLAN and ctl.plan["kind"] == "grow",
                  f"plan={ctl.plan}")
            pre = params()
            ctl.pre_train(d, {}, b"")             # 执行手术（真实 widen）
            P1 = params()
            params_t += [P0, P0, P0, P0, pre, P1]
            check("M4-T 需求上升落地：生长手术后参数总量上升（涨）", P1 > P0,
                  f"P0={P0} → P1={P1}（+{P1 - P0}）")
            resolve(ctl)
            inject(v, 0.10)                       # 需求满足（回带内）
            thermo(ctl, v); kinds.append("hold")  # 持有
            thermo(ctl, v); kinds.append("hold")  # 持有
            check("M4-T 带内持有：参数总量保持不变", params() == P1,
                  f"params={params()} == P1={P1}")
            params_t += [P1, P1]
            inject(v, 0.35, dorm=True)            # 需求长期下降（过剩）
            thermo(ctl, v)
            rep = thermo(ctl, v)
            kinds.append(rep["body"]["action"]["kind"])
            check("M4-T 持续过剩 → 收缩计划（衰减）",
                  ctl.state == STATE_WITHER_PLAN and ctl.plan["mode"] == "decay")
            ctl.pre_train(d, {}, b"")             # 执行衰减战役
            resolve(ctl)
            inject(v, 0.60, dorm=True)            # 结构级错配
            thermo(ctl, v)
            thermo(ctl, v)
            check("M4-T 深度过剩 → born-again 计划（缩）",
                  ctl.plan and ctl.plan["mode"] == "born_again")
            ctl.pre_train(d, {}, b"")             # 执行换装（学生从头出生）
            P2 = params()
            params_t.append(P2)
            check("M4-T 需求下降落地：born-again 后参数总量低于基线（缩）", P2 < P0,
                  f"P2={P2} < P0={P0}")
            # 测试收尾：按出生快照还原睡脑（born-again 换装后与醒脑异构；后续
            # T9 的纯 M0 回滚路径要求同构——真实部署中该还原由验收/放弃语义承担）
            ctl._restore(d, d.sleeping(), snap_base)
            resolve(ctl)
            flips = sum(1 for a, b2 in zip(kinds, kinds[1:])
                        if a != "hold" and b2 != "hold" and a != b2)
            check("M4-T 需求波动跟随时间线：涨/持有/缩 全程有界（方向翻转 ≤4）",
                  flips <= 4 and P2 < P0 < P1,
                  f"kinds={kinds} 翻转 {flips} 次 params={params_t}")
        finally:
            life_mod.ghost_scan = orig_scan

        # ================= T6 振荡熔断（阻尼⑤） =================
        ctl, v = fresh_ctl()
        rep = {}
        ctl._record_action(["b0.mlp_hidden"], "shrink", 10, rep)
        ctl._record_action(["b0.mlp_hidden"], "grow", 12, rep)   # 反向 <6 周期：振荡①
        check("M4-T 反向操作间隔 <6 周期记振荡", len(ctl.osc_events) == 1)
        ctl._record_action(["b0.mlp_hidden"], "shrink", 14, rep)  # 振荡② → 熔断
        check("M4-T 振荡 ≥2 次/6 周期 → 该 site 熔断 + 死区放宽（阻尼⑤）",
              ctl.site_fuse.get("b0.mlp_hidden") == 14 + OSC_FUSE_CYCLES
              and ctl.deadband_scale > 1.0,
              f"fuse={ctl.site_fuse} deadband={ctl.deadband_scale}")

        # ================= T7 守卫：数据供给门 / 反刍期不动刀 =================
        ctl, v = fresh_ctl()
        inject(v, 0.35, dorm=True)
        ctl.supply_hist = [[d.cycle - 2, 0], [d.cycle - 1, 0], [d.cycle, 0]]
        rep = thermo(ctl, v)
        check("M4-T 供给门关闭（反刍期零新鲜摄入）→ 结构冻结不动刀",
              rep["body"]["action"].get("class") == "数据耗尽（供给门）"
              and ctl.plan is None and rep["body"]["supply"] == "closed",
              rep["body"]["action"])
        thermo(ctl, v)  # 第 2 窗口（越带确认已满）——供给门仍冻结
        check("M4-T 越带确认已满但供给门关闭 → 仍不动刀（守卫优先于信号）",
              ctl.plan is None and ctl.state == STATE_NORMAL)
        ctl.supply_account(d, [(1.0, Experience(999, "全新经验非反刍内容" .encode() * 20, 3.0))])
        rep = thermo(ctl, v)
        check("M4-T 新鲜经验入账 → 供给门开 → 越带信号放行为收缩计划",
              ctl.state == STATE_WITHER_PLAN and ctl.plan["mode"] == "decay"
              and rep["body"]["supply"] == "open", rep["body"]["action"])

        # ================= T7b 守卫④预算/供给比 =================
        ctl, v = fresh_ctl()
        ghost_now = {(0, "mlp_hidden"): 1e-4, (0, "attn_v"): 1e-4,
                     (1, "mlp_hidden"): 1e-4, (1, "attn_v"): 1e-4}
        orig_scan = life_mod.ghost_scan
        life_mod.ghost_scan = lambda model, data, **kw: dict(ghost_now)
        try:
            inject(v, 0.10, hot=1.5)
            thermo(ctl, v)
            thermo(ctl, v)                        # 过热确认 + 基线扫描
            ghost_now[(0, "mlp_hidden")] = 2e-4
            inject(v, 0.10, hot=1.5)
            thermo(ctl, v)
            inject(v, 0.10, hot=1.5)
            rep = thermo(ctl, v)                  # ghost 需求确认，但证据流不足
            check("M4-T 守卫④：新鲜证据不足最小步（8 通道当量）→ 拒绝生长",
                  ctl.plan is None
                  and "预算" in rep["body"]["action"].get("class", ""),
                  rep["body"]["action"])
        finally:
            life_mod.ghost_scan = orig_scan

        # ================= T7c 守卫⑤：选址禁用探测集（L8 红线） =================
        ctl, v = fresh_ctl()
        seen_streams = []
        orig_scan = life_mod.ghost_scan
        life_mod.ghost_scan = lambda model, data, **kw: (
            seen_streams.append(bytes(data)) or {(0, "mlp_hidden"): 1e-4,
                                                 (0, "attn_v"): 1e-4,
                                                 (1, "mlp_hidden"): 1e-4,
                                                 (1, "attn_v"): 1e-4})
        try:
            marker = "训练粮独有标记：恒温器选址只许看这条回放流。" .encode() * 10
            inject(v, 0.10, hot=1.5)
            ctl.fresh_since_surgery = 10 ** 7
            thermo(ctl, v)
            d.cycle += 1
            rep = {}
            ctl.thermo_cycle(d, rep, v, marker, sel=None)
            check("M4-T 幽灵扫描真实发生（方向通道活性）", len(seen_streams) >= 1)
            check("M4-T 选址禁用探测集：扫描输入=训练回放流（含训练标记），"
                  "与合成探测卷零交集（守卫⑤，L8 红线）",
                  seen_streams and marker[:60] in seen_streams[0]
                  and _PROBE_SEG not in seen_streams[0],
                  f"scan 输入 {len(seen_streams[0])}B")
        finally:
            life_mod.ghost_scan = orig_scan

        # ================= T7d 守卫⑥成熟刹车：λ_g 永不归零 + 容量下限非零 =================
        ctl, v = fresh_ctl()
        # 体型当量占比 = (d_model/64)×(n_layers/2)：26/64×1/2 ≈ 0.203 < 0.25 下限
        ctl.morph_for(d, d.sleeping().name)["shrink"] = {"d_model": 26, "n_layers": 1}
        inject(v, 0.60, dorm=True)
        thermo(ctl, v)
        rep = thermo(ctl, v)
        check("M4-T 容量下限：再缩将破 MIN_BODY_FRAC → 拒绝 born-again（防无限萎缩）",
              ctl.plan is None and "容量下限" in rep["body"]["action"].get("reason", ""),
              rep["body"]["action"])
        # 40/64×1/2 = 0.3125：可缩但被托到下限允许的 0.8（=0.25/0.3125）
        ctl.morph_for(d, d.sleeping().name)["shrink"] = {"d_model": 40, "n_layers": 1}
        rep = thermo(ctl, v)
        check("M4-T 容量下限刹车：T 被托到下限允许的 0.8（比误差要求的 0.64 缩得少）",
              ctl.plan and ctl.plan["mode"] == "born_again"
              and abs(ctl.plan["target"] - 0.8) < 1e-6,
              f"target={ctl.plan and ctl.plan.get('target')}")
        del ctl.morph_for(d, d.sleeping().name)["shrink"]  # 拆掉假记录（形态=真相，
        # 回滚测试的快照/还原要求记录与真实模型一致——真实运行中由换装语义保证）
        ctl.lambda_g = 0.015
        ctl.state = STATE_GROWN
        h = d.sleeping()
        ctl.campaign = {"kind": "grow", "gen": 1, "snap": ctl._snapshot(d, h),
                        "delta": 16}
        ctl.grown_since_surgery = 16  # R3 验真（监督审计 2026-10-05）：先真实记账
        # 再回滚——原断言在 grown=0 上空转（0−campaign→0 恒真，账目死代码不可见）
        rep = {}
        ctl.post_exam(d, rep, False)  # 生长验收失败
        check("M4-T 增益自适应：生长回滚 → λ_g×0.5 且触底 LAMBDA_MIN（永不归零）",
              ctl.lambda_g == LAMBDA_MIN and rep.get("m4_lambda_g") == LAMBDA_MIN,
              f"λ_g={ctl.lambda_g}")
        check("M4-T 供给账回冲：回滚后 grown_since_surgery 归零（守卫④账实相符）",
              ctl.grown_since_surgery == 0)
        # R3 补强：部分回冲（账面 20、本次 delta 16 → 剩 4）——钉死"按 delta 精确
        # 回冲"而非清零/不动，防修复退化成两种更简单的错法
        ctl.state = STATE_GROWN
        ctl.campaign = {"kind": "grow", "gen": 1, "snap": ctl._snapshot(d, h),
                        "delta": 16}
        ctl.grown_since_surgery = 20
        ctl.lambda_g = LAMBDA_MIN
        ctl.post_exam(d, {}, False)
        check("M4-R3 回滚账目按 campaign.delta 精确回冲（原实现先清 campaign 再读它"
              "——扣账恒 0，死代码）",
              ctl.grown_since_surgery == 4, f"grown={ctl.grown_since_surgery} 期望 4")
        ctl.maturity = 1.0
        check("M4-T 成熟刹车：m=1 时 λ_g_eff=0.5λ_g>0、死区放宽 1.5×（可塑性不归零）",
              abs(ctl.lambda_g_eff() - 0.5 * ctl.lambda_g) < 1e-12
              and abs(ctl.h_eff() - life_mod.H_SHRINK * 1.5) < 1e-12)

        # ================= T7e R2 修复验收：成熟度信号可流动 + 死区收窄可达 =================
        # 病灶（监督审计 2026-10-05）：m 的原料是"margin<EPS_PLATEAU=0.005 的平台期
        # 计数"，而呼吸实验实测真实系统 margin 恒 +0.025~0.055 ≫ 0.005 → plateau
        # 恒 0 → m≡0 → 成熟刹车与死区收窄两路结构性不可达（与被退役阶梯同型病在
        # 守卫内复发）。修复后原料 = margin 滚动分布分位（_mature_input）。
        ctl, v = fresh_ctl()
        inject(v, 0.10)
        for _ in range(16):  # 平稳改善期：margin 恒 0.04（≫ 旧 ε——旧口径视之为"永不平台"）
            ctl.post_exam(d, {"probe_new": 5.0, "gate_margin": 0.04}, True)
        check("M4-R2 平稳期（margin 恒 0.04）→ 成熟输入=0、m≈0、旧口径 plateau 仍恒 0（对照）",
              ctl._mature_input() == 0.0 and ctl.maturity < 0.05 and ctl.plateau == 0,
              f"input={ctl._mature_input()} m={ctl.maturity:.3f} plateau={ctl.plateau}")
        for _ in range(4):   # 改善率下台阶：0.02 仍 ≫ 旧 ε=0.005（旧口径照样视而不见）
            ctl.post_exam(d, {"probe_new": 5.0, "gate_margin": 0.02}, True)
        check("M4-R2 改善率跌破自身滚动分布 q25 → 成熟输入=1（世界信号可流动，R2）",
              ctl._mature_input() == 1.0,
              f"近窗中位 0.02 < 自身 q25 0.04（且 0.02 ≫ 旧 ε——旧口径下此信号永不存在）")
        for _ in range(8):
            rep = thermo(ctl, v)
        check("M4-R2 成熟刹车真实生效：m EMA 上行 → λ_g_eff < λ_g、死区放宽",
              ctl.maturity >= MATURITY_NARROW and ctl.lambda_g_eff() < ctl.lambda_g
              and ctl.h_eff() > life_mod.H_SHRINK,
              f"m={ctl.maturity:.3f} λ_eff={ctl.lambda_g_eff():.4f} "
              f"h_eff={ctl.h_eff():.4f}")
        ctl.no_action_streak = life_mod.DEADBAND_CAL_EVERY - 1
        rep = thermo(ctl, v)
        check("M4-R2 死区收窄路径可达（R2：原 plateau≥1 门在真实 margin 分布下结构性死）",
              rep["body"]["action"].get("deadband_narrowed", 1.0) < 1.0
              and ctl.deadband_scale < 1.0,
              f"scale={ctl.deadband_scale}")
        check("M4-R2 冷启动不刹车：margin 历史不足 MARGIN_HIST_CAP → 输入恒 0",
              LifeController()._mature_input() == 0.0)

        # ================= T7f R1 修复验收：稳态 gap̄=0.40 不触发重锤、带自动上浮 =================
        # 病灶（监督审计 2026-10-05）：XI_HI=0.20 低于 57M 真实身体实测稳态 gap̄≈0.45
        # （docstring 自引数），带不可自校准 → born_sustained 12 周期重锤在健康系统
        # 几乎必触发。修复后带 = max(XI_HI, gap̄ 滞后滚动 p90)，随系统自身稳态上浮。
        ctl, v = fresh_ctl()
        kinds = []
        for i in range(30):  # 稳态 = docstring 引用的真实身体量级 gap̄≈0.40
            inject(v, 0.40, dorm=True)
            rep = thermo(ctl, v)
            act = rep["body"]["action"]
            kinds.append((act.get("kind"), act.get("class")))
            if ctl.plan:  # 保险收场（稳态不应排出 born）
                ctl.state = STATE_NORMAL
                ctl.plan = None
                ctl.campaign = None
        check("M4-R1 稳态 gap̄=0.40 全程零 born-again（重锤不被稳态触发，R1 验收）",
              not any(c and "结构级" in str(c) for _, c in kinds),
              f"非 hold 动作 {[(k, c) for k, c in kinds if k != 'hold']}")
        check("M4-R1 设定点带自动上浮到稳态：xi_hi_eff → 0.40（> 律定下限 0.20）",
              abs(ctl._xi_hi_eff() - 0.40) < 0.02 and ctl._xi_hi_eff() > XI_HI + 0.1,
              f"xi_hi_eff={ctl._xi_hi_eff():.3f} "
              f"band_hi={ctl._xi_hi_eff() + ctl.h_eff():.3f}")
        check("M4-R1 带浮起后 site 级越带收缩同样熄火（冷启动窗后零结构动作）",
              all(k == "hold" for k, _ in kinds[10:]),
              f"周期 11-30 动作 {[k for k, _ in kinds[10:]]}")
        check("M4-R1 body setpoint 如实上报浮动带（可观测义务，L11 措辞防线）",
              abs(rep["body"]["setpoint"]["xi_hi"] - 0.40) < 0.02
              and rep["body"]["setpoint"]["xi_hi_floor"] == XI_HI,
              f"setpoint={rep['body']['setpoint']}")
        inject(v, 0.65)   # 稳态之上的真实跳变：带滞后窗不吸收
        thermo(ctl, v)
        inject(v, 0.65)
        rep = thermo(ctl, v)
        check("M4-R1 稳态之上的真实跳变仍触发 born-again（带滞后于信号——升级路不灭）",
              ctl.plan and ctl.plan["mode"] == "born_again",
              f"target={ctl.plan and ctl.plan.get('target')}")

        # ================= T8 捕获对账（标签-捕获的固化侧） =================
        for tag, act_val, expect in (("捕获成功", 0.9, round(LAMBDA_G * 1.1, 6)),
                                     ("捕获失败", 0.1, round(LAMBDA_G * 0.9, 6))):
            ctl, v = fresh_ctl()
            key = "b0.mlp_hidden.new.0"
            v.sites[key] = SiteLedger(key, 8, {"is_new": [True] * 8,
                                               "probation": [3] * 8})
            v.sites["b0.mlp_hidden"] = set_ledger("b0.mlp_hidden", 256, 0.10)
            ctl.pending_capture = {"due_cycle": d.cycle + 1, "keys": [key],
                                   "hname": d.sleeping().name}
            v.sites[key].stable_act = [act_val] * 8  # 新单元实现利用率
            rep = thermo(ctl, v)
            cap = rep.get("m4_capture") or {}
            check(f"M4-T 捕获对账（{tag}）：rate 如实入 report，λ_g 增益自适应",
                  abs(cap.get("rate", -1) - (1.0 if act_val > 0.5 else 0.0)) < 1e-9
                  and abs(ctl.lambda_g - expect) < 1e-9,
                  f"rate={cap.get('rate')} λ_g={ctl.lambda_g}")

        # ================= T9 L5/L8 不破 + L11 总开关 + body 可观测 =================
        def feed_n(n, tag):
            for i in range(n):
                d.learn(f"恒温器集成验证{tag}记录第{i}条，内容足够长以稳定通过选拔训练。{i}" * 3,
                        source="test")

        feed_n(12, "甲")
        d.life_enabled = False
        r = run_cycle(d, steps=2, verbose=False)
        check("M4-T 总开关关闭：M4 钩子与 body 全灭（纯 M0 语义，律 L11 安全阀）",
              not any(k.startswith("m4_") for k in r) and "body" not in r,
              f"keys={sorted(r)}")
        d.life_enabled = True
        feed_n(12, "乙")  # 上一周期已摘除缓冲：不补喂则早退、body/gate 键缺席
        r = run_cycle(d, steps=2, verbose=False)
        check("M4-T body 可观测字段在周期报告（d_model/参数总量/设定点/缺口/动作）",
              "body" in r and all(k in r["body"] for k in
                                  ("d_model", "params", "setpoint", "util_gap", "action")),
              f"body={r.get('body')}")
        check("M4-T 体检未绕过：gate 字段在场且换班判决与体检一致（L8）",
              "probe_new" in r and "gate_margin" in r
              and r.get("swapped") == r.get("passed"),
              f"passed={r.get('passed')} swapped={r.get('swapped')}")
        ctl = d.life_ctl
        ctl.plan = {"kind": "wither", "mode": "born_again"}  # 手工计划（无 target）
        ctl.state = STATE_WITHER_PLAN
        feed_n(12, "丙")
        r = run_cycle(d, steps=3, verbose=False)
        check("M4-T 手工 born-again 计划无 target → 回落 WITHER_TARGET=0.7"
              "（退役件身份），且体检门控判决照常在场",
              r.get("m4_wither_start", {}).get("target") == 0.7
              and "probe_new" in r and "gate_margin" in r
              and ((r.get("m4_wither_keep") is not None)
                   == (ctl.state == STATE_WITHERING)),
              f"start={r.get('m4_wither_start')}")

        # ================= T10 恒温器状态随档回环 =================
        ctl, v = fresh_ctl()
        ctl.lambda_g, ctl.deadband_scale, ctl.maturity = 0.033, 1.5, 0.4
        ctl.thermo_hist = {"b0.mlp_hidden": [[1, 0.2, 1.3], [2, 0.3, 1.4]]}
        ctl.ghost_hist = {"0.mlp_hidden": [[1, 1e-4], [2, 2e-4]]}
        ctl.supply_hist = [[1, 500], [2, 700]]
        ctl.seen_hashes = [11, 22, 33]
        ctl.site_last_action = {"b0.mlp_hidden": [2, "grow"]}
        st = ctl.to_state()
        json.dumps(st)  # weights_only 载荷纪律：必须可 JSON 化
        ctl2 = LifeController().from_state(st)
        check("M4-T 恒温器状态随档回环（JSON 安全 + 字段逐一复原）",
              ctl2.lambda_g == 0.033 and ctl2.deadband_scale == 1.5
              and ctl2.maturity == 0.4 and ctl2.ghost_hist == ctl.ghost_hist
              and ctl2.supply_hist == ctl.supply_hist
              and ctl2._seen == {11, 22, 33}
              and ctl2.site_last_action == ctl.site_last_action)

        # ================= T11 触发可达性（现实信号链，受限场景真实开刀） =================
        # 平台期阶梯的死因是"数学不可达"（margin 恒正，9 连击期望 ~1300 周期）。
        # 反面证据：恒温器在同量级受限场景（十来个周期、真实训练/真实账本/真实
        # 幽灵扫描/真实守卫，零信号桩）内必须至少真实排一次结构手术。
        torch.manual_seed(2026)
        d2 = make_dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=4, block_size=256),
                          probe_path=probe, device="cpu")
        d2.rng = _random.Random(11)
        ctl = d2.life_ctl
        v = ctl.vitals_for(d2, d2.sleeping().name)
        seg_a = "逻辑推理恒温器实测片段：若甲高于乙则甲高于丙，铁是金属故铁导电。"
        plans, decided, scans0 = [], 0, sum(len(x) for x in ctl.ghost_hist.values())
        for c in range(3):  # 阶段 A：基线需求（结构化中文）
            for i in range(25):
                d2.learn(seg_a + f"基线编号{c * 25 + i}。", source="reach-A")
            # maybe_sleep = 部署单线程路径（周期后睡眠债清零——与 SleepTrainer 的
            # feeding 路径同语义；直调 run_cycle 不清债，会把 Bellesi 守卫误触发）
            r = d2.maybe_sleep(force=True, steps=25, verbose=False)
            act = (r.get("body") or {}).get("action") or {}
            decided += 1 if act.get("kind") else 0
            if r.get("m4_plan"):
                plans.append(r["m4_plan"][0])
        seg_b = "arithmetic drill {i}: {i}+{i}=2i {i}*3={t} {i}/2={h} results follow"
        for c in range(6):  # 阶段 B：需求跳变（全新字节分布，~240KB/周期新鲜证据）
            for i in range(30):
                i2 = c * 30 + i
                d2.learn((seg_b.format(i=i2, t=i2 * 3, h=i2 // 2) + " ") * 90,
                         source="reach-B")
            r = d2.maybe_sleep(force=True, steps=25, verbose=False)
            act = (r.get("body") or {}).get("action") or {}
            decided += 1 if act.get("kind") else 0
            if r.get("m4_plan"):
                plans.append(r["m4_plan"][0])
        check("M4-T 触发可达性①：每个周期都有信号落地的判决（class+reason 在场）",
              decided >= 8, f"{decided} 个周期出判决")
        check("M4-T 触发可达性②：现实信号下真实排出结构手术（非阶梯数学不可达）",
              len(plans) >= 1, f"plans={plans}")
        check("M4-T 触发可达性③：方向通道真实开动（幽灵扫描基线已记账）",
              sum(len(x) for x in ctl.ghost_hist.values()) > scans0
              and len(ctl.ghost_hist) > 0, f"ghost 位点 {len(ctl.ghost_hist)} 个")
        check("M4-T 触发可达性④：恒温器全程只由内部信号驱动（无人工干预痕迹）",
              all(p in ("grow", "wither") for p in plans))


# ============ 冷层隔离总闸自检（本身也是断言） ============


def t_cold_quarantine():
    """红线断言：本套件不得改动任何生产资产（memory_cold.jsonl / probe.txt）。

    这条断言是防线③的兑现点，也是整个隔离机制唯一的「可观测出口」——
    若前两道防线（默认值改写 + 显式路径拦截）某天被误删，这里会立刻变红
    并打印测试前后的完整指纹，而不是让污染悄悄发生。
    放在最后跑：必须等所有测试都结束才有意义（中途比对会误报）。
    """
    # ① 工厂产出的冷层必须落在隔离沙箱内，且与生产路径无关
    d = make_dolphin(cfg=Config(d_model=32, n_layers=1, n_heads=2, block_size=64),
                     device="cpu", probe_path=None)
    check("隔离 工厂冷层不指向生产 memory_cold.jsonl",
          os.path.realpath(d.memory.cold_path) != _PROD_COLD
          and os.path.realpath(d.memory.cold_path).startswith(os.path.realpath(_SANDBOX)),
          d.memory.cold_path)
    # ② 默认路径（cold_path=None）也必须被改写到沙箱——这正是历史事故的漏点
    m_def = MemoryStore()
    check("隔离 默认 cold_path=None 也被改写到沙箱（历史事故漏点）",
          os.path.realpath(m_def.cold_path) != _PROD_COLD,
          m_def.cold_path)
    # ③ 显式传入生产路径也拦得住（意图不明的显式传参不能绕过隔离）
    m_exp = MemoryStore(cold_path=_PROD_COLD)
    check("隔离 显式传入生产路径同样被改写（绕过路径封死）",
          os.path.realpath(m_exp.cold_path) != _PROD_COLD, m_exp.cold_path)
    # ④ 两个实例的冷层互相隔离（不共享文件，避免跨测试串味）
    check("隔离 每实例独占冷层文件（实例间不串味）",
          m_def.cold_path != m_exp.cold_path)
    # ⑤ 对沙箱冷层做真实的逐出落盘 + flush：确认写操作确实落在沙箱、生产零改动。
    #    这一条是「隔离不是空谈」的正面证据：写入路径真的被走了一遍。
    #    cap=2（属性名是 cap，不是构造参数名 cap_entries）→ 第 3 条 add 起逐出。
    d.memory.cap = 2
    for i in range(6):
        d.memory.add(f"隔离自检条目第{i}号，触发逐出落盘以验证写入落在沙箱。{i}".encode(),
                     0.0, 0, "residue")
    d.memory.flush_cold()
    wrote = os.path.exists(d.memory.cold_path)
    check("隔离 沙箱冷层可真实落盘（写入路径走得通）", wrote, d.memory.cold_path)


def t_cold_untouched():
    """红线断言（收尾）：全部测试跑完后，生产资产指纹必须与开工时逐字节一致。

    2026-10-07 修复（审计B）：生产进程 feed.py 会在测试期间并发写入
    memory_cold.jsonl（正常冷层日志追加/重写）。把「外部进程正常写入」与
    「测试自身污染」区分开：追加方向 → 黄色警告（测试自身未污染，check 仍
    通过）；删减/重写或 probe.txt 变化 → 判红失败。

    干净 clone / 生产资产不在场时（dolphin/memory_cold.jsonl 未生成）：
    本测试无可验证对象，显式跳过而非判红——但文件一旦存在，必须继续验证
    「生产资产零触碰」（before == after），检测能力保持不变。
    """
    if not os.path.exists(_PROD_COLD):
        pytest.skip("dolphin/memory_cold.jsonl not present (production asset absent)")
    ok, warnings, detail = verify_cold_quarantine()
    for w in warnings:
        print(f"  ⚠ {w}")
    check("红线 全套测试零改动生产资产（memory_cold.jsonl / probe.txt）", ok, detail)


def t_stdin_chat():
    """无模式 stdin 对话通道（2026-10-07）：serve 即学习信号 + 打分反馈 + 异常不炸。

    用户输入经 serve() 自动 add 进 buffer（learn_total+1），与批量喂食同一条链路；
    打印回复后从 stdin 读一行打分并写 reward。任何异常都不应炸部署（返回 None）。
    本测试全部落在隔离沙箱：临时 probe + make_dolphin 隔离冷层，不碰生产资产。
    """
    import io
    from feed import chat_once

    tmp = tempfile.mkdtemp(prefix="dolphin_stdinchat_")
    try:
        probe = os.path.join(tmp, "probe.txt")
        _write_synthetic_probe(probe, 12)
        cfg = Config(d_model=64, n_layers=2, n_heads=2, block_size=256)
        d = make_dolphin(cfg=cfg, probe_path=probe, device="cpu")

        # 预热一次，让经验编号从 1 开始（满足 eid > 0 的断言）
        warm_eid = d.learn("预热记录：让经验编号从 1 开始。")
        check("stdin-chat 预热 eid 从 0 开始", warm_eid == 0, f"预热 eid={warm_eid}")
        before_total = d.learn_total

        # 用 StringIO 替换 sys.stdin：预写打分行 "+1\n"
        orig_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO("+1\n")
            result = chat_once(d, "你好，海豚")
        finally:
            sys.stdin = orig_stdin

        eid_ok = isinstance(result, tuple) and len(result) == 2
        eid = result[0] if eid_ok else None
        resp = result[1] if eid_ok else None
        check("stdin-chat 返回 (eid, resp) 且 eid 为 int > 0",
              eid_ok and isinstance(eid, int) and eid > 0,
              f"eid={eid!r} resp={resp!r}")
        check("stdin-chat serve 使 learn_total 增长",
              d.learn_total == before_total + 1,
              f"before={before_total} after={d.learn_total}")
        in_buffer = any(e.data == "你好，海豚".encode("utf-8") for e in d.buffer.items)
        check("stdin-chat 用户输入已进入 buffer", in_buffer,
              f"buffer 条目数={len(d.buffer.items)}")
        target = [e for e in d.buffer.items if e.id == eid]
        reward_ok = bool(target) and target[0].reward == 1.0
        check("stdin-chat 打分 +1 写入经验 reward",
              reward_ok, f"reward={target[0].reward if target else None}")

        # 异常路径 1：serve 抛异常 → chat_once 返回 None（部署不停机）
        orig_serve = d.serve
        try:
            d.serve = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("模拟 serve 故障"))
            r_none = chat_once(d, "这条会触发 serve 异常")
        finally:
            d.serve = orig_serve
        check("stdin-chat serve 异常返回 None（部署不停机）", r_none is None,
              f"result={r_none!r}")

        # 异常路径 2：空输入不应崩溃（serve 对空串正常处理或返回 None）
        orig_stdin2 = sys.stdin
        try:
            sys.stdin = io.StringIO("\n")  # 空打分=跳过
            r_empty = chat_once(d, "")
        finally:
            sys.stdin = orig_stdin2
        check("stdin-chat 空输入不崩溃", r_empty is None or (
            isinstance(r_empty, tuple) and len(r_empty) == 2),
              f"result={r_empty!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_stdin_chat_loop():
    """stdin 对话通道线程循环（2026-10-07）：有输入调用 chat_once、EOF 继续等待、
    stop_flag 置位后退出。

    用 os.pipe 模拟 stdin（写入后关闭写端 → 读端 EOF），monkeypatch chat_once
    记录调用次数；线程 join 超时防卡死测试。全部落在隔离沙箱，不碰生产资产。
    """
    import threading
    import time

    import feed as feed_mod

    tmp = tempfile.mkdtemp(prefix="dolphin_stdinloop_")
    try:
        probe = os.path.join(tmp, "probe.txt")
        _write_synthetic_probe(probe, 12)
        cfg = Config(d_model=64, n_layers=2, n_heads=2, block_size=256)
        d = make_dolphin(cfg=cfg, probe_path=probe, device="cpu")

        calls = []
        orig_chat_once = feed_mod.chat_once
        orig_stdin = sys.stdin
        t = None
        stop = threading.Event()
        try:
            # a) 有输入时调用 chat_once（monkeypatch 记录调用次数）
            feed_mod.chat_once = lambda d_, text: calls.append(text) or None
            r, w = os.pipe()
            os.write(w, "你好，海豚\n".encode("utf-8"))
            os.close(w)  # 关闭写端 → 读端最终 EOF
            sys.stdin = os.fdopen(r, "r", encoding="utf-8")
            t = threading.Thread(target=feed_mod.stdin_chat_loop, args=(d, stop),
                                 daemon=True, name="stdin-chat-loop-test")
            t.start()
            deadline = time.time() + 3.0
            while time.time() < deadline and not calls:
                time.sleep(0.02)
            check("stdin-chat-loop 有输入时调用 chat_once",
                  len(calls) == 1 and calls[0] == "你好，海豚", f"calls={calls}")

            # b) EOF（读端 line==""）且 stop_flag 未置位 → 继续等待（不退出）
            time.sleep(0.5)
            check("stdin-chat-loop EOF 且 stop 未置位时线程仍存活",
                  t.is_alive(), f"alive={t.is_alive()}")

            # c) stop_flag 置位后线程退出
            stop.set()
            t.join(timeout=3.0)
            check("stdin-chat-loop stop_flag 置位后线程退出",
                  not t.is_alive(), f"alive={t.is_alive()}")
        finally:
            stop.set()  # 确保即使中间断言失败，线程也会退出（daemon + join 防泄漏）
            if t is not None:
                t.join(timeout=3.0)
            sys.stdin = orig_stdin
            feed_mod.chat_once = orig_chat_once
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_amygdala():
    """杏仁核（Amygdala，2026-10-08）：威胁等级计算、慢环调整、钳位、持久化。

    直接构造 Amygdala 验证：冷启动 threat≈0.4；高威胁多次 adjust 后温度下降并
    自动启用 top_k/top_p 收紧；低威胁温度上升且不意外启用采样钳制；反复调整
    温度不越 [0.4, 1.4]；慢环首次/间隔不足返回 None、达到 AMYGDALA_EVERY 才调整；
    top_k/top_p 初值 0/1.0，手动设置后高威胁会继续收紧；to_state/from_state 往返一致。
    """
    from dolphin.amygdala import AMYGDALA_EVERY, Amygdala

    # a) 冷启动 threat_level 合理（margin_ema=0 → margin_threat=1，rollback=0，
    #    kd_ratio=0.5 → kd_threat=0 → threat=0.4）
    a0 = Amygdala()
    tl0 = a0.threat_level()
    check("amygdala 冷启动 threat_level 合理（≈0.4）",
          abs(tl0 - 0.4) < 1e-6, f"threat={tl0:.4f}")

    # b) 高威胁 report → 多次 adjust 后温度下降（direction=down）
    hi_report = {"gate_margin": 0.001, "passed": False, "ce_last": 1, "kd_last": 1}
    a_hi = Amygdala()
    r0 = a_hi.adjust(hi_report, cycle=0)
    check("amygdala 首次 adjust 返回 None（只记录基准）", r0 is None, f"r0={r0!r}")
    r1 = a_hi.adjust(hi_report, cycle=5)
    check("amygdala 间隔不足返回 None", r1 is None, f"r1={r1!r}")
    r2 = a_hi.adjust(hi_report, cycle=AMYGDALA_EVERY)
    check("amygdala 高威胁达到间隔后调整（温度下降）",
          r2 is not None and r2["direction"] == "down"
          and r2["temperature"] < 0.8,
          f"r2={r2}")
    check("amygdala 高威胁自动启用 top_k/top_p 收紧",
          r2 is not None and r2["top_k"] == 50 and abs(r2["top_p"] - 0.95) < 1e-6,
          f"top_k={r2 and r2['top_k']} top_p={r2 and r2['top_p']}")

    # c) 低威胁 report → 温度上升（direction=up），且不意外启用采样钳制
    lo_report = {"gate_margin": 0.5, "passed": True, "ce_last": 1, "kd_last": 0}
    a_lo = Amygdala()
    a_lo.adjust(lo_report, cycle=0)  # 基准
    r_lo = a_lo.adjust(lo_report, cycle=AMYGDALA_EVERY)
    check("amygdala 低威胁达到间隔后调整（温度上升）",
          r_lo is not None and r_lo["direction"] == "up"
          and r_lo["temperature"] > 0.8,
          f"r_lo={r_lo}")
    check("amygdala 低威胁不意外启用 top_k/top_p",
          r_lo is not None and r_lo["top_k"] == 0 and abs(r_lo["top_p"] - 1.0) < 1e-6,
          f"top_k={r_lo and r_lo['top_k']} top_p={r_lo and r_lo['top_p']}")

    # d) 钳位：反复 adjust 温度不越 [0.4, 1.4]
    a_clamp = Amygdala()
    a_clamp.adjust(hi_report, cycle=0)
    for c in range(AMYGDALA_EVERY, AMYGDALA_EVERY * 30, AMYGDALA_EVERY):
        a_clamp.adjust(hi_report, cycle=c)
    check("amygdala 高威胁反复收紧温度钳位 ≥0.4",
          a_clamp.temperature >= 0.4 - 1e-9, f"temp={a_clamp.temperature:.4f}")
    a_clamp2 = Amygdala()
    a_clamp2.adjust(lo_report, cycle=0)
    for c in range(AMYGDALA_EVERY, AMYGDALA_EVERY * 30, AMYGDALA_EVERY):
        a_clamp2.adjust(lo_report, cycle=c)
    check("amygdala 低威胁反复放松温度钳位 ≤1.4",
          a_clamp2.temperature <= 1.4 + 1e-9, f"temp={a_clamp2.temperature:.4f}")

    # e) 慢环：首次返回 None（只记录基准），间隔 <20 返回 None，达到 20 才调整
    a_slow = Amygdala()
    check("amygdala 慢环 首次返回 None",
          a_slow.adjust(hi_report, cycle=0) is None)
    check("amygdala 慢环 间隔<20 返回 None",
          a_slow.adjust(hi_report, cycle=19) is None)
    r_slow = a_slow.adjust(hi_report, cycle=20)
    check("amygdala 慢环 达到 20 才调整", r_slow is not None, f"r_slow={r_slow!r}")

    # f) top_k/top_p 初值 0/1.0，威胁高不意外启用；手动设置后高威胁会收紧
    a_f = Amygdala()
    check("amygdala top_k 初值 0（不启用）", a_f.top_k == 0)
    check("amygdala top_p 初值 1.0（不启用）", abs(a_f.top_p - 1.0) < 1e-9)
    a_f.adjust(hi_report, cycle=0)
    a_f.adjust(hi_report, cycle=5)  # 间隔不足，不应意外启用
    check("amygdala 间隔不足时 top_k 仍 0", a_f.top_k == 0)
    check("amygdala 间隔不足时 top_p 仍 1.0", abs(a_f.top_p - 1.0) < 1e-9)
    a_f.top_k = 50
    a_f.top_p = 0.9
    r_f = a_f.adjust(hi_report, cycle=AMYGDALA_EVERY)
    check("amygdala 手动设置 top_k=50 后高威胁继续收紧",
          r_f is not None and r_f["top_k"] < 50, f"top_k={r_f and r_f['top_k']}")
    check("amygdala 手动设置 top_p=0.9 后高威胁继续收紧",
          r_f is not None and r_f["top_p"] < 0.9,
          f"top_p={r_f and r_f['top_p']}")

    # g) save/load：to_state/from_state 往返一致
    a_save = Amygdala()
    a_save.adjust(hi_report, cycle=0)
    a_save.adjust(hi_report, cycle=AMYGDALA_EVERY)
    st = a_save.to_state()
    a_load = Amygdala()
    a_load.from_state(st)
    check("amygdala to_state/from_state 往返一致",
          a_load.temperature == a_save.temperature
          and a_load.top_k == a_save.top_k
          and a_load.top_p == a_save.top_p
          and a_load.signals == a_save.signals
          and a_load.last_adjust_cycle == a_save.last_adjust_cycle
          and a_load.last_adjust == a_save.last_adjust,
          f"orig={a_save.to_state()} load={a_load.to_state()}")


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
    t_loop_feed()         # 无限循环喂食（--loop-feed，2026-10-07）
    t_g3_structural_reward()
    t_g7_atomic_save()
    t_g7_rolling_backup()
    t_g7_eid_stability()
    t_g7_probe_hash()
    t_g8_gate_failfast()
    t_g9_threshold_clamp()
    t_g10_resurrect_all()
    t_g10_resurrect_cold_load()
    t_m4_vitals()
    t_m4_dev_birth()
    t_m4_widen_exact()
    t_m4_ghost_probe()
    t_m4_shrink_born_again()
    t_m4_life_equivalence()
    t_m4_redo_recycle()
    t_m4_anchor_persistence()
    t_m4_state_machine()
    t_m4_thermostat()
    t_stdin_chat()        # 无模式 stdin 对话通道（2026-10-07）
    t_stdin_chat_loop()   # stdin 对话通道线程循环（EOF/stop_flag 退出）
    t_amygdala()          # 杏仁核自动温度调节（2026-10-08）
    t_cold_quarantine()   # 隔离机制自检（先跑：此时污染若已发生，下面那条会一并变红）
    t_cold_untouched()    # 收尾红线：必须最后跑
    n_ok = sum(PASS)
    if SKIP:
        print(f"（跳过 {len(SKIP)} 项：{'; '.join(SKIP)}）")
    print(f"\n== 结论：{n_ok}/{len(PASS)} 绿 ==")
    sys.exit(0 if n_ok == len(PASS) else 1)
