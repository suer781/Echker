"""B 路线部署期喂食数据集解析器。

每个解析器是生成器函数，yield {"prompt": str, "answer": str, "source": str}。
铁律：test/dev 等评测 split 永不进训练粮（律 L8 前置）——它们是体检候选。
序列化后短于 MIN_CHARS 的碎片一律跳过。
"""
import json
import os

# 数据根：部署机上 O: 盘挂载数据集（工作目录里的 datasets 符号链接亦指向它）
DATA_ROOT = "O:/数据集"
# 碎片过滤：序列化后短于该长度没有统计结构的噪声（语料白名单.md 工序 4）
MIN_CHARS = 30

# LogiQA 2.0 的 MRC train split（只吃 train；dev/test/ood_test 是体检候选，禁碰）
_LOGIQA2_TRAIN = DATA_ROOT + "/01_LogiQA/LogiQA-2.0" \
    "/LogiQA2.0-main/logiqa/DATA/LOGIQA/train_zh.txt"


def _norm(path):
    return os.path.normpath(path)


def _utf8_ok(b):
    """抽样切片可能在头/尾拦腰截断多字节字符：两端各容忍最多 3 字节残缺再试。"""
    for skip in range(4):
        seg = b[skip:]
        for trim in range(4):
            body = seg[:len(seg) - trim] if trim else seg
            try:
                body.decode("utf-8")
                return True
            except UnicodeDecodeError:
                continue
    return False


def _open_text(path):
    """utf-8 优先；头尾抽样解码失败再退 gbk。返回文件对象。"""
    with open(path, "rb") as f:
        head = f.read(65536)
        f.seek(-min(65536, os.path.getsize(path)), 2)
        tail = f.read()
    enc = "utf-8" if (_utf8_ok(head) and _utf8_ok(tail)) else "gbk"
    return open(path, encoding=enc)


def _iter_jsonl(path):
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _clean(s):
    return " ".join(str(s or "").split())


# ---------- 逻辑推理 ----------

def _read_logiqa_txt(path):
    """LogiQA 1.0 txt：空行分块，每块 7 行 = 答案字母 / 题干 / 问题 / A-D 选项。"""
    with _open_text(path) as f:
        block = []
        for line in f:
            line = line.strip()
            if not line:
                if len(block) == 7:
                    yield block
                block = []
                continue
            block.append(line)
        if len(block) == 7:
            yield block


def read_logiqa():
    p = _norm(DATA_ROOT + "/01_LogiQA/LogiQA-1.0/LogiQA-dataset-master/zh_train.txt")
    for b in _read_logiqa_txt(p):
        letter = b[0].strip().lower()
        gold = "abcd".find(letter)
        opts = b[3:7]
        if gold < 0 or not opts[gold][2:]:
            continue
        prompt = b[1] + "\n" + b[2] + "\n" + "\n".join(opts)
        # 答案 = 字母 + 对应选项原文（去掉选项行自带的 "X." 前缀）
        answer = letter.upper() + ". " + opts[gold][2:]
        yield {"prompt": prompt, "answer": answer, "source": "logiqa"}


def read_logiqa2():
    """LogiQA 2.0 中文 MRC train（jsonl）。dev/test/ood_test 一律不碰。"""
    p = _norm(_LOGIQA2_TRAIN)
    for rec in _iter_jsonl(p):
        opts = [str(o) for o in rec.get("options") or []]
        if len(opts) != 4:
            continue
        ans = int(rec.get("answer", -1))
        if not 0 <= ans <= 3 or not opts[ans]:
            continue
        prompt = _clean(rec.get("text")) + "\n" + _clean(rec.get("question")) \
            + "\n" + "\n".join(opts)
        answer = "ABCD"[ans] + ". " + opts[ans][2:]
        yield {"prompt": prompt, "answer": answer, "source": "logiqa2"}


def read_logiconbench():
    """LogiConBench：程序生成的逻辑一致性语料（模板英文陈述）。

    只在 --sources 显式指定时启用；模板味重，默认不当口粮。
    """
    base = _norm(DATA_ROOT + "/06_LogiConBench/LogiConBench-main")
    for name in ("2statements.jsonl", "3statements.jsonl",
                 "4statements.jsonl", "5statements.jsonl"):
        for rec in _iter_jsonl(os.path.join(base, name)):
            nodes = [str(s) for s in rec.get("nl_nodes") or []]
            if not nodes:
                continue
            prompt = "\n".join(nodes) + "\n问：以上推理链是否自洽？"
            answer = ("等价逻辑式 " + str(rec.get("rewritten_expr"))
                      + "；一致赋值集 " + str(rec.get("valid_sets"))
                      + "；不一致赋值集 " + str(rec.get("invalid_sets")))
            yield {"prompt": prompt, "answer": answer, "source": "logiconbench"}


# ---------- 数学推理 ----------

def read_cmath():
    """CMATH dev（train 无公开答案；cmath_test 是体检候选，禁碰）。"""
    p = _norm(DATA_ROOT + "/03_数学推理/CMATH/cmath_dev.jsonl")
    for rec in _iter_jsonl(p):
        q, g = _clean(rec.get("question")), _clean(rec.get("golden"))
        if q and g:
            yield {"prompt": q, "answer": g, "source": "cmath"}


def read_gsm8k():
    """GSM8K 中文平行版。带 split 字段：只吃 train，test 不进粮。"""
    p = _norm(DATA_ROOT + "/03_数学推理/GSM8K_zh/GSM8K_zh.json")
    with _open_text(p) as f:
        data = json.load(f)
    for rec in data:
        if rec.get("split") != "train":
            continue
        q = _clean(rec.get("question_zh"))
        a = str(rec.get("answer_zh") or "").strip()
        if q and a:
            yield {"prompt": q, "answer": a, "source": "gsm8k"}


def read_metamath():
    """MetaMathQA 去重版（jsonl，尽管扩展名是 .json）。取中文平行字段。"""
    p = _norm(DATA_ROOT + "/03_数学推理/MetaMathQA_GSM8K_zh/"
              "MetaMathQA_GSM8K_zh_dedup.json")
    for rec in _iter_jsonl(p):
        q = _clean(rec.get("query_zh"))
        a = " ".join(str(rec.get("response_zh") or "").split())
        if q and a:
            yield {"prompt": q, "answer": a, "source": "metamath"}


# ---------- 推理指令 ----------

def read_distil():
    """Chinese-Reasoning-Distil-Data：17.9 万条带思维链的核心粮。

    keys 实测为 id/prompt/reasoning/response——response 是思维链之后的正式答案。
    """
    for name in ("train.jsonl", "train-2.jsonl"):
        p = _norm(DATA_ROOT + "/05_推理指令/Chinese-Reasoning-Distil-Data/" + name)
        for rec in _iter_jsonl(p):
            prompt = _clean(rec.get("prompt"))
            reasoning = str(rec.get("reasoning") or "").strip()
            response = str(rec.get("response") or "").strip()
            if not prompt or not (reasoning or response):
                continue
            answer = reasoning + ("\n" + response if response else "")
            yield {"prompt": prompt, "answer": answer, "source": "distil"}


def read_coig():
    """COIG-CQIA 去重版。instruction+input 为题面，output 为答案。"""
    p = _norm(DATA_ROOT + "/05_推理指令/COIG-CQIA/COIG-CQIA-full_dedup.jsonl")
    for rec in _iter_jsonl(p):
        prompt = _clean(rec.get("instruction"))
        extra = _clean(rec.get("input"))
        if extra:
            prompt += "\n" + extra
        answer = str(rec.get("output") or "").strip()
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "coig"}


def _zh_ratio(s):
    s = s or ""
    if not s.strip():
        return 0.0
    zh = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return zh / max(1, len(s.replace(" ", "")))


def read_synlogic():
    """SynLogic 合成逻辑题（parquet，easy+hard 的 train split；validation 禁碰）。

    中英混合：只取 question 字段中文字符占比 > 30% 的条目。
    题面在 extra_info.game_data_str（json 串）的 question/answer 里。
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[datasets] 未安装 pyarrow，synlogic 源跳过（pip install pyarrow 后可用）")
        return
    for sub in ("synlogic_easy", "synlogic_hard"):
        p = _norm(DATA_ROOT + "/05_推理指令/SynLogic/" + sub + "/train.parquet")
        pf = pq.ParquetFile(p)  # 流式分批读，不整表进内存
        for batch in pf.iter_batches(batch_size=256):
            for rec in batch.to_pylist():
                ei = rec.get("extra_info") or {}
                gds = ei.get("game_data_str")
                if not gds:
                    continue
                try:
                    gd = json.loads(gds)
                except json.JSONDecodeError:
                    continue
                q = _clean(gd.get("question"))
                a = _clean(gd.get("answer"))
                if not q or not a:
                    continue
                if _zh_ratio(q) <= 0.30:
                    continue
                yield {"prompt": q, "answer": a, "source": "synlogic"}


# ---------- 统一序列化 ----------

SOURCES = ["logiqa", "logiqa2", "cmath", "gsm8k", "distil",
           "coig", "synlogic", "metamath", "logiconbench"]

# 质量排序的默认喂食顺序（logiconbench 默认不启用）
DEFAULT_ORDER = ["logiqa", "cmath", "gsm8k", "distil", "coig", "synlogic", "metamath"]

_READERS = {
    "logiqa": read_logiqa, "logiqa2": read_logiqa2, "cmath": read_cmath,
    "gsm8k": read_gsm8k, "distil": read_distil, "coig": read_coig,
    "synlogic": read_synlogic, "metamath": read_metamath,
    "logiconbench": read_logiconbench,
}


def serialize(rec):
    """统一口粮格式。短于 MIN_CHARS 的碎片返回空串（调用方跳过）。"""
    s = "问：" + rec["prompt"] + "\n答：" + rec["answer"]
    return s if len(s) >= MIN_CHARS else ""


def iter_source(name):
    """按名取解析器；未知名直接报错（feed.py 的 --sources 白名单校验用）。"""
    if name not in _READERS:
        raise KeyError(f"未知数据源 {name}，可选：{SOURCES}")
    return _READERS[name]()
