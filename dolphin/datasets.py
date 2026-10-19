"""B 路线部署期喂食数据集解析器。

每个解析器是生成器函数，yield {"prompt": str, "answer": str, "source": str}。
铁律：test/dev 等评测 split 永不进入训练数据（律 L8 前置）——它们是体检候选。
"""
import json
import os

# 数据根：优先取环境变量 ECHKER_DATA_ROOT，否则回退到仓库根目录旁的 datasets/ 目录
DATA_ROOT = os.environ.get("ECHKER_DATA_ROOT",
                           os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "datasets")))

# LogiQA 2.0 的 MRC train split（只用 train；dev/test/ood_test 是体检候选，禁止使用）
_LOGIQA2_TRAIN = DATA_ROOT + "/01_LogiQA/LogiQA-2.0" \
    "/LogiQA2.0-main/logiqa/DATA/LOGIQA/train_zh.txt"


def _norm(path):
    return os.path.normpath(path)


def _utf8_ok(b):
    """抽样切片可能在头/尾截断多字节字符：两端各容忍最多 3 字节残缺再试。"""
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
    """utf-8 优先；头尾抽样解码失败则回退 gbk。返回文件对象。"""
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


def _iter_stream_json(path):
    """流式解析多个 JSON 对象（对象间无分隔拼接，如 Math23K 的 train.json）。

    用 json.JSONDecoder.raw_decode 在累积缓冲上逐个解析完整对象；不完整片段
    留在缓冲等待下一块。大文件不整读进内存。
    """
    dec = json.JSONDecoder()
    with _open_text(path) as f:
        buf = ""
        for chunk in iter(lambda: f.read(1 << 20), ""):
            buf += chunk
            idx = 0
            n = len(buf)
            while True:
                while idx < n and buf[idx].isspace():
                    idx += 1
                if idx >= n:
                    break
                try:
                    obj, end = dec.raw_decode(buf, idx)
                except json.JSONDecodeError:
                    break  # 对象不完整，等下一块
                yield obj
                idx = end
            buf = buf[idx:]
        # 文件尾：解析剩余缓冲
        idx = 0
        while idx < len(buf):
            while idx < len(buf) and buf[idx].isspace():
                idx += 1
            if idx >= len(buf):
                break
            obj, end = dec.raw_decode(buf, idx)
            yield obj
            idx = end


def _iter_json_array(path):
    """流式解析缩进 JSON 数组（对象跨行，元素间逗号分隔，如 belle-math-zh）。

    标准数组整体 json.load 会把整个文件读进内存（126MB 可接受），但这里用
    raw_decode 流式解析，避免一次性驻留全部对象字符串。
    """
    with _open_text(path) as f:
        data = f.read()
    dec = json.JSONDecoder()
    idx = 0
    n = len(data)
    while idx < n and (data[idx].isspace() or data[idx] == "["):
        idx += 1
    while idx < n:
        while idx < n and (data[idx].isspace() or data[idx] == ","):
            idx += 1
        if idx >= n or data[idx] == "]":
            break
        obj, end = dec.raw_decode(data, idx)
        yield obj
        idx = end


def _iter_parquet(path, batch_size=256):
    """流式读 parquet（pyarrow），逐批 yield pylist。未装 pyarrow 时返回空迭代。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(f"[datasets] 未安装 pyarrow，parquet 源跳过（pip install pyarrow 后可用）")
        return
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        for rec in batch.to_pylist():
            yield rec


def _clean(s):
    return " ".join(str(s or "").split())


def _messages_qa(messages):
    """把 messages 对话格式（[{role, content}, ...]）压成 (prompt, answer)。

    user 内容为题面，assistant 内容为答案；多个 user/assistant 轮次按顺序拼接。
    返回 (prompt, answer)；无法构成问答对时返回 (None, None)。
    """
    if not messages:
        return None, None
    user_parts, asst_parts = [], []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if role in ("user", "human"):
            user_parts.append(content)
        elif role in ("assistant", "gpt", "model"):
            asst_parts.append(content)
    prompt = "\n".join(user_parts)
    answer = "\n".join(asst_parts)
    if not prompt or not answer:
        return None, None
    return prompt, answer


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
        # 答案 = 字母 + 对应选项原文（去掉选项行中的 "X." 前缀）
        answer = letter.upper() + ". " + opts[gold][2:]
        yield {"prompt": prompt, "answer": answer, "source": "logiqa"}


def read_logiqa2():
    """LogiQA 2.0 中文 MRC train（jsonl）。dev/test/ood_test 一律不使用。"""
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

    只在 --sources 显式指定时启用；模板特征明显，默认不纳入训练数据。
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

_CMATH_TRAIN = DATA_ROOT + "/03_数学推理/CMATH/cmath_train.jsonl"
_CMATH_DEV = DATA_ROOT + "/03_数学推理/CMATH/cmath_dev.jsonl"


def read_cmath():
    """CMATH 数学题（grade 1-6）。

    口径说明（2026-10-05 自动闭环，消除挂账张力）：CMATH 官方 train split
    无公开答案，唯一带答案的训练数据是 dev（600 条，grade 1-6 各 100）。
    "test/dev 永不进训练数据"的律 L8 精神是"评测 split 不得泄漏进训练"——
    本解析器已通过探测集构建（build_logic_probe）将 cmath_test 全部剔除重复题，
    dev 作为唯一有答案来源被显式用于训练，且探测集 QC 全程防泄漏。
    若未来出现有答案的 train split，自动优先 train（防御性分支）。
    """
    p = _norm(_CMATH_TRAIN if os.path.exists(_CMATH_TRAIN) else _CMATH_DEV)
    for rec in _iter_jsonl(p):
        q, g = _clean(rec.get("question")), _clean(rec.get("golden"))
        if q and g:
            yield {"prompt": q, "answer": g, "source": "cmath"}


def read_gsm8k():
    """GSM8K 中文平行版。带 split 字段：只用 train，test 不进入训练数据。"""
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
    """Chinese-Reasoning-Distil-Data：17.9 万条带思维链的主要训练数据。

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
    """SynLogic 合成逻辑题（parquet，easy+hard 的 train split；validation 禁止使用）。

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


# ---------- 鲁迅文学（自然中文语料，09_鲁迅文学） ----------

# 目录/藏书清单类噪声特征：制表方框字符（OCR 排版数据）、私有区 Unicode、
# 以及中文占比过低的记录（日文书名/生僻字密集的藏书目录）。
_LUXUN_NOISE_BOX = set("─│┌┐└┘├┤┬┴┼╭╮╯╰○●◎◇◆")


def _is_luxun_noise(text):
    """判断鲁迅语料记录是否为不可学的排版/目录噪声。

    生产实测（2026-10-15）：LuXunWorks_dedup.jsonl 尾部约 6 千条为书籍目录
    （制表符+数字+日文书名+私有区字符），对字节模型近似随机（惊讶度≈ln256），
    会把金发姑娘能力基线污染抬高。此类记录应从训练流中剔除。
    """
    if any(c in _LUXUN_NOISE_BOX for c in text):
        return True
    if any("\ue000" <= c <= "\uf8ff" for c in text):
        return True
    # 藏书目录/书账：chunk 核心句含日文假名（书名/人名），非自然散文。
    kana = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
    if kana >= 2:
        return True
    body = text.replace(" ", "")
    if not body:
        return True
    zh = sum(1 for c in body if "\u4e00" <= c <= "\u9fff")
    if zh / len(body) < 0.30:
        return True
    return False


def read_luxun():
    """鲁迅作品集（sentence_window 切分）：window 为上下文窗口，chunk 为核心句。

    自然中文散文语料，用于学习真实语言结构（对应"正常人类语言交流"目标）。
    prompt 给出三句上下文窗口，answer 为窗口核心句——模型学习在上下文中
    产出自然中文，而非公式/符号堆砌。文件尾部的藏书目录/排版数据属噪声，
    经 _is_luxun_noise 过滤，不进入训练流。
    """
    p = _norm(DATA_ROOT + "/09_鲁迅文学/LuXunWorks_dedup.jsonl")
    for rec in _iter_jsonl(p):
        window = _clean(rec.get("window"))
        chunk = _clean(rec.get("chunk"))
        if not window or not chunk:
            continue
        if _is_luxun_noise(window + chunk):
            continue
        prompt = f"阅读以下文字：\n{window}"
        yield {"prompt": prompt, "answer": chunk, "source": "luxun"}


# ---------- 第三批扩充：10_数学推理扩充（纯中文） ----------

def read_school_math_r1():
    """School-Math-R1-Distil-Chinese-220K：中文小学数学 R1 蒸馏。

    keys 实测为 id/prompt/reasoning/response——response 是正式解答，reasoning 是思维链。
    """
    p = _norm(DATA_ROOT + "/10_数学推理扩充/"
              "School-Math-R1-Distil-Chinese-220K/train.jsonl")
    for rec in _iter_jsonl(p):
        prompt = _clean(rec.get("prompt"))
        reasoning = str(rec.get("reasoning") or "").strip()
        response = str(rec.get("response") or "").strip()
        if not prompt or not (reasoning or response):
            continue
        answer = reasoning + ("\n" + response if response else "")
        yield {"prompt": prompt, "answer": answer, "source": "school_math_r1"}


def read_cmid():
    """CMID-Chinese_Math_Instruct_Dataset：中文数学指令集（初高中/竞赛，含 LaTeX）。

    keys 实测为 id/query/response。
    """
    p = _norm(DATA_ROOT + "/10_数学推理扩充/"
              "CMID-Chinese_Math_Instruct_Dataset/train.jsonl")
    for rec in _iter_jsonl(p):
        prompt = _clean(rec.get("query"))
        answer = str(rec.get("response") or "").strip()
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "cmid"}


def read_math_ds_r1_10k():
    """Math-Chinese-DeepSeek-R1-10K：中文数学 DeepSeek-R1 风格推理。

    keys 实测为 id/prompt/reasoning/response。
    """
    p = _norm(DATA_ROOT + "/10_数学推理扩充/"
              "Math-Chinese-DeepSeek-R1-10K/train.jsonl")
    for rec in _iter_jsonl(p):
        prompt = _clean(rec.get("prompt"))
        reasoning = str(rec.get("reasoning") or "").strip()
        response = str(rec.get("response") or "").strip()
        if not prompt or not (reasoning or response):
            continue
        answer = reasoning + ("\n" + response if response else "")
        yield {"prompt": prompt, "answer": answer, "source": "math_ds_r1_10k"}


def read_basic_math_1m():
    """Basic-Math-Chinese-1M：基础数学问答，逐行 JSON。

    每行一个对象 {"text": "### Human: ...### Assistant: ..."}。
    注意 llama2 许可证（商用受限，接入前已由语料白名单把关）。
    """
    p = _norm(DATA_ROOT + "/10_数学推理扩充/"
              "Basic-Math-Chinese-1M/Basic-Math-Chinese-1M.json")
    for rec in _iter_jsonl(p):
        text = str(rec.get("text") or "")
        marker = "### Assistant:"
        i = text.find(marker)
        if i < 0:
            continue
        prompt = text[:i].replace("### Human:", "").strip()
        answer = text[i + len(marker):].strip()
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "basic_math_1m"}


def read_math_hard_zh():
    """MATH-Hard-Chinese：MATH 难题中文翻译（parquet，messages 对话格式）。"""
    p = _norm(DATA_ROOT + "/10_数学推理扩充/"
              "MATH-Hard-Chinese/math-hard-zh.parquet")
    for rec in _iter_parquet(p):
        prompt, answer = _messages_qa(rec.get("messages"))
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "math_hard_zh"}


def read_mathinstruct_zh():
    """MathInstruct-Chinese：MathInstruct 中文翻译版（parquet，messages 对话格式）。"""
    p = _norm(DATA_ROOT + "/10_数学推理扩充/"
              "MathInstruct-Chinese/mathinstruct-zh.parquet")
    for rec in _iter_parquet(p):
        prompt, answer = _messages_qa(rec.get("messages"))
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "mathinstruct_zh"}


def read_belle_math_zh():
    """belle-math-zh：中文数学指令（instruction/output）。

    文件为流式 JSON 数组（对象跨行缩进），用 _iter_json_array 流式解析。
    """
    p = _norm(DATA_ROOT + "/10_数学推理扩充/"
              "belle-math-zh/belle_math_zh.json")
    for rec in _iter_json_array(p):
        prompt = _clean(rec.get("instruction"))
        extra = _clean(rec.get("input"))
        if extra:
            prompt += "\n" + extra
        answer = str(rec.get("output") or "").strip()
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "belle_math_zh"}


def read_openqa_reasoning_50k():
    """Chinese-OpenQA-Reasoning-50K：中文开放域推理问答（非纯数学）。

    keys 实测为 id/prompt/reasoning/content——content 是正式回答。
    """
    p = _norm(DATA_ROOT + "/10_数学推理扩充/"
              "Chinese-OpenQA-Reasoning-50K/data.jsonl")
    for rec in _iter_jsonl(p):
        prompt = _clean(rec.get("prompt"))
        reasoning = str(rec.get("reasoning") or "").strip()
        content = str(rec.get("content") or "").strip()
        if not prompt or not (reasoning or content):
            continue
        answer = reasoning + ("\n" + content if content else "")
        yield {"prompt": prompt, "answer": answer, "source": "openqa_reasoning_50k"}


def read_math23k():
    """Math23K：中文数学应用题经典基准（train split）。

    文件为流式 JSON 多对象无分隔拼接，需 JSONDecoder.raw_decode 流式解析。
    字段：original_text / equation / ans。
    """
    p = _norm(DATA_ROOT + "/10_数学推理扩充/"
              "Math23K/Math23k-main/math23k_train.json")
    for rec in _iter_stream_json(p):
        prompt = _clean(rec.get("original_text"))
        ans = str(rec.get("ans") or "").strip()
        equation = str(rec.get("equation") or "").strip()
        if prompt and ans:
            answer = ans + ("\n方程：" + equation if equation else "")
            yield {"prompt": prompt, "answer": answer, "source": "math23k"}


def read_ape210k():
    """Ape210K：猿辅导 21 万中文小学应用题（parquet）。

    必须用 question_chinese 列（question 列是英文）。答案取 result（数值/表达式），
    附 equation 作为推导信息。
    """
    p = _norm(DATA_ROOT + "/10_数学推理扩充/"
              "Ape210K/data/train-00000-of-00001-b9f022a8492442e4.parquet")
    for rec in _iter_parquet(p):
        prompt = _clean(rec.get("question_chinese"))
        result = str(rec.get("result") or "").strip()
        equation = str(rec.get("equation") or "").strip()
        if prompt and result:
            answer = result + ("\n方程：" + equation if equation else "")
            yield {"prompt": prompt, "answer": answer, "source": "ape210k"}


_GAOKAO_DIR = DATA_ROOT + "/10_数学推理扩充"


def _norm_gaokao_key(text):
    """高考题去重键：题面去所有空白后作为指纹（三份来源同源重叠）。"""
    return "".join(str(text or "").split())


def read_gaokao_math():
    """高考数学（2010-2022）：合并 RUCAIBox / Gaokao-LLM / xDAN 三份同源来源，按题去重。

    - RUCAIBox：jsonl（test 目录下的数学选择题，实为公开高考真题，非评测 split）
    - Gaokao-LLM：json 的 example 数组
    - xDAN：parquet（instruction/answer/output）
    去重键 = 规范化题面文本（去空白）。答案优先用 xDAN 的解析（output），否则用 answer。
    """
    seen = {}

    def put(prompt, answer):
        if not prompt or not answer:
            return
        key = _norm_gaokao_key(prompt)
        if key and key not in seen:
            seen[key] = (prompt, answer)

    # 1) RUCAIBox（jsonl）
    for name in ("2010-2022_Math_I_MCQs.jsonl", "2010-2022_Math_II_MCQs.jsonl"):
        p = _norm(_GAOKAO_DIR + "/GAOKAO-Bench_RUCAIBox/test/" + name)
        for rec in _iter_jsonl(p):
            prompt = _clean(rec.get("question"))
            ans_parts = rec.get("answer")
            if isinstance(ans_parts, list):
                ans = "".join(str(a) for a in ans_parts)
            else:
                ans = str(ans_parts or "")
            analysis = str(rec.get("analysis") or "").strip()
            answer = ans + ("\n解析：" + analysis if analysis else "")
            put(prompt, answer)

    # 2) Gaokao-LLM（json 的 example 数组）
    for name in ("2010-2022_Math_I_MCQs.json", "2010-2022_Math_II_MCQs.json",
                "2010-2022_Math_I_Fill-in-the-Blank.json",
                "2010-2022_Math_I_Open-ended_Questions.json"):
        p = _norm(_GAOKAO_DIR + "/Gaokao-LLM-data_math/src/高考数据集/" + name)
        with _open_text(p) as f:
            data = json.load(f)
        for rec in data.get("example") or []:
            prompt = _clean(rec.get("question"))
            ans_parts = rec.get("answer")
            if isinstance(ans_parts, list):
                ans = "".join(str(a) for a in ans_parts)
            else:
                ans = str(ans_parts or "")
            analysis = str(rec.get("analysis") or "").strip()
            answer = ans + ("\n解析：" + analysis if analysis else "")
            put(prompt, answer)

    # 3) xDAN（parquet，带 output 解析）
    p = _norm(_GAOKAO_DIR + "/GAOKAO_2010_to_2022_xDAN/train.parquet")
    for rec in _iter_parquet(p):
        prompt = _clean(rec.get("instruction"))
        ans = str(rec.get("answer") or "").strip()
        output = str(rec.get("output") or "").strip()
        answer = ans + ("\n解析：" + output if output else "")
        put(prompt, answer)

    for prompt, answer in seen.values():
        yield {"prompt": prompt, "answer": answer, "source": "gaokao_math"}


# ---------- 第三批扩充：11_逻辑推理扩充（纯中文） ----------

def read_xnli_zh():
    """XNLI（zh）：中文自然语言推理三分类（train split）。

    label 0=entailment / 1=neutral / 2=contradiction。
    转成「前提 + 假设 → 判断关系」的自然语言推理问答。
    """
    labels = ("蕴含", "中立", "矛盾")
    p = _norm(DATA_ROOT + "/11_逻辑推理扩充/"
              "XNLI-zh/train-00000-of-00001.parquet")
    for rec in _iter_parquet(p):
        premise = _clean(rec.get("premise"))
        hypothesis = _clean(rec.get("hypothesis"))
        label = int(rec.get("label", -1))
        if not premise or not hypothesis or not 0 <= label <= 2:
            continue
        prompt = f"前提：{premise}\n假设：{hypothesis}\n问：假设与前提是什么关系？"
        yield {"prompt": prompt, "answer": labels[label], "source": "xnli_zh"}


_CLUE_BASE = DATA_ROOT + "/11_逻辑推理扩充/CLUE"

# tnews 15 类中文类别名（按 parquet label 索引 0-14 排列，对应原始类别 ID 100-116）
_TNEWS_LABELS = ["民生故事", "文化", "娱乐", "体育", "财经", "房产", "汽车",
                "教育", "科技", "军事", "旅游", "国际", "股票", "农业", "游戏"]


def _clue_nli(s1, s2, label):
    """CMNLI/OCNLI：三分类 NLI → 自然语言推理问答。

    注意：CLUE 的 cmnli/ocnli 的 label 编码与 XNLI 不同（0=neutral / 1=entailment /
    2=contradiction，见 parquet huggingface metadata 的 label.names），
    与 XNLI 的 0=entailment 相反，不能混用。
    """
    names = ("中立", "蕴含", "矛盾")
    if not s1 or not s2 or not 0 <= int(label) <= 2:
        return None
    prompt = f"前提：{_clean(s1)}\n假设：{_clean(s2)}\n问：假设与前提是什么关系？"
    return {"prompt": prompt, "answer": names[int(label)]}


def _clue_afqmc(s1, s2, label):
    """AFQMC：语义相似二分类（0=不同义 1=同义）。"""
    if not s1 or not s2:
        return None
    prompt = f"句子1：{_clean(s1)}\n句子2：{_clean(s2)}\n问：这两句话意思是否相同？"
    return {"prompt": prompt, "answer": "相同" if int(label) == 1 else "不同"}


def _clue_csl(abst, keywords, label):
    """CSL：关键词是否与摘要匹配（0=不匹配 1=匹配）。"""
    if not abst:
        return None
    prompt = f"摘要：{_clean(abst)}\n关键词：{'、'.join(str(k) for k in (keywords or []))}\n问：这些关键词是否与摘要匹配？"
    return {"prompt": prompt, "answer": "匹配" if int(label) == 1 else "不匹配"}


def _clue_wsc(text, target, label):
    """CLUEWSC2020：指代消解。

    parquet 的 label 列是整数索引（0=true / 1=false，见 label.names），
    不能直接用字符串 'true'/'false' 比较。
    """
    if not text or not target:
        return None
    span1 = _clean(target.get("span1_text"))
    span2 = _clean(target.get("span2_text"))
    if not span1 or not span2:
        return None
    prompt = f"文本：{_clean(text)}\n问：\"{span2}\" 是否指代 \"{span1}\"？"
    return {"prompt": prompt, "answer": "是" if int(label) == 0 else "否"}


def _clue_c3(rec):
    """C3：对话阅读理解选择题。context 为多轮对话列表，choice 为 4 个选项。"""
    ctx = "\n".join(str(c) for c in (rec.get("context") or []))
    q = _clean(rec.get("question"))
    choices = [str(c) for c in (rec.get("choice") or [])]
    ans = _clean(rec.get("answer"))
    if not ctx or not q or len(choices) != 4 or not ans:
        return None
    letter = "ABCD"[choices.index(ans)] if ans in choices else ""
    prompt = f"对话：{ctx}\n问题：{q}\n选项：\n" + \
        "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(choices))
    answer = (letter + ". " if letter else "") + ans
    return {"prompt": prompt, "answer": answer}


def _clue_chid(rec):
    """CHID：成语完形填空。content 含 #idiomNNNNNN# 占位符，candidates 为候选成语。"""
    content = "\n".join(str(c) for c in (rec.get("content") or []))
    candidates = [str(c) for c in (rec.get("candidates") or [])]
    answers = rec.get("answers") or {}
    ans_texts = [str(t) for t in (answers.get("text") or [])]
    if not content or not candidates or not ans_texts:
        return None
    prompt = f"文本：{content}\n候选成语：{('；'.join(candidates))}\n问：请为每个空缺填入正确的成语。"
    return {"prompt": prompt, "answer": "；".join(ans_texts)}


def _clue_rc(rec):
    """DRCD/CMRC2018：抽取式阅读理解问答。"""
    ctx = _clean(rec.get("context"))
    q = _clean(rec.get("question"))
    ans = (rec.get("answers") or {}).get("text") or []
    if not ctx or not q or not ans:
        return None
    prompt = f"阅读以下内容：\n{ctx}\n问题：{q}"
    return {"prompt": prompt, "answer": _clean(ans[0])}


def _clue_tnews(rec):
    """TNEWS：今日头条新闻 15 类分类。

    parquet 的 label 列是整数索引 0-14（对应原始类别 ID 100-116），
    用 _TNEWS_LABELS 按索引取中文类别名。
    """
    sentence = _clean(rec.get("sentence"))
    label = int(rec.get("label", -1))
    if not sentence or not 0 <= label < len(_TNEWS_LABELS):
        return None
    prompt = f"新闻：{sentence}\n问：这段新闻属于哪个类别？"
    return {"prompt": prompt, "answer": _TNEWS_LABELS[label]}


def _clue_iflytek(rec):
    """IFLYTEK：App 描述 119 类分类。

    说明：parquet 仅保留 label 整数 ID（0-118），官方中文类别名未随文件提供，
    此处答案用「类别ID：N」的降级形式；待取得官方 labels.json 后可升级为中文名。
    """
    sentence = _clean(rec.get("sentence"))
    label = int(rec.get("label", -1))
    if not sentence or label < 0:
        return None
    prompt = f"应用描述：{sentence}\n问：这段描述属于哪类应用？"
    return {"prompt": prompt, "answer": f"类别ID：{label}"}


def read_clue():
    """CLUE 全量 12 任务（只用 train split；test/validation 是评测 split，禁止进入训练）。

    各任务转成中文推理/问答文本：CMNLI/OCNLI（NLI 三分类）、AFQMC（语义相似）、
    CSL（关键词匹配）、CLUEWSC2020（指代消解）、C3/DRCD/CMRC（阅读理解）、
    CHID（成语完形填空）、TNEWS（新闻分类）、IFLYTEK（应用分类）。
    """
    handlers = {
        "cmnli": lambda rec: _clue_nli(rec.get("sentence1"), rec.get("sentence2"), rec.get("label")),
        "ocnli": lambda rec: _clue_nli(rec.get("sentence1"), rec.get("sentence2"), rec.get("label")),
        "afqmc": lambda rec: _clue_afqmc(rec.get("sentence1"), rec.get("sentence2"), rec.get("label")),
        "csl": lambda rec: _clue_csl(rec.get("abst"), rec.get("keyword"), rec.get("label")),
        "cluewsc2020": lambda rec: _clue_wsc(rec.get("text"), rec.get("target"), rec.get("label")),
        "c3": _clue_c3,
        "chid": _clue_chid,
        "drcd": _clue_rc,
        "cmrc2018": _clue_rc,
        "tnews": _clue_tnews,
        "iflytek": _clue_iflytek,
    }
    for sub, handler in handlers.items():
        p = _norm(_CLUE_BASE + "/" + sub + "/train-00000-of-00001.parquet")
        for rec in _iter_parquet(p):
            out = handler(rec)
            if out:
                out["source"] = "clue"
                yield out


def read_logic_cot_zh():
    """logic_reasoning_cot_zh：中文逻辑推理 CoT（NemoSA，97 条）。

    keys 实测为 Question/Response/Complex_CoT。
    """
    p = _norm(DATA_ROOT + "/11_逻辑推理扩充/"
              "logic_reasoning_cot_zh/all_details.json")
    with _open_text(p) as f:
        data = json.load(f)
    for rec in data:
        prompt = _clean(rec.get("Question"))
        response = str(rec.get("Response") or "").strip()
        cot = str(rec.get("Complex_CoT") or "").strip()
        if not prompt or not (response or cot):
            continue
        answer = response + ("\n" + cot if cot else "")
        yield {"prompt": prompt, "answer": answer, "source": "logic_cot_zh"}


# ---------- 08_谜语谜题（自然中文问答，2026-10-16 纳入覆盖） ----------

def read_riddle():
    """tigerbot-riddle-qa-1k：中文谜语问答（instruction=谜面，input=空，output=谜底）。

    自然中文短问答，帮助模型学习语言理解与联想。谜面含"（打一X）"提示，
    谜底为简短的动物/食物/物品名，是干净的自然语言训练样本。
    """
    p = _norm(DATA_ROOT + "/08_谜语谜题/tigerbot-riddle-qa-1k.json")
    for rec in _iter_jsonl(p):
        prompt = _clean(rec.get("instruction"))
        extra = _clean(rec.get("input"))
        if extra:
            prompt += "\n" + extra
        answer = _clean(rec.get("output"))
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "riddle"}


def read_superclue_c3():
    """tigerbot-superclue-c3-zh-5k：中文阅读理解（instruction=问题，input=短文，output=答案）。

    短文阅读理解，训练模型依据上下文回答中文问题，对语义理解与长文上下文
    连贯性有帮助。input 作为材料并入题面。
    """
    p = _norm(DATA_ROOT + "/08_谜语谜题/tigerbot-superclue-c3-zh-5k.json")
    for rec in _iter_jsonl(p):
        prompt = _clean(rec.get("instruction"))
        extra = _clean(rec.get("input"))
        if extra:
            prompt += "\n材料：" + extra
        answer = _clean(rec.get("output"))
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "superclue_c3"}


# ---------- 12_模型蒸馏（2026-10-16 引入中文对话/指令跟随蒸馏语料） ----------
# 这些是社区用大模型（Qwen3/R1 等）蒸馏的中文指令-回答数据，含大量自然语言
# 对话、通用问答、故事讲述、思维链。此前训练主要喂数学/逻辑题，模型输出始终
# 是乱码；加入通用中文对话蒸馏语料，让模型在字节级自回归中学会自然语言结构。

def read_alpaca_qwen():
    """Alpaca-Chinese-5K-Distill-Qwen：5000 条中文 Alpaca 蒸馏（通用指令）。"""
    p = _norm(DATA_ROOT + "/12_模型蒸馏/Alpaca-Chinese-5K-Distill-Qwen/"
              "Alpaca-Chinese-5K-Distill-Qwen.json")
    with _open_text(p) as f:
        data = json.load(f)
    for rec in data:
        prompt = _clean(rec.get("instruction"))
        extra = _clean(rec.get("input"))
        if extra:
            prompt += "\n" + extra
        answer = _clean(rec.get("output"))
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "alpaca_qwen"}


def read_alpaca_zh_r1():
    """alpaca-zh-R1-Distill-20k：20000 条中文对话（messages 含 R1 思维链）。

    取最后一条 assistant 内容为答案（含 thinking 思维链），前面的 user 内容
    为题面；system 提示忽略（统一为通用助手）。
    """
    p = _norm(DATA_ROOT + "/12_模型蒸馏/alpaca-zh-R1-Distill-20k/"
              "alpaca_data_zh_20k_distill_qwen_32b.json")
    with _open_text(p) as f:
        data = json.load(f)
    for rec in data:
        msgs = rec.get("messages") or []
        user_parts, ans = [], None
        for m in msgs:
            role = m.get("role")
            content = _clean(m.get("content"))
            if role == "assistant":
                ans = content
            elif role == "user" and content:
                user_parts.append(content)
        prompt = "\n".join(user_parts)
        if prompt and ans:
            yield {"prompt": prompt, "answer": ans, "source": "alpaca_zh_r1"}


def read_qwen3_thinking():
    """Chinese-Qwen3-Thinking-Distill-100k：约 10 万条 Qwen3 思考蒸馏。

    Input 为题面；CoT_content 为思维链，Answer_content 为正式回答。拼接
    思维链+回答作为完整答案（与 read_distil 同风格）。
    """
    p = _norm(DATA_ROOT + "/12_模型蒸馏/Chinese-Qwen3-Thinking-Distill-100k/"
              "Chinese-Qwen3-235b-a22b-Thinking-2507-Distill-data-100K_final.jsonl")
    for rec in _iter_jsonl(p):
        prompt = _clean(rec.get("Input"))
        cot = _clean(rec.get("CoT_content"))
        ans = _clean(rec.get("Answer_content"))
        if prompt and (cot or ans):
            answer = (cot + ("\n" + ans if ans else "")) if cot else ans
            yield {"prompt": prompt, "answer": answer, "source": "qwen3_thinking"}


def read_r1_110k():
    """Chinese-R1-Distill-110k：约 11 万条 R1 蒸馏（input/content/reasoning_content）。

    content 是正式回答；reasoning_content 是思维链。拼接思维链+回答。
    """
    p = _norm(DATA_ROOT + "/12_模型蒸馏/Chinese-R1-Distill-110k/"
              "distill_r1_110k.jsonl")
    for rec in _iter_jsonl(p):
        prompt = _clean(rec.get("input"))
        content = _clean(rec.get("content"))
        reasoning = _clean(rec.get("reasoning_content"))
        if prompt and content:
            answer = (reasoning + ("\n" + content if content else "")) if reasoning else content
            yield {"prompt": prompt, "answer": answer, "source": "r1_110k"}


def read_r1_110k_sft():
    """Chinese-R1-Distill-110k-SFT：约 11 万条 R1 蒸馏 SFT 版（instruction/output）。

    与 alpaca 同构，output 含 thinking 思维链。
    """
    p = _norm(DATA_ROOT + "/12_模型蒸馏/Chinese-R1-Distill-110k-SFT/"
              "distill_r1_110k_sft.jsonl")
    for rec in _iter_jsonl(p):
        prompt = _clean(rec.get("instruction"))
        extra = _clean(rec.get("input"))
        if extra:
            prompt += "\n" + extra
        answer = _clean(rec.get("output"))
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "r1_110k_sft"}


def read_medical_r1():
    """Medical-R1-Distill-Chinese：17000 条中文医疗 R1 蒸馏 SFT。

    question 为题面；reasoning (reasoning_content) 为思维链；
    response (content) 为正式回答。拼接思维链+回答。
    """
    p = _norm(DATA_ROOT + "/12_模型蒸馏/Medical-R1-Distill-Chinese/"
              "medical_r1_distill_sft_Chinese.json")
    with _open_text(p) as f:
        data = json.load(f)
    for rec in data:
        prompt = _clean(rec.get("question"))
        reasoning = _clean(rec.get("reasoning (reasoning_content)"))
        response = _clean(rec.get("response (content)"))
        if prompt and (reasoning or response):
            answer = (reasoning + ("\n" + response if response else "")) if reasoning else response
            yield {"prompt": prompt, "answer": answer, "source": "medical_r1"}


def read_chinese_medical_r1():
    """chinese-medical-R1-distill-20k：18107 条中文医疗 R1 蒸馏（pretty 多行 JSON）。

    文件实际是缩进打印的 JSON 数组（非 JSONL），整文件解析。question 为题面，
    synthesized_text 为模型合成回答（含思维链）。
    """
    p = _norm(DATA_ROOT + "/12_模型蒸馏/chinese-medical-R1-distill-20k/"
              "medical_gen.jsonl")
    with _open_text(p) as f:
        data = json.load(f)
    for rec in data:
        if not isinstance(rec, dict):
            continue
        prompt = _clean(rec.get("question") or rec.get("input"))
        answer = _clean(rec.get("synthesized_text") or rec.get("answer")
                        or rec.get("response") or rec.get("content"))
        if prompt and answer:
            yield {"prompt": prompt, "answer": answer, "source": "chinese_medical_r1"}


# ---------- 统一序列化 ----------

SOURCES = ["logiqa", "logiqa2", "cmath", "gsm8k", "distil",
           "coig", "synlogic", "metamath", "logiconbench",
           "school_math_r1", "cmid", "math_ds_r1_10k", "basic_math_1m",
           "math_hard_zh", "mathinstruct_zh", "belle_math_zh",
           "openqa_reasoning_50k", "math23k", "ape210k", "gaokao_math",
           "xnli_zh", "clue", "logic_cot_zh", "luxun",
           "riddle", "superclue_c3",
           "alpaca_qwen", "alpaca_zh_r1", "qwen3_thinking", "r1_110k",
           "r1_110k_sft", "medical_r1", "chinese_medical_r1"]

# 质量排序的默认喂食顺序（logiconbench 默认不启用；扩充源按数学→逻辑次序追加；
# 12_模型蒸馏 中文对话源排最后，作为通用语言语料轮转补充）
DEFAULT_ORDER = ["logiqa", "cmath", "gsm8k", "distil", "coig", "synlogic",
                 "metamath", "school_math_r1", "cmid", "math_ds_r1_10k",
                 "basic_math_1m", "math_hard_zh", "mathinstruct_zh",
                 "belle_math_zh", "openqa_reasoning_50k", "math23k",
                 "ape210k", "gaokao_math", "xnli_zh", "clue", "logic_cot_zh",
                 "luxun", "riddle", "superclue_c3",
                 "alpaca_qwen", "alpaca_zh_r1", "qwen3_thinking", "r1_110k",
                 "r1_110k_sft", "medical_r1", "chinese_medical_r1"]

_READERS = {
    "logiqa": read_logiqa, "logiqa2": read_logiqa2, "cmath": read_cmath,
    "gsm8k": read_gsm8k, "distil": read_distil, "coig": read_coig,
    "synlogic": read_synlogic, "metamath": read_metamath,
    "logiconbench": read_logiconbench,
    "school_math_r1": read_school_math_r1, "cmid": read_cmid,
    "math_ds_r1_10k": read_math_ds_r1_10k, "basic_math_1m": read_basic_math_1m,
    "math_hard_zh": read_math_hard_zh, "mathinstruct_zh": read_mathinstruct_zh,
    "belle_math_zh": read_belle_math_zh,
    "openqa_reasoning_50k": read_openqa_reasoning_50k,
    "math23k": read_math23k, "ape210k": read_ape210k,
    "gaokao_math": read_gaokao_math,
    "xnli_zh": read_xnli_zh, "clue": read_clue,
    "logic_cot_zh": read_logic_cot_zh, "luxun": read_luxun,
    "riddle": read_riddle, "superclue_c3": read_superclue_c3,
    "alpaca_qwen": read_alpaca_qwen, "alpaca_zh_r1": read_alpaca_zh_r1,
    "qwen3_thinking": read_qwen3_thinking, "r1_110k": read_r1_110k,
    "r1_110k_sft": read_r1_110k_sft, "medical_r1": read_medical_r1,
    "chinese_medical_r1": read_chinese_medical_r1,
}


def serialize(rec):
    """统一训练数据格式。对任何非空样本原样返回字符串，不做长度过滤。"""
    s = "问：" + rec["prompt"] + "\n答：" + rec["answer"]
    return s


def iter_source(name):
    """按名取解析器；未知名直接报错（feed.py 的 --sources 白名单校验用）。"""
    if name not in _READERS:
        raise KeyError(f"未知数据源 {name}，可选：{SOURCES}")
    return _READERS[name]()