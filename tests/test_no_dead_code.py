"""S11 反功能吹捧扫描：断言 dolphin/ 下不存在「定义但零调用」的公共方法。

原理→代码映射：工程纪律（无生物对应）——每个公共函数/方法必须有真实调用方，
杜绝「预留接口/后续接入」式幻觉实现。映射为 tests/test_no_dead_code.py 的
AST 常驻扫描（纳入 pytest 套件）。

扫描规则：
1. 用 ast 遍历 dolphin/ 下所有 .py 文件，收集模块级函数与类方法名。
2. 只扫描公共方法（不以 "_" 开头的函数/方法）。
3. 豁免反射调用：__init__（构造）、to_state / from_state（save/load 序列化路径
   反射调用）、forward（PyTorch Module.__call__ 反射调用）。
4. 在全仓库 .py 文件（含 tests/ 与入口脚本）中搜索每个公共方法名的 AST 引用
   （Name / Attribute.attr / import 别名）。定义自身的 def 语句不计。
5. 断言死方法列表为空；若不为空，测试失败并列出。
"""
import ast
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOLPHIN_DIR = os.path.join(REPO_ROOT, "dolphin")

# 反射调用豁免：save/load 序列化路径与 PyTorch 运行时按名字调用，AST 中无直接调用点。
# settle（S5 债务结算）与 salience_score（S7 杏仁核显著性）已由 life.py 接线，
# cfg_dict 无真实调用方（save 用 dataclasses.asdict），已移出白名单并删除方法。
EXEMPT = {"__init__", "to_state", "from_state", "forward"}


def _collect_definitions():
    """遍历 dolphin/ 收集模块级函数与类方法名（不含嵌套局部函数）。"""
    defs = set()
    for fname in sorted(os.listdir(DOLPHIN_DIR)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(DOLPHIN_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in tree.body:  # 模块级
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and node.name not in EXEMPT:
                    defs.add(node.name)
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:  # 类方法
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not sub.name.startswith("_") and sub.name not in EXEMPT:
                            defs.add(sub.name)
    return defs


def _collect_references():
    """遍历全仓库 .py 文件，收集所有被引用的名字。

    覆盖 Name（如模块级函数调用/字典引用）、Attribute.attr（如 self.method()、
    obj.method()）、import 别名（from .x import y / import y）。定义自身的 def
    语句不会产生 Name/Attribute 节点，天然不计。
    """
    refs = set()
    for root, dirs, files in os.walk(REPO_ROOT):
        # 跳过隐藏目录、缓存与外部数据集（datasets/ 是下载的第三方语料，
        # 其中的脚本不属于本项目代码，不能作为 dolphin 方法的调用方）。
        dirs[:] = [d for d in dirs
                  if not d.startswith((".", "__pycache__")) and d != "datasets"]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    refs.add(node.id)
                elif isinstance(node, ast.Attribute):
                    refs.add(node.attr)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        refs.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        refs.add(alias.asname or alias.name)
    return refs


def test_no_dead_public_methods():
    """断言：不存在「定义但零调用」的公共方法。"""
    defs = _collect_definitions()
    refs = _collect_references()
    dead = sorted(name for name in defs if name not in refs)
    assert not dead, f"存在定义但零调用的公共方法（S11 反功能吹捧扫描失败）：{dead}"


if __name__ == "__main__":
    test_no_dead_public_methods()
    print("S11 死代码扫描通过：dolphin/ 下无「定义但零调用」的公共方法。")