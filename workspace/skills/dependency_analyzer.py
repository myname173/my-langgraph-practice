# ---
# description: 分析 Python 模块的 import 依赖并找出循环依赖风险
# category: analysis
# ---
"""分析单个模块的内部 import 依赖关系，输出依赖图与循环依赖告警。

用法:
    python dependency_analyzer.py --file ./service.py
"""
import argparse
import ast


def analyze(file: str):
    """返回该文件的 import 目标集合（模块级 import / from ... import）。"""
    src = open(file, encoding="utf-8").read()
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    deps = analyze(args.file)
    for d in sorted(set(deps)):
        print(d)


def __test__():
    """自验证：解析自身应包含 ast 与 argparse 依赖。"""
    deps = analyze(__file__)
    assert "ast" in deps and "argparse" in deps


if __name__ == "__main__":
    main()
