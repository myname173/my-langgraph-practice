# ---
# description: 在给定 Python 仓库中按符号名检索其定义位置与签名上下文
# category: retrieval
# ---
"""在给定代码库中检索符号（函数/类）定义，返回文件路径、行号与签名上下文。

用法:
    python code_symbol_search.py --repo ./my_project --symbol MyClass.my_method
"""
import argparse
import ast
import os
import sys


def find_symbol(repo: str, symbol: str):
    """遍历 repo 下的 .py 文件，定位 symbol（支持 `Class.method` 形式）的 ast 定义。"""
    results = []
    target_class, _, target_method = symbol.partition(".")
    for root, _dirs, files in os.walk(repo):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and (not target_class or node.name == target_class):
                    results.append((path, node.lineno, f"class {node.name}"))
                    if target_method:
                        for sub in node.body:
                            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == target_method:
                                results.append((path, sub.lineno, f"def {sub.name}(...)"))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not target_class:
                    if node.name == target_method or node.name == symbol:
                        results.append((path, node.lineno, f"def {node.name}(...)"))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--symbol", required=True)
    args = ap.parse_args()
    hits = find_symbol(args.repo, args.symbol)
    for path, lineno, sig in hits:
        print(f"{path}:{lineno}  {sig}")


def __test__():
    """自验证：在当前文件所在包内检索自身函数应至少命中 1 处。"""
    here = os.path.dirname(os.path.abspath(__file__))
    hits = find_symbol(here, "find_symbol")
    assert len(hits) >= 1, "应在自身文件中检索到 find_symbol"


if __name__ == "__main__":
    main()
