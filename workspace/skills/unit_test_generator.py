# ---
# description: 为指定的 Python 函数/类生成最小 pytest 单测脚手架
# category: testing
# ---
"""为给定源文件中的目标函数生成 pytest 单测脚手架。

用法:
    python unit_test_generator.py --file ./calc.py --symbol add --out ./test_calc.py
"""
import argparse
import ast
import os


def generate_test(file: str, symbol: str) -> str:
    """解析源文件，提取 symbol 的函数签名，生成带占位断言的 pytest 用例。"""
    src = open(file, encoding="utf-8").read()
    tree = ast.parse(src)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            target = node
            break
    if target is None:
        raise ValueError(f"未找到符号: {symbol}")
    args = [a.arg for a in target.args.args if a.arg != "self"]
    sig = ", ".join(args)
    return (
        f"def test_{symbol}():\n"
        f"    from {os.path.splitext(os.path.basename(file))[0]} import {symbol}\n"
        f"    # TODO: 补充真实断言\n"
        f"    result = {symbol}({sig})\n"
        f"    assert result is not None\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    code = generate_test(args.file, args.symbol)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(code)
    else:
        print(code)


def __test__():
    """自验证：对自身 generate_test 逻辑做最小 smoke 检查。"""
    code = generate_test(__file__, "generate_test")
    assert "def test_generate_test" in code


if __name__ == "__main__":
    main()
