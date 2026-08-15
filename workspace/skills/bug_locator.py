# ---
# description: 根据异常堆栈文本定位最可能的出错源文件与行号
# category: debugging
# ---
"""接收一段 traceback 文本，提取最后一帧的 `File "...", line N` 信息。

用法:
    python bug_locator.py --traceback ./err.log
"""
import argparse
import re


_TRACE_RE = re.compile(r'File "([^"]+)", line (\d+)')


def locate(traceback_text: str):
    """返回 traceback 中所有 (文件, 行号) 帧，按出现顺序。"""
    return [(m.group(1), int(m.group(2))) for m in _TRACE_RE.finditer(traceback_text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traceback", required=True)
    args = ap.parse_args()
    text = open(args.traceback, encoding="utf-8").read()
    frames = locate(text)
    for path, lineno in frames:
        print(f"{path}:{lineno}")


def __test__():
    """自验证：标准 traceback 片段应解析出至少一帧。"""
    tb = 'Traceback (most recent call last):\n  File "app.py", line 10, in <module>\n'
    frames = locate(tb)
    assert frames == [("app.py", 10)]


if __name__ == "__main__":
    main()
