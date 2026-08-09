"""
大 state 瘦身单测（不依赖 graph 重依赖，零 DashScope 依赖）。

验证：
  - config_loader.persist_studio_output 能把大 studio_result 落盘并返回路径
  - 落盘文件可被读回且内容完整（下游按路径取回，避免大对象随 state 入库）
"""
import os
import sys
import json
import tempfile
import unittest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.multimedia.config_loader import persist_studio_output  # noqa: E402


class TestStudioSlim(unittest.TestCase):
    def test_persist_returns_path_and_file_exists(self):
        big = {
            "studio_consensus": 0.8,
            "modifications": [{"k": i} for i in range(500)],
            "debate": ["x"] * 200,
            "formatted_studio_report": "A" * 50000,
        }
        path = persist_studio_output(big, idx=3)
        self.assertTrue(path)
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["studio_consensus"], 0.8)
        self.assertEqual(len(loaded["modifications"]), 500)
        # 清理
        os.remove(path)

    def test_persist_handles_unserializable(self):
        # default=str 兜底：含不可序列化对象时仍能落盘
        obj = {"ts": __import__("datetime").datetime.now(), "nested": {"x": 1}}
        path = persist_studio_output(obj, idx=9)
        self.assertTrue(path)
        self.assertTrue(os.path.isfile(path))
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
