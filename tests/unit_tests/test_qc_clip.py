"""成片质量巡检单测：inspect_clip 元数据读取（mock MoviePy，零额度）。

仅验证 qc.inspect_clip 的解析与降级行为，不触碰真实 agnes / 网络。
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.multimedia.tools import qc  # noqa: E402


class _FakeClip:
    """模拟 moviepy VideoFileClip 上下文管理器（with 协议）。

    构造时接受任意位置参数（与 ``VideoFileClip(path)`` 签名兼容），
    便于直接作为 ``_video_file_clip()`` 的替身。
    """

    def __init__(self, *args, duration: float = 8.0, size=(1920, 1080), fps: float = 24.0):
        self.duration = duration
        self.size = size
        self.fps = fps

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestInspectClip(unittest.TestCase):
    def _tmp_path(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        return path

    def test_reads_metadata(self):
        path = self._tmp_path()
        try:
            with mock.patch.object(qc, "_video_file_clip", return_value=_FakeClip):
                meta = qc.inspect_clip(path)
        finally:
            os.remove(path)
        self.assertTrue(meta["readable"])
        self.assertAlmostEqual(meta["duration"], 8.0)
        self.assertEqual(meta["width"], 1920)
        self.assertEqual(meta["height"], 1080)
        self.assertAlmostEqual(meta["fps"], 24.0)

    def test_missing_file_degrades(self):
        meta = qc.inspect_clip("/path/does/not/exist.mp4")
        self.assertFalse(meta["readable"])
        self.assertIn("不存在", meta["error"])

    def test_decode_error_degrades(self):
        path = self._tmp_path()
        try:
            with mock.patch.object(
                qc, "_video_file_clip", side_effect=RuntimeError("boom")
            ):
                meta = qc.inspect_clip(path)
        finally:
            os.remove(path)
        self.assertFalse(meta["readable"])
        self.assertIn("boom", meta["error"])


class TestCosceCosine(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(qc._cosine([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(qc._cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector(self):
        # 零向量视为无差异（返回 1.0）
        self.assertAlmostEqual(qc._cosine([0.0, 0.0], [1.0, 0.0]), 1.0)

    def test_mismatch_length(self):
        self.assertAlmostEqual(qc._cosine([1.0, 0.0], [1.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
