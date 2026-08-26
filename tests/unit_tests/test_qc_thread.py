"""成片质量巡检单测：check_thread 注入 / 一致性判定（mock state，零额度）。

验证 qc.check_thread 在以下场景的判定：
  - 成片缺失 → FAIL
  - 镜头片段时长偏离 → WARN
  - 素材未命中 → WARN（asset_match_report）
  - 跨镜头 embedding 漂移 → WARN（reference_embeddings）
  - 全绿 → PASS
不触碰真实 agnes / 网络 / 真实本地文件。
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.multimedia.tools import qc  # noqa: E402

_TID = "test_thread_001"


def _fake_clip_meta(duration: float, size=(1920, 1080)) -> dict:
    return {
        "readable": True,
        "duration": duration,
        "width": size[0],
        "height": size[1],
        "fps": 24.0,
        "error": "",
    }


class TestCheckThread(unittest.TestCase):
    def setUp(self):
        # 建一个临时 clips 目录并把 _PROJECT_ROOT 指向它，使成片 / 片段在
        # is_file() 判定中“存在”，从而让检查逻辑进入元数据分支而非缺失分支。
        self._tmp = tempfile.TemporaryDirectory()
        self._clips = Path(self._tmp.name) / "output" / "clips" / _TID
        self._clips.mkdir(parents=True)
        (self._clips / "FINAL_成片.mp4").write_bytes(b"\x00")
        (self._clips / "scene_00.mp4").write_bytes(b"\x00")
        (self._clips / "scene_01.mp4").write_bytes(b"\x00")
        self._root_patch = mock.patch.object(qc, "_PROJECT_ROOT", Path(self._tmp.name))
        self._root_patch.start()
        self._clip_patch = mock.patch.object(
            qc, "inspect_clip", return_value=_fake_clip_meta(8.0)
        )
        self._clip_patch.start()

    def tearDown(self):
        self._clip_patch.stop()
        self._root_patch.stop()
        self._tmp.cleanup()

    def _make_state(self, scenes, **extra):
        state = {"scenes": scenes}
        state.update(extra)
        return state

    def test_final_missing_is_fail(self):
        # 临时目录不放成片 → 成片缺失 → FAIL
        (self._clips / "FINAL_成片.mp4").unlink()
        report = qc.check_thread(_TID, state=self._make_state([]))
        self.assertEqual(report["summary"], "FAIL")
        names = [c["name"] for c in report["checks"]]
        self.assertIn("playable", names)

    def test_scene_duration_drift_is_warn(self):
        scenes = [
            {"image_url": "u1", "last_image_url": "v1"},
            {"image_url": "u2", "last_image_url": "v2"},
        ]
        # 仅镜头 2 时长偏离（3s）
        def _side(path: str) -> dict:
            if "scene_01.mp4" in path:
                return _fake_clip_meta(3.0)
            return _fake_clip_meta(8.0)

        with mock.patch.object(qc, "inspect_clip", side_effect=_side):
            report = qc.check_thread(_TID, state=self._make_state(scenes))
        levels = {c["name"]: c["level"] for c in report["checks"]}
        self.assertEqual(levels.get("scene_01_duration"), "WARN")
        self.assertEqual(levels.get("scene_00_duration"), "OK")

    def test_asset_unmatched_is_warn(self):
        state = self._make_state(
            [{"image_url": "u1"}],
            asset_match_report={"matched": {}, "unmatched": ["char_main.jpg"]},
        )
        report = qc.check_thread(_TID, state=state)
        levels = {c["name"]: c["level"] for c in report["checks"]}
        self.assertEqual(levels.get("asset_unmatched"), "WARN")
        self.assertEqual(levels.get("asset_matched"), "WARN")

    def test_consistency_drift_is_warn(self):
        # 两个差异很大的 embedding → 余弦相似度低 → WARN
        state = self._make_state(
            [{"image_url": "u1"}, {"image_url": "u2"}],
            reference_embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        report = qc.check_thread(_TID, state=state)
        drift_warn = any(
            c["level"] == "WARN" and c["name"].startswith("consistency_")
            for c in report["checks"]
        )
        self.assertTrue(drift_warn)

    def test_all_green_is_pass(self):
        state = self._make_state(
            [{"image_url": "u1", "last_image_url": "v1"}],
            asset_match_report={"matched": {"char_main.jpg": 1}, "unmatched": []},
            reference_embeddings=[[1.0, 0.0, 0.0], [0.99, 0.01, 0.0]],
        )
        report = qc.check_thread(_TID, state=state)
        self.assertEqual(report["summary"], "PASS")


if __name__ == "__main__":
    unittest.main()
