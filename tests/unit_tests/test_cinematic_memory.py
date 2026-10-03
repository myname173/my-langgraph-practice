# -*- coding: utf-8 -*-
"""P0-3 记忆持久化单测：cinematic_memory 线程作用域 + 落盘重建（零额度、无网络）。

验证：
  - 写入即落盘（data/memory/<thread>.json）
  - 缓存清空后（模拟进程重启）能从落盘重建前一镜头风格快照
  - 线程作用域隔离（不同 thread_id 互不可见）
  - reset_memory 缓存 + 落盘双清
  - get_current_style_state 聚合 + generate_film_brain_report 注入
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.multimedia import cinematic_memory as cm  # noqa: E402


def _cine(grade="teal-orange", score=0.9, shots=("wide",)):
    return {"style_summary": {"grade": grade, "shot_types": list(shots)}, "consistency_score": score}


class TestCinematicMemoryPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._orig_dir = cm._MEM_DIR
        cm._MEM_DIR = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cm._MEM_DIR = cls._orig_dir
        cls._tmp.cleanup()

    def setUp(self):
        self.tid = "test_cm_thread"
        self.other = "test_cm_other"
        cm.reset_memory(self.tid)
        cm.reset_memory(self.other)

    def tearDown(self):
        cm.reset_memory(self.tid)
        cm.reset_memory(self.other)

    def _record(self, tid, idx, cine=None, light=None, edit=None):
        cm.record_scene_style(
            tid, idx,
            cine or _cine(shots=("wide", "close-up")),
            light or {"style_summary": {"key": "low-key"}, "consistency_score": 0.85},
            edit or {"style_summary": {"transitions": ["cut", "dissolve"]}, "pacing_score": 0.8},
        )

    def test_write_persists_file(self):
        self._record(self.tid, 0)
        f = cm._MEM_DIR / f"{self.tid}.json"
        self.assertTrue(f.exists(), "写入后应生成落盘文件")
        data = json.loads(f.read_text(encoding="utf-8"))
        self.assertEqual(set(data.keys()), {"0"})
        self.assertEqual(data["0"]["cinematography"]["grade"], "teal-orange")

    def test_rebuild_after_restart(self):
        """模拟进程重启：清空内存缓存与已加载标记，应从落盘重建 scene-0。"""
        self._record(self.tid, 0, _cine(grade="amber", score=0.93))
        cm._banks.pop(self.tid, None)
        cm._loaded.discard(self.tid)
        prev = cm.get_previous_scene_style(self.tid, 1)
        self.assertIsNotNone(prev, "重启后应能从落盘重建前一镜头风格快照")
        self.assertEqual(prev["cinematography"]["grade"], "amber")
        self.assertEqual(prev["coherence_scores"]["camera"], 0.93)

    def test_thread_isolation(self):
        self._record(self.tid, 0)
        self.assertIsNone(cm.get_previous_scene_style(self.other, 1),
                          "其他线程不应看到本线程记忆")
        self.assertIsNotNone(cm.get_previous_scene_style(self.tid, 1))

    def test_reset_clears_cache_and_disk(self):
        self._record(self.tid, 0)
        f = cm._MEM_DIR / f"{self.tid}.json"
        self.assertTrue(f.exists())
        cm.reset_memory(self.tid)
        self.assertFalse(f.exists(), "reset 应删除落盘文件")
        self.assertIsNone(cm.get_previous_scene_style(self.tid, 1))

    def test_state_aggregation_and_report(self):
        self._record(self.tid, 0, _cine(shots=("wide",)))
        self._record(self.tid, 1, _cine(score=1.0, shots=("tracking",)))
        st = cm.get_current_style_state(self.tid)
        self.assertEqual(st["scenes_recorded"], 2)
        self.assertEqual(st["established_style"], "established")
        self.assertEqual(st["camera_consistency"], round((0.9 + 1.0) / 2, 2))
        self.assertIn("tracking", st["all_shot_types"])
        report = cm.generate_film_brain_report("global", total_scenes=5, thread_id=self.tid)
        self.assertIn("2/5 scenes processed", report)
        self.assertIn("wide", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
