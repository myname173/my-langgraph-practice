# -*- coding: utf-8 -*-
"""P2-3 graph 拆分的输出路径回归守卫（零网络 / 零额度）。

背景：`assemble.py` / `render.py` 从 <root>/src/agent/multimedia/ 移入
<root>/src/agent/multimedia/nodes/ 后，仍沿用旧的 `__file__` 上溯层数，
实际落到 <root>/src 或 <root>/src/agent（比仓库根少一层），
导致成片写进 src/output，而 /media、history_api、agnes 都按 <root>/output 解析
→ 前端取不到带配音+字幕的完整成片。

本测试锁定「写入侧」与「读取侧」的输出根必须一致，防止再次漂移。
"""
import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.multimedia.tools import media_paths as mp  # noqa: E402


def _expect_output_dir():
    return os.path.join(_PROJECT_ROOT, "output")


def test_media_paths_output_dir_is_repo_output():
    assert str(mp.output_dir()) == _expect_output_dir()


def test_node_modules_use_repo_output_dir():
    """nodes/ 下的写入侧必须解析到仓库根的 output/。"""
    assemble = pytest.importorskip("agent.multimedia.nodes.assemble")
    render = pytest.importorskip("agent.multimedia.nodes.render")
    assert str(assemble._output_dir()) == _expect_output_dir()
    assert str(render._output_dir()) == _expect_output_dir()


def test_reader_modules_use_repo_output_dir():
    """读取侧（history_api / static_server / 各 tool）的输出根必须与写入侧一致。"""
    history_api = pytest.importorskip("agent.multimedia.history_api")
    static_server = pytest.importorskip("agent.multimedia.static_server")
    agnes = pytest.importorskip("agent.multimedia.tools.agnes")
    jimeng = pytest.importorskip("agent.multimedia.tools.jimeng")
    image_gen = pytest.importorskip("agent.multimedia.tools.image_gen")

    assert os.path.join(history_api._project_root(), "output") == _expect_output_dir()
    assert str(static_server.OUTPUT_DIR) == _expect_output_dir()
    assert str(agnes.PROJECT_ROOT / "output") == _expect_output_dir()
    assert str(jimeng.PROJECT_ROOT / "output") == _expect_output_dir()
    assert str(image_gen.PROJECT_ROOT / "output") == _expect_output_dir()


def test_write_and_read_roots_agree():
    """核心断言：写入根 == 读取根（回归守卫）。"""
    assemble = pytest.importorskip("agent.multimedia.nodes.assemble")
    history_api = pytest.importorskip("agent.multimedia.history_api")
    write_root = str(assemble._output_dir())
    read_root = os.path.join(history_api._project_root(), "output")
    assert write_root == read_root


def test_to_media_url_maps_repo_output():
    """<root>/output/... 必须映射为 /media/...（且不重复 output 段）。"""
    history_api = pytest.importorskip("agent.multimedia.history_api")
    p = os.path.join(_PROJECT_ROOT, "output", "clips", "t1", "FINAL_成片_packed_subs_audio.mp4")
    url = history_api._to_media_url(p)
    assert url == "/media/clips/t1/FINAL_成片_packed_subs_audio.mp4"
    assert "output/output" not in url


def test_history_api_fallback_names_cover_packed_variants():
    """P2-1 包装层插入 _packed：磁盘兜底名单必须两种命名都覆盖。"""
    history_api = pytest.importorskip("agent.multimedia.history_api")
    import inspect

    src = inspect.getsource(history_api)
    for name in (
        "FINAL_成片_subs_audio.mp4",
        "FINAL_成片_packed_subs_audio.mp4",
        "FINAL_成片_subs.mp4",
        "FINAL_成片_packed_subs.mp4",
        "FINAL_成片_packed.mp4",
        "FINAL_成片.mp4",
    ):
        assert name in src, f"兜底名单缺 {name}"
