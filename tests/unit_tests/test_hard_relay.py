"""P0-hard 硬接力：把上一镜真实末帧直接用作下一镜视频首帧（零额度单测）。

背景（thread real2shot_lib_1791279154 实测）：把上一镜真实末帧仅当 img2img 软参考时，
即梦「多参考融合」保不住构图 —— 重绘首帧与接力帧像素相关 r≈-0.02、视频实际首帧与
接力帧 r≈0.11、接缝 r≈0.11，观众看到明显跳变。故新增硬接力：存在可解析的真实接力帧
时，直接以其作为本镜视频首帧（真正的 FLF 首尾帧链）。

覆盖 image_gen_node 的硬接力分支：
- 默认开启：命中可解析接力帧 → image_url == 接力帧本地路径、跳过生图、置 continuity 标记；
- ``MULTIMEDIA_FLF_HARD_RELAY=0`` → 回退软参考重绘（不再直接取接力帧）；
- 接力引用不可解析 → 回退；
- ``idx == 0`` → 不触发；
- ``/media`` URL → 解析为存在的本地文件。
"""
import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

render = pytest.importorskip("agent.multimedia.nodes.render")
mp = pytest.importorskip("agent.multimedia.tools.media_paths")


def _mk_state(idx, relay_url, assets_imported=False, use_flf=True):
    return {
        "current_scene_index": idx,
        "scenes": [
            {"image_url": "", "script": "s", "dialogue": "", "iterations": 0,
             "image_prompt": "p", "assets": []}
            for _ in range(2)
        ],
        "scene_anchors": {"last_frame_url": relay_url},
        "use_first_last_frame": use_flf,
        "assets_imported": assets_imported,
        "reference_sheets": {},
        "asset_match_report": {},
        "thread_id": "pytest_hard_relay",
        "quality_tier": "standard",
    }


def _relay_file(tmp_path):
    p = tmp_path / "linked_last_shot1.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return str(p)


def test_hard_relay_enabled_default(monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_FLF_HARD_RELAY", raising=False)
    assert render._hard_relay_enabled() is True
    monkeypatch.setenv("MULTIMEDIA_FLF_HARD_RELAY", "off")
    assert render._hard_relay_enabled() is False
    monkeypatch.setenv("MULTIMEDIA_FLF_HARD_RELAY", "0")
    assert render._hard_relay_enabled() is False


def test_hard_relay_uses_prev_last_frame(tmp_path, monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_FLF_HARD_RELAY", raising=False)
    relay = _relay_file(tmp_path)
    calls = {"n": 0}

    def _fake_kf(*a, **k):
        calls["n"] += 1
        return "MOCK_KF.png"

    monkeypatch.setattr(render, "generate_keyframe", _fake_kf)
    out = render.image_gen_node(_mk_state(1, relay))
    sc = out["scenes"][1]
    assert sc["image_url"] == relay
    assert sc.get("continuity_relay_first_frame") is True
    assert calls["n"] == 0                     # 跳过首帧重绘（省额度）
    assert out["use_first_last_frame"] is True
    # 必须是视频节点接受的「本地绝对路径」形态（盘符: + isabs）
    assert os.path.isabs(sc["image_url"]) and sc["image_url"][1] == ":"
    # 配方卡记录来源为 relay
    assert (sc.get("render_params") or {}).get("image", {}).get("backend") == "relay"


def test_hard_relay_disabled_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_FLF_HARD_RELAY", "0")
    relay = _relay_file(tmp_path)
    calls = {"n": 0}

    def _fake_kf(*a, **k):
        calls["n"] += 1
        return "MOCK_KF.png"

    monkeypatch.setattr(render, "generate_keyframe", _fake_kf)
    out = render.image_gen_node(_mk_state(1, relay))
    assert out["scenes"][1]["image_url"] != relay
    assert calls["n"] >= 1                     # 走了软参考重绘


def test_hard_relay_unresolvable_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_FLF_HARD_RELAY", raising=False)
    missing = str(tmp_path / "does_not_exist.png")
    monkeypatch.setattr(render, "generate_keyframe", lambda *a, **k: "MOCK_KF.png")
    out = render.image_gen_node(_mk_state(1, missing))
    assert out["scenes"][1]["image_url"] != missing


def test_first_shot_no_relay(tmp_path, monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_FLF_HARD_RELAY", raising=False)
    relay = _relay_file(tmp_path)
    monkeypatch.setattr(render, "generate_keyframe", lambda *a, **k: "MOCK_KF.png")
    out = render.image_gen_node(_mk_state(0, relay))
    assert out["scenes"][0]["image_url"] != relay


def test_hard_relay_resolves_media_url(monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_FLF_HARD_RELAY", raising=False)
    d = mp.output_dir() / "keyframes" / "pytest_hard_relay"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "linked_last_shot1_test.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    url = mp.local_to_media_url(f)
    assert url and url.startswith("/media/")
    calls = {"n": 0}

    def _fake_kf(*a, **k):
        calls["n"] += 1
        return "MOCK.png"

    monkeypatch.setattr(render, "generate_keyframe", _fake_kf)
    out = render.image_gen_node(_mk_state(1, url))
    assert out["scenes"][1]["image_url"] == str(f)
    assert calls["n"] == 0
