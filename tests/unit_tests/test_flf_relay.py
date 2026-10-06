"""P0 真实末帧接力：从上一镜成片视频抽取物理末帧作为下一镜承接锚点（零额度单测）。

覆盖：
- ``relay_mode`` 默认 video / env 覆盖 keyframe；
- ``resolve_local_video``：本地路径 / ``/media`` URL / 不存在；
- ``extract_last_frame``：真实 mp4 → ``/media`` URL + 落盘非空；不存在 → None；
- ``_relay_frame_from_scene``：video 模式走抽取、失败回退计划尾帧图；keyframe 模式直取图。

依赖 imageio_ffmpeg 自带 ffmpeg 生成测试片段；缺失则 skip，不破坏 CI。
"""
import os
import subprocess
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

graph = pytest.importorskip("agent.multimedia.graph")
fe = pytest.importorskip("agent.multimedia.tools.frame_extract")

_TH = "pytest_relay"


def _make_test_mp4(path):
    """用 imageio_ffmpeg 的 ffmpeg 生成 1s 测试色块视频；不可用返回 None。"""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = [
        exe, "-y", "-f", "lavfi",
        "-i", "testsrc=size=320x240:rate=10:duration=1",
        "-pix_fmt", "yuv420p", path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        return None
    return path if (r.returncode == 0 and os.path.isfile(path) and os.path.getsize(path) > 0) else None


# ── relay_mode ──────────────────────────────────────────────────────────
def test_relay_mode_default_video(monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_FLF_RELAY", raising=False)
    assert fe.relay_mode() == "video"


def test_relay_mode_env_override(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_FLF_RELAY", "keyframe")
    assert fe.relay_mode() == "keyframe"


# ── resolve_local_video ─────────────────────────────────────────────────
def test_resolve_local_video_none_for_falsy():
    assert fe.resolve_local_video(None) is None
    assert fe.resolve_local_video("") is None
    assert fe.resolve_local_video("C:/no/such/file.mp4") is None


def test_resolve_local_video_existing(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"x")
    assert fe.resolve_local_video(str(p)) == str(p)


def test_resolve_local_video_bad_media_url():
    assert fe.resolve_local_video("/media/definitely/missing.mp4") is None


# ── extract_last_frame ──────────────────────────────────────────────────
def test_extract_last_frame_missing_video():
    assert fe.extract_last_frame("C:/no/such.mp4") is None


def test_extract_last_frame_real(tmp_path):
    mp4 = _make_test_mp4(str(tmp_path / "clip.mp4"))
    if not mp4:
        pytest.skip("ffmpeg/lavfi 不可用，跳过真实抽取")
    url = fe.extract_last_frame(mp4, thread_id=_TH, tag="t1")
    assert url and url.startswith("/media/"), f"应产出 /media URL，得到 {url!r}"
    local = fe.media_url_to_local(url)
    assert local is not None and local.is_file(), "抽出的帧应能被 /media 反解析回真实文件"
    assert local.stat().st_size > 0


# ── _relay_frame_from_scene ─────────────────────────────────────────────
def test_relay_keyframe_mode_returns_keyframe(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_FLF_RELAY", "keyframe")
    scene = {"final_video_url": "C:/x.mp4", "last_image_url": "/media/kf.png"}
    assert graph._relay_frame_from_scene(scene, _TH, 1) == "/media/kf.png"


def test_relay_video_mode_extracts_real_last_frame(monkeypatch, tmp_path):
    monkeypatch.delenv("MULTIMEDIA_FLF_RELAY", raising=False)
    mp4 = _make_test_mp4(str(tmp_path / "clip.mp4"))
    if not mp4:
        pytest.skip("ffmpeg/lavfi 不可用，跳过真实抽取")
    scene = {"final_video_url": mp4, "last_image_url": "/media/kf.png"}
    got = graph._relay_frame_from_scene(scene, _TH, 0)
    assert got and "linked_last_" in got, f"video 模式应返回抽取的真实末帧，得到 {got!r}"


def test_relay_video_mode_falls_back_on_bad_video(monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_FLF_RELAY", raising=False)
    scene = {"final_video_url": "C:/no/such.mp4", "last_image_url": "/media/kf.png"}
    assert graph._relay_frame_from_scene(scene, _TH, 2) == "/media/kf.png"


def test_relay_video_mode_falls_back_to_image_url(monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_FLF_RELAY", raising=False)
    scene = {"last_image_url": "", "image_url": "/media/first.png"}
    assert graph._relay_frame_from_scene(scene, _TH, 3) == "/media/first.png"
