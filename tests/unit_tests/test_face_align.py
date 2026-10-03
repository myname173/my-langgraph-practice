"""P1-4 前置：mediapipe 人脸对齐 + 比较前对齐的零额度单测（全部 mock，无网络）。"""
import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

fa = pytest.importorskip("agent.multimedia.tools.face_align")
ve = pytest.importorskip("agent.multimedia.tools.vision_eval")


class _EmptyResult:
    face_landmarks = []


class _FakeLandmarker:
    def detect(self, img):
        return _EmptyResult()


@pytest.fixture
def no_landmarker(monkeypatch):
    monkeypatch.setattr(fa, "_get_landmarker", lambda: None)


@pytest.fixture
def empty_landmarker(monkeypatch):
    monkeypatch.setattr(fa, "_get_landmarker", lambda: _FakeLandmarker())


# ── align_face 降级路径 ──
def test_align_no_landmarker(no_landmarker):
    r = fa.align_face("x.png")
    assert r["available"] is False and r["reason"] == "mediapipe 不可用"


def test_align_no_face(empty_landmarker, tmp_path):
    from PIL import Image
    p = tmp_path / "t.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(p)
    r = fa.align_face(str(p))
    assert r["available"] is False and "未检出人脸" in r["reason"]


def test_align_missing_file(empty_landmarker):
    r = fa.align_face("/no/such/file_xyz.png")
    assert r["available"] is False


def test_align_empty_input():
    r = fa.align_face("")
    assert r["available"] is False


# ── _prepare_for_compare 分支 ──
def test_prepare_disabled(monkeypatch):
    monkeypatch.setattr(ve, "_FACE_ALIGN_ENABLED", False)
    u, al = ve._prepare_for_compare("http://x/y.png")
    assert u == "http://x/y.png" and al is False


def test_prepare_aligned(monkeypatch):
    monkeypatch.setattr(ve, "_FACE_ALIGN_ENABLED", True)
    monkeypatch.setattr(fa, "align_face", lambda *a, **k: {"available": True, "aligned_data_url": "data:image/jpeg;base64,AAA"})
    u, al = ve._prepare_for_compare("http://x/y.png")
    assert u.startswith("data:image/jpeg") and al is True


def test_prepare_fallback_when_align_unavailable(monkeypatch):
    monkeypatch.setattr(ve, "_FACE_ALIGN_ENABLED", True)
    monkeypatch.setattr(fa, "align_face", lambda *a, **k: {"available": False, "reason": "未检出人脸"})
    u, al = ve._prepare_for_compare("http://x/y.png")
    assert u == "http://x/y.png" and al is False


def test_prepare_empty_input(monkeypatch):
    monkeypatch.setattr(ve, "_FACE_ALIGN_ENABLED", True)
    _, al = ve._prepare_for_compare("")
    assert al is False
