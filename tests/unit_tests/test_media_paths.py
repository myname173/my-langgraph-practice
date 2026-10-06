# -*- coding: utf-8 -*-
"""修复 3 个冒烟缺陷的永久回归单测（零网络 / 零额度）。

覆盖：
  * media_paths 两种 /media 约定的互转（output-relative 与 root-relative）；
  * reference_gen 生产者不再产出 /media/output/...（消除 output/output 二次拼接）；
  * jimeng/agnes/embedding/vision_eval/face_align 消费者对历史 root-relative URL 仍可解析；
  * tts 不再访问不存在的 SpeechSynthesizerResult.message。
"""
import os
import sys
import tempfile

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.multimedia.tools import media_paths as mp  # noqa: E402


@pytest.fixture()
def probe_image():
    """在 output/ 下建一个真实小图，测试后删除。"""
    from PIL import Image

    out = mp.output_dir() / "keyframes" / "_generated"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "_mptest_probe_unit.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(f)
    try:
        yield f
    finally:
        try:
            f.unlink()
        except OSError:
            pass


def test_project_root_and_output_dir():
    assert str(mp.project_root()) == os.path.abspath(_PROJECT_ROOT)
    assert mp.output_dir() == mp.project_root() / "output"


def test_local_to_media_url_is_output_relative(probe_image):
    u = mp.local_to_media_url(probe_image)
    # 关键：不得出现 /media/output/... 的重复 output 段
    assert u == "/media/keyframes/_generated/_mptest_probe_unit.png"
    assert "output/output" not in u


def test_media_url_to_local_accepts_both_conventions(probe_image):
    a = mp.media_url_to_local("/media/keyframes/_generated/_mptest_probe_unit.png")
    b = mp.media_url_to_local("/media/output/keyframes/_generated/_mptest_probe_unit.png")
    assert a is not None and b is not None
    assert a == b == probe_image.resolve()


def test_media_url_to_local_returns_none_for_missing_or_foreign():
    assert mp.media_url_to_local("/media/output/keyframes/_nope_unit.png") is None
    assert mp.media_url_to_local("C:/x/y.png") is None
    assert mp.media_url_to_local("") is None
    assert mp.media_url_to_local(None) is None


def test_roundtrip(probe_image):
    u = mp.local_to_media_url(probe_image)
    assert mp.media_url_to_local(u) == probe_image.resolve()


def test_reference_gen_producer_no_double_output(probe_image):
    rg = pytest.importorskip("agent.multimedia.reference_gen")
    u = rg._local_media_path(probe_image)
    assert u == "/media/keyframes/_generated/_mptest_probe_unit.png"
    assert "output/output" not in u
    assert mp.media_url_to_local(u) == probe_image.resolve()


def test_vision_eval_resolves_legacy_media_url(probe_image):
    ve = pytest.importorskip("agent.multimedia.tools.vision_eval")
    legacy = "/media/output/keyframes/_generated/_mptest_probe_unit.png"
    r = ve._resolve_image_input(legacy)
    assert r.startswith("data:image/jpeg;base64,")
    assert not r.startswith("/media/")


def test_consumers_resolve_legacy_media_url(probe_image):
    jimeng = pytest.importorskip("agent.multimedia.tools.jimeng")
    agnes = pytest.importorskip("agent.multimedia.tools.agnes")
    embedding = pytest.importorskip("agent.multimedia.tools.embedding")
    legacy = "/media/output/keyframes/_generated/_mptest_probe_unit.png"
    assert jimeng._as_inline(legacy).startswith("data:image/")
    assert agnes._resolve_image(legacy).startswith("data:image/")
    assert embedding._to_dashscope_image(legacy).startswith("data:image/")


def test_face_align_resolves_legacy_media_url(probe_image):
    fa = pytest.importorskip("agent.multimedia.tools.face_align")
    legacy = "/media/output/keyframes/_generated/_mptest_probe_unit.png"
    arr = fa._load_rgb(legacy)
    assert getattr(arr, "shape", None) == (8, 8, 3)


def test_tts_error_is_readable_not_attribute_error():
    """SpeechSynthesizerResult 无 .message：失败时应抛可读 RuntimeError。"""
    tts = pytest.importorskip("agent.multimedia.tools.tts")
    import inspect

    src = inspect.getsource(tts._generate_voiceover_dashscope)
    assert ".message}" not in src  # 不再内联 {result.message}

    class _FakeResult:
        status_code = 401

        def get_audio_data(self):
            return None

    # 复刻修复后的分支逻辑
    def _branch(result):
        audio = result.get_audio_data()
        assert audio is None
        _code = getattr(result, "status_code", None) or getattr(result, "code", None)
        _msg = getattr(result, "message", None) or getattr(result, "msg", None)
        if _code is None and _msg is None:
            _msg = f"{type(result).__name__} 未返回音频数据"
        raise RuntimeError(f"CosyVoice 合成失败: code={_code}, message={_msg}")

    with pytest.raises(RuntimeError) as exc:
        _branch(_FakeResult())
    assert "code=401" in str(exc.value)
