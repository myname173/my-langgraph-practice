"""P1-4：VLM 成对一致性判断（结构化输出）+ 强制层路由 零额度单测。

验证 ``assess_character_consistency`` 的结构化解析/容错/降级，以及
``decide_image_quality`` 中 VLM 一致性强制层的分流条件。全部 mock，无网络。
"""
import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ve = pytest.importorskip("agent.multimedia.tools.vision_eval")
graph = pytest.importorskip("agent.multimedia.graph")


class _Msg:
    def __init__(self, c): self.content = c


class _Choice:
    def __init__(self, c): self.message = _Msg(c)


class _Resp:
    def __init__(self, c): self.choices = [_Choice(c)]


class _Completions:
    def __init__(self, text=None, exc=None): self._t = text; self._e = exc

    def create(self, **k):
        if self._e:
            raise self._e
        return _Resp(self._t)


class _Chat:
    def __init__(self, c): self.completions = c


class _Client:
    def __init__(self, text=None, exc=None): self.chat = _Chat(_Completions(text, exc))


@pytest.fixture
def set_vlm(monkeypatch):
    def _set(text=None, exc=None):
        monkeypatch.setattr(ve, "client", _Client(text, exc))
    return _set


# ── _extract_first_json ──
@pytest.mark.parametrize("raw,expect", [
    ('{"a":1}', {"a": 1}),
    ('```json\n{"same_character": true, "score": 0.8}\n```', {"same_character": True, "score": 0.8}),
    ('好的，结果是 {"x": 2} 完毕', {"x": 2}),
    ("no json here", None),
    ("", None),
])
def test_extract_first_json(raw, expect):
    assert ve._extract_first_json(raw) == expect


# ── assess_character_consistency ──
def test_assess_same(set_vlm):
    set_vlm('{"same_character": true, "score": 0.9, "reason": "一致"}')
    r = ve.assess_character_consistency("a.png", "b.png")
    assert r["available"] and r["same_character"] is True
    assert abs(r["score"] - 0.9) < 1e-6 and r["drifting"] is False


def test_assess_drifting(set_vlm):
    set_vlm('{"same_character": false, "score": 0.2, "reason": "不同"}')
    r = ve.assess_character_consistency("a.png", "b.png", threshold=0.6)
    assert r["drifting"] is True


def test_assess_low_score_but_same_not_drifting(set_vlm):
    set_vlm('{"same_character": true, "score": 0.3, "reason": "x"}')
    r = ve.assess_character_consistency("a.png", "b.png", threshold=0.6)
    assert r["drifting"] is False


def test_assess_score_0_100_normalized(set_vlm):
    set_vlm('{"same_character": true, "score": 85, "reason": "x"}')
    r = ve.assess_character_consistency("a.png", "b.png")
    assert abs(r["score"] - 0.85) < 1e-6


def test_assess_non_json(set_vlm):
    set_vlm("这不是 JSON")
    r = ve.assess_character_consistency("a.png", "b.png")
    assert r["available"] is True and r["score"] is None and r["drifting"] is False


def test_assess_exception_degrades(set_vlm):
    set_vlm(exc=RuntimeError("boom"))
    r = ve.assess_character_consistency("a.png", "b.png")
    assert r["available"] is False and r["drifting"] is False


def test_assess_empty_input():
    r = ve.assess_character_consistency("", "b.png")
    assert r["available"] is False


# ── decide_image_quality 强制层 ──
def _vstate(passed=True, iters=0, vlm=None, needs=False, reviewer_un=False, psim=None):
    sc = {"is_perfect": passed, "iterations": iters, "image_url": "kf.png",
          "portrait_similarity": psim, "script": "a man walks alone",
          "last_image_is_perfect": False, "last_image_iterations": 0}
    if vlm is not None:
        sc["vlm_consistency"] = vlm
    if needs:
        sc["needs_realign"] = True
    st = {"current_scene_index": 0, "scenes": [sc], "use_first_last_frame": True}
    if reviewer_un:
        st["reviewer_unavailable"] = True
    return st


VLM_DRIFT = {"available": True, "same_character": False, "score": 0.1, "drifting": True, "method": "vlm_pair", "threshold": 0.6}
VLM_OK = {"available": True, "same_character": True, "score": 0.9, "drifting": False, "method": "vlm_pair", "threshold": 0.6}


def test_vlm_drift_triggers_regen():
    assert graph.decide_image_quality(_vstate(vlm=VLM_DRIFT, needs=True)) == "director"


def test_vlm_ok_passes():
    assert graph.decide_image_quality(_vstate(vlm=VLM_OK, needs=False)) == "end_frame_director"


def test_vlm_drift_reviewer_unavailable_passes():
    assert graph.decide_image_quality(_vstate(vlm=VLM_DRIFT, needs=True, reviewer_un=True)) == "end_frame_director"


def test_vlm_drift_iterations_exhausted_passes():
    # 一致性重试预算耗尽（consistency_attempts >= 3）→ VLM 判 drift 也放行到尾帧导演。
    # P1-4 起该预算由 consistency_attempts 独立计数，不再复用 iterations。
    st = _vstate(vlm=VLM_DRIFT, needs=True, iters=3)
    st["scenes"][0]["consistency_attempts"] = 3
    assert graph.decide_image_quality(st) == "end_frame_director"


def test_vlm_branch_skipped_when_embedding_present():
    assert graph.decide_image_quality(_vstate(vlm=VLM_DRIFT, needs=True, psim=0.9)) == "end_frame_director"


def test_no_vlm_passes():
    assert graph.decide_image_quality(_vstate()) == "end_frame_director"
