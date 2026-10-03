# tests/unit_tests/test_review_flywheel.py
"""P1-6 审核飞轮单测：偏好对配对、统计、导出、蒸馏与影子建议。

运行（需 pytest；本机若未装 pytest，可用 tests/unit_tests 同级断言脚本等价验证）：
    pytest tests/unit_tests/test_review_flywheel.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture()
def fw(tmp_path, monkeypatch):
    """把飞轮重定向到临时目录，隔离真实训练数据。"""
    monkeypatch.setenv("MULTIMEDIA_FLYWHEEL_DIR", str(tmp_path))
    monkeypatch.setenv("MULTIMEDIA_FLYWHEEL_ENABLED", "1")
    from agent.multimedia import review_flywheel as mod
    mod._DIR = str(tmp_path)
    mod._PATH = str(tmp_path / "multimedia_review_pairs.jsonl")
    mod._POLICY_PATH = str(tmp_path / "review_policy.json")
    mod._POLICY_CACHE = None
    mod._AUTO_REVIEW = "off"
    mod.reset_pending()
    return mod


def _prompt_pair(fw, gate, rejected_text, chosen_text, **kw):
    return fw.record_decision(
        gate,
        {"action": "edit_prompt", "image_prompt": chosen_text},
        rejected=fw.artifact_from_payload(gate, {"image_prompt": rejected_text}),
        **kw,
    )


# ---------------------------------------------------------------- 产物归一
def test_artifact_from_payload_prompt_gate(fw):
    a = fw.artifact_from_payload("prompt_preview", {"image_prompt": "  x  "})
    assert a["kind"] == "prompt" and a["text"] == "x"


def test_artifact_from_payload_image_gate(fw):
    a = fw.artifact_from_payload(
        "image_review",
        {"image_prompt": "p", "image_url": "u", "portrait_similarity": 0.5},
    )
    assert a["kind"] == "image" and a["url"] == "u"
    assert a["meta"]["portrait_similarity"] == 0.5


# ---------------------------------------------------------------- 收集端
def test_edit_creates_pair_immediately(fw):
    p = _prompt_pair(fw, "prompt_preview", "一个人站在山上", "道士立于山巅", thread_id="t1", scene_index=0)
    assert p is not None and p["source"] == "edit"
    assert p["rejected"]["text"] == "一个人站在山上"
    assert p["chosen"]["text"] == "道士立于山巅"


def test_rewrite_then_approve_pairs(fw):
    fw.reset_pending()
    a = fw.record_decision(
        "image_review", {"action": "rewrite", "reason": "比例失真"},
        thread_id="t2", scene_index=1,
        rejected={"kind": "image", "text": "", "url": "bad.png", "meta": {}},
    )
    assert a is None and fw.pending_count() == 1
    b = fw.record_decision(
        "image_review", {"action": "approve"},
        thread_id="t2", scene_index=1,
        rejected={"kind": "image", "text": "", "url": "good.png", "meta": {}},
    )
    assert b is not None and b["source"] == "rewrite"
    assert b["rejected"]["url"] == "bad.png" and b["chosen"]["url"] == "good.png"
    assert b["decision"]["reason"] == "比例失真"
    assert fw.pending_count() == 0


def test_plain_approve_no_pair(fw):
    assert fw.record_decision("video_review", {"action": "approve"},
                              rejected={"kind": "video", "text": "", "url": "v", "meta": {}}) is None


def test_auto_approved_no_pair(fw):
    fw.reset_pending()
    fw.record_decision("prompt_preview", {"action": "rewrite"}, thread_id="t", scene_index=0,
                       rejected={"kind": "prompt", "text": "a", "url": "", "meta": {}})
    assert fw.pending_count() == 1
    assert fw.record_decision("prompt_preview", {"action": "approve", "auto_approved": True},
                              thread_id="t", scene_index=0,
                              rejected={"kind": "prompt", "text": "b", "url": "", "meta": {}}) is None
    assert fw.pending_count() == 0


def test_pending_keyed_by_thread_gate_scene(fw):
    fw.reset_pending()
    fw.record_decision("image_review", {"action": "rewrite"}, thread_id="A", scene_index=0,
                       rejected={"kind": "image", "text": "", "url": "1", "meta": {}})
    fw.record_decision("image_review", {"action": "rewrite"}, thread_id="B", scene_index=0,
                       rejected={"kind": "image", "text": "", "url": "2", "meta": {}})
    assert fw.pending_count() == 2
    # 只配对 (A, image_review, 0)
    fw.record_decision("image_review", {"action": "approve"}, thread_id="A", scene_index=0,
                       rejected={"kind": "image", "text": "", "url": "1b", "meta": {}})
    assert fw.pending_count() == 1


# ---------------------------------------------------------------- 落盘 / 统计
def test_persisted_and_stats(fw):
    _prompt_pair(fw, "prompt_preview", "a1", "b1", thread_id="t1", scene_index=0)
    fw.reset_pending()
    a = _prompt_pair(fw, "prompt_preview", "a2", "b2", thread_id="t1", scene_index=1)
    assert a is not None
    pairs = fw._read_pairs()
    assert len(pairs) == 2
    assert all(p["schema"] == fw.SCHEMA for p in pairs)
    st = fw.summarize()
    assert st["total_pairs"] == 2 and st["by_source"]["edit"] == 2


# ---------------------------------------------------------------- 导出
def test_export_dpo_only_textual(fw):
    _prompt_pair(fw, "prompt_preview", "old prompt", "new prompt", thread_id="t", scene_index=0)
    fw.reset_pending()
    fw.record_decision("image_review", {"action": "edit_prompt", "image_prompt": ""},
                       thread_id="t", scene_index=1,
                       rejected={"kind": "image", "text": "", "url": "u", "meta": {}})
    out = fw.export_dpo()
    rows = [json.loads(l) for l in Path(out).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1  # 图片类被跳过
    assert rows[0]["chosen"] == "new prompt" and rows[0]["rejected"] == "old prompt"


def test_export_sft(fw):
    _prompt_pair(fw, "prompt_preview", "old", "new", thread_id="t", scene_index=0)
    out = fw.export_sft()
    rows = [json.loads(l) for l in Path(out).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["output"] == "new"


# ---------------------------------------------------------------- 蒸馏 + 影子
def _load_distiller():
    spec = importlib.util.spec_from_file_location(
        "distill_reviewer", str(ROOT / "scripts" / "distill_reviewer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_distill_learns_tokens_and_threshold():
    dr = _load_distiller()
    good = ["道士立于山巅青衫猎猎远景", "道士立于云海长剑出鞘剪影", "道士立于山巅衣袂翻飞广角"]
    bad = ["一个人站在山上", "一个人站着", "有个人在山上站着"]
    pairs = [{
        "gate": "prompt_preview",
        "rejected": {"kind": "prompt", "text": b},
        "chosen": {"kind": "prompt", "text": g},
    } for g, b in zip(good, bad)]
    policy = dr.distill(pairs, min_pairs=3)
    assert "prompt_preview" in policy["gates"]
    gm = policy["gates"]["prompt_preview"]
    assert gm["chosen_tokens"] and gm["rejected_tokens"]


def test_suggest_pass_flag(fw, monkeypatch):
    dr = _load_distiller()
    good = ["道士立于山巅青衫猎猎远景", "道士立于云海长剑出鞘剪影", "道士立于山巅衣袂翻飞广角"]
    bad = ["一个人站在山上", "一个人站着", "有个人在山上站着"]
    pairs = [{"gate": "prompt_preview",
              "rejected": {"kind": "prompt", "text": b},
              "chosen": {"kind": "prompt", "text": g}} for g, b in zip(good, bad)]
    policy = dr.distill(pairs, min_pairs=3)
    Path(fw._POLICY_PATH).write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")
    fw._POLICY_CACHE = None
    assert fw.suggest("prompt_preview", {"kind": "prompt", "text": good[0]})["decision"] == "pass"
    assert fw.suggest("prompt_preview", {"kind": "prompt", "text": bad[0]})["decision"] == "flag"


def test_suggest_returns_none_without_policy(fw):
    assert fw.suggest("prompt_preview", {"kind": "prompt", "text": "x"}) is None


# ---------------------------------------------------------------- 影子开关
def test_auto_review_switch(fw):
    fw._AUTO_REVIEW = "off"
    assert fw.auto_review_enabled("image_review") is False
    fw._AUTO_REVIEW = "shadow"
    assert fw.auto_review_enabled("image_review") is True
    fw._AUTO_REVIEW = "image_review"
    assert fw.auto_review_enabled("image_review") and not fw.auto_review_enabled("video_review")
