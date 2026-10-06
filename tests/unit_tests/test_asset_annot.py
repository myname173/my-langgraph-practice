"""P1 资产标注（零额度单测）。

覆盖「分镜生成阶段标注每镜出场资产 + 匹配直读结构化提示」这一改动的全部新增能力：

- `_scene_base` 默认携带 `assets` 字段；
- `_extract_scene_assets` 对总导演输出的归一化（去空 / 去重 / 非 list 兜底）；
- `_desc_tokens` 中文按字符 2-gram 切分（修复「中文描述整段无空格导致永不命中」）；
- `_check_match` 结构化提示（scene["assets"]）优先命中，且不破坏原有字面匹配行为；
- 中文 desc 需 >=2 个 bigram 命中（修复同时抑制误判）；英文 desc 任一命中（保持原行为）；
- `_match_reference_elements` 的 `assets_hint` 接线，及无匹配时仍返回 None 触发既有回退；
- `_build_asset_match_baseline` 把 scene["assets"] 纳入基线统计。

不触发生图 / 生视频，不消耗任何额度。
"""
import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

common = pytest.importorskip("agent.multimedia.nodes.common")
scripting = pytest.importorskip("agent.multimedia.nodes.scripting")
design = pytest.importorskip("agent.multimedia.nodes.design")


def _sheets():
    return {
        "characters": {
            "char_main": {"urls": ["u_main"], "description": "白衣男剑仙"},
            "char_villain": {"urls": ["u_vill"], "description": "黑袍魔尊"},
        },
        "props": {"jade": {"urls": ["u_jade"], "description": "古玉"}},
        "environments": {"cliff": {"urls": ["u_cliff"], "description": "断崖"}},
    }


# ── 1. _scene_base 默认字段 ─────────────────────────────────────────────
def test_scene_base_has_assets_default():
    sb = common._scene_base("测试脚本")
    assert "assets" in sb
    assert sb["assets"] == []


# ── 2. _extract_scene_assets 归一化 ─────────────────────────────────────
def test_extract_assets_non_list_returns_empty():
    assert scripting._extract_scene_assets("主角A") == []
    assert scripting._extract_scene_assets(None) == []
    assert scripting._extract_scene_assets({"a": 1}) == []


def test_extract_assets_strip_dedup_and_drop_empty():
    assert scripting._extract_scene_assets(
        [" 主角A ", "", None, "主角A", "玉佩", "   "]
    ) == ["主角A", "玉佩"]


def test_extract_assets_numeric_items():
    assert scripting._extract_scene_assets([1, 2, 1]) == ["1", "2"]


# ── 3. _desc_tokens 中文 bigram ─────────────────────────────────────────
def test_desc_tokens_cjk_bigrams():
    cn, en = design._desc_tokens("白衣仙尊 warrior_king")
    assert "白衣" in cn and "仙尊" in cn
    assert "warrior" in en and "king" in en


def test_desc_tokens_single_cjk_char_no_bigram():
    cn, _ = design._desc_tokens("剑")
    assert cn == []


def test_desc_tokens_cjk_count():
    cn, _ = design._desc_tokens("腰悬玉佩")
    assert len(cn) == 3


# ── 4. _check_match 结构化提示优先命中 ─────────────────────────────────
def test_check_match_hint_equals_key():
    sh = _sheets()
    assert design._check_match("char_main", sh["characters"]["char_main"], "他缓步上前。", ["char_main"])


def test_check_match_hint_contains_key():
    sh = _sheets()
    assert design._check_match("char_main", sh["characters"]["char_main"], "他缓步上前。", ["char_main.jpg"])


def test_check_match_key_contains_hint():
    sh = _sheets()
    assert design._check_match("char_main", sh["characters"]["char_main"], "他缓步上前。", ["main"])


def test_check_match_no_hint_and_no_literal():
    sh = _sheets()
    assert not design._check_match("char_main", sh["characters"]["char_main"], "他缓步上前。", None)


# ── 5. _check_match 保留原有字面行为 ────────────────────────────────────
def test_check_match_literal_full_name():
    sh = _sheets()
    assert design._check_match("char_main", sh["characters"]["char_main"], "char_main 出现。", None)


def test_check_match_literal_word_level():
    sh = _sheets()
    assert design._check_match("char_main", sh["characters"]["char_main"], "main 角色登场。", None)


# ── 6. _check_match 中文 desc 修复 ─────────────────────────────────────
def test_check_match_cn_desc_two_bigrams_hit():
    data = {"urls": ["u"], "description": "白衣男剑仙"}
    assert design._check_match("mystery_char", data, "白衣剑仙立于崖边。", None)


def test_check_match_cn_desc_single_bigram_miss():
    data = {"urls": ["u"], "description": "白衣男剑仙"}
    assert not design._check_match("mystery_char", data, "白衣缓缓飘动。", None)


def test_check_match_en_desc_any_hit_preserved():
    data = {"urls": ["u"], "description": "silver hair warrior"}
    assert design._check_match("m2", data, "the warrior stands.", None)


# ── 7. _match_reference_elements 带 assets_hint ────────────────────────
def test_match_hint_hits_character():
    r = design._match_reference_elements(_sheets(), "他缓步上前，四下无人。", assets_hint=["char_main"])
    assert r["char_urls"] == ["u_main"]
    assert r["matched_char_name"] == "char_main"


def test_match_hint_hits_other_character():
    r = design._match_reference_elements(_sheets(), "他缓步上前，四下无人。", assets_hint=["char_villain"])
    assert r["matched_char_name"] == "char_villain"


def test_match_hint_no_match_returns_none():
    r = design._match_reference_elements(_sheets(), "他缓步上前，四下无人。", assets_hint=["不存在的资产"])
    assert r["char_urls"] is None


def test_match_without_hint_falls_back_to_literal():
    r = design._match_reference_elements(_sheets(), "cliff 断崖边。")
    assert r["env_urls"] == ["u_cliff"]


def test_match_hint_hits_prop():
    r = design._match_reference_elements(_sheets(), "毫无关联的话。", assets_hint=["jade"])
    assert r["matched_prop_name"] == "jade"


def test_match_signature_backward_compatible():
    """位置参数 (reference_sheets, script_text) 保持不变，assets_hint 为可选。"""
    r = design._match_reference_elements(_sheets(), "char_main 登场。")
    assert r["matched_char_name"] == "char_main"


# ── 8. _scene_asset_hints 收集 ─────────────────────────────────────────
def test_scene_asset_hints_dedup_across_scenes():
    hints = design._scene_asset_hints([
        {"assets": ["A", "B"]},
        {"assets": ["B", "C"]},
        {"script": "x"},
    ])
    assert hints == ["A", "B", "C"]


# ── 9. _build_asset_match_baseline 纳入 scene assets ───────────────────
def test_baseline_uses_scene_assets():
    scenes = [
        {"script": "他缓步上前。", "assets": ["char_main"]},
        {"script": "风起。", "assets": []},
    ]
    rep = design._build_asset_match_baseline(_sheets(), scenes)
    assert "char_main" in rep["matched"]
    assert "char_villain" in rep["unmatched"]
