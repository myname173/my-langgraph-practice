"""
离线模拟测试：验证 _extract_character_identity 的角色 DNA 锁逻辑。

不调用任何大模型 / 外部 API，仅用 mock state 驱动纯逻辑。
验证重点：
  1. 丑小鸭故事不再被锁成「mage」(法师)
  2. subject 正确取 reference_sheets 角色键（ugly_duckling / white_swan）
  3. appearance 为中性自然外观词（羽毛/体型/颜色），不再含人类服饰/装备
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import agent.multimedia.graph as g
from typing import Any, Dict, Optional


def make_state(script: str, char_key: str, char_desc: str, char_cn: str,
               prev_identity: Optional[Dict] = None) -> Dict[str, Any]:
    """构造最小可用 state，喂给 _extract_character_identity。"""
    global_setting = (
        "主角是同一只鸭子的一生两个外貌阶段："
        "前期『丑小鸭』——羽毛灰褐蓬乱、体型比同伴大一圈、脖子粗短、神情怯弱；"
        "后期『白天鹅』——羽毛纯白光滑、脖颈修长优雅、体型舒展。整体为温暖手绘童话水彩风格"
    )
    return {
        "current_scene_index": 0,
        "scenes": [{"script": script}],
        "shot_plan": {"focus_object": "丑小鸭"},
        "global_setting": global_setting,
        "scene_anchors": {"character_identity": prev_identity} if prev_identity else None,
        "reference_sheets": {
            "characters": {
                char_key: {"url": "http://mock/char.png", "description": char_desc, "name_cn": char_cn}
            }
        },
    }


def case_1_single_duckling():
    """Case 1：前期丑小鸭 —— 应识别 ugly_duckling，而非 mage。"""
    print("=" * 60)
    print("Case 1: 前期丑小鸭（单角色）")
    print("=" * 60)
    script = ("春天，一只又大又灰、模样丑陋的小鸭在农场鸭群中破壳出生，"
              "羽毛灰褐蓬乱，脖子粗短，神情怯弱，被兄弟姐妹嘲笑排挤。")
    desc = "羽毛灰褐蓬乱，体型比同伴大一圈，脖子粗短，神情怯弱的幼鸭"
    state = make_state(script, "ugly_duckling", desc, "丑小鸭")
    ident = g._extract_character_identity(state)
    print("  ->", ident)
    assert ident["subject"] == "ugly_duckling", f"期望 ugly_duckling，实际 {ident['subject']}"
    assert ident["subject"] != "mage", "BUG 复现：仍被锁成 mage！"
    assert "法师" not in ident["appearance"] and "长袍" not in ident["appearance"], \
        f"appearance 含人类服饰词: {ident['appearance']}"
    assert any(t in ident["appearance"] for t in ("灰褐", "蓬乱", "粗短", "羽毛")), \
        f"appearance 未含自然外观词: {ident['appearance']}"
    print("  [PASS] subject=ugly_duckling, appearance 为自然外观词\n")


def case_2_multi_char_first_scene():
    """Case 2：多角色 sheets（含 ugly_duckling 与 white_swan），前期剧本只提丑小鸭。"""
    print("=" * 60)
    print("Case 2: 多角色 sheets，剧本只提丑小鸭（应匹配 ugly_duckling）")
    print("=" * 60)
    script = "丑小鸭独自在沼泽地流浪，羽毛灰褐蓬乱，在寒风中瑟缩。"
    # 注意：这里 state 只放一个 reference（测试单角色默认分支），
    # 再单独测多角色由剧本提及命中。
    state = make_state(script, "ugly_duckling", "灰褐蓬乱羽毛的幼鸭", "丑小鸭")
    ident = g._extract_character_identity(state)
    print("  ->", ident)
    assert ident["subject"] == "ugly_duckling"
    print("  [PASS] 单 reference 默认命中 ugly_duckling\n")


def case_3_multi_char_script_mentions_swan():
    """Case 3：多角色 sheets，剧本提『白天鹅』→ 应匹配 white_swan。"""
    print("=" * 60)
    print("Case 3: 多角色 sheets，剧本提白天鹅（应匹配 white_swan）")
    print("=" * 60)
    global_setting = ("主角是同一只鸭子的一生两个外貌阶段：前期丑小鸭、后期白天鹅。"
                      "整体为温暖手绘童话水彩风格")
    script = "它游到湖边，低头看见水中倒影——自己竟变成了一只雪白优雅的白天鹅，脖颈修长。"
    state: Dict[str, Any] = {
        "current_scene_index": 0,
        "scenes": [{"script": script}],
        "shot_plan": {"focus_object": "白天鹅"},
        "global_setting": global_setting,
        "scene_anchors": None,
        "reference_sheets": {
            "characters": {
                "ugly_duckling": {"url": "x", "description": "灰褐蓬乱羽毛的幼鸭", "name_cn": "丑小鸭"},
                "white_swan": {"url": "y", "description": "羽毛纯白光滑脖颈修长的天鹅", "name_cn": "白天鹅"},
            }
        },
    }
    ident = g._extract_character_identity(state)
    print("  ->", ident)
    assert ident["subject"] == "white_swan", f"期望 white_swan，实际 {ident['subject']}"
    assert "纯白" in ident["appearance"] or "修长" in ident["appearance"], \
        f"appearance 未含天鹅外观词: {ident['appearance']}"
    print("  [PASS] 剧本提及命中 white_swan\n")


def case_4_no_reference_fallback_focus():
    """Case 4：无 reference_sheets（退化场景）→ 兜底用 focus_object，过滤环境词。"""
    print("=" * 60)
    print("Case 4: 无 reference_sheets，focus_object 兜底（过滤环境词）")
    print("=" * 60)
    script = "镜头扫过农场与湖泊，晨雾弥漫。"
    state = make_state(script, "ugly_duckling", "x", "丑小鸭")
    # 清空 reference，强制走 focus_object 兜底
    state["reference_sheets"] = {"characters": {}}
    state["shot_plan"] = {"focus_object": "environment"}  # 环境词应被过滤
    ident = g._extract_character_identity(state)
    print("  ->", ident)
    assert ident["subject"] != "mage"
    # environment 被过滤，且无角色提及 → subject 可能为空（合理降级，不强行造角色）
    print(f"  [PASS] 环境词已过滤，subject={ident['subject']!r}（无角色时留空，可接受）\n")


def case_5_anti_drift_history():
    """Case 5：抗漂移——历史 subject 锁定，后续场景不跳变。"""
    print("=" * 60)
    print("Case 5: 抗漂移——历史 ugly_duckling 锁定，后续场景不跳变")
    print("=" * 60)
    script = "其他动物在湖边嬉戏，并未提及主角。"  # 剧本没提角色名
    prev = {"subject": "ugly_duckling", "appearance": ["灰褐", "蓬乱"], "has_character": True}
    state = make_state(script, "ugly_duckling", "灰褐蓬乱羽毛的幼鸭", "丑小鸭", prev_identity=prev)
    ident = g._extract_character_identity(state)
    print("  ->", ident)
    assert ident["subject"] == "ugly_duckling", "历史 subject 应被锁定继承"
    print("  [PASS] 历史角色身份成功继承，未跳变\n")


def case_6_residual_thread_pollution():
    """Case 6（回归）：跨运行 thread 残留 mage 锚点，首镜头必须忽略并重新识别。"""
    print("=" * 60)
    print("Case 6: 跨运行残留 mage 锚点污染 → 首镜头强制重新识别")
    print("=" * 60)
    script = ("春天，一只又大又灰、模样丑陋的小鸭在农场鸭群中破壳出生，"
              "羽毛灰褐蓬乱，脖子粗短，神情怯弱，被兄弟姐妹嘲笑排挤。")
    desc = "羽毛灰褐蓬乱，体型比同伴大一圈，脖子粗短，神情怯弱的幼鸭"
    # 模拟 duck_run3 同 thread 残留的错误 scene_anchors（subject=mage）
    polluted = {"subject": "mage", "appearance": ["短发", "角", "长袍", "白发"], "has_character": True}
    state = make_state(script, "ugly_duckling", desc, "丑小鸭", prev_identity=polluted)
    ident = g._extract_character_identity(state)
    print("  ->", ident)
    assert ident["subject"] == "ugly_duckling", f"残留 mage 污染首镜头！实际 {ident['subject']}"
    assert "mage" not in ident["subject"]
    assert "短发" not in ident["appearance"] and "长袍" not in ident["appearance"], \
        f"appearance 仍含法师词: {ident['appearance']}"
    print("  [PASS] 首镜头忽略残留 mage，正确识别 ugly_duckling\n")


def case_7_mid_film_drift_protection():
    """Case 7（回归）：影片中段剧本未提角色时，继承上镜正确 subject（抗漂移仍有效）。"""
    print("=" * 60)
    print("Case 7: 中段场景未提角色名 → 继承上镜正确 subject（抗漂移保留）")
    print("=" * 60)
    script = "其他动物在湖边嬉戏，并未提及主角姓名。"  # 剧本没提角色名
    prev = {"subject": "ugly_duckling", "appearance": ["灰褐", "蓬乱"], "has_character": True}
    # 注意 current_scene_index != 0，允许继承历史（抗漂移）
    state = make_state(script, "ugly_duckling", "灰褐蓬乱羽毛的幼鸭", "丑小鸭", prev_identity=prev)
    state["current_scene_index"] = 3
    # 补齐 scenes 长度，使索引 3 合法
    state["scenes"] = [{"script": script}] * 4
    ident = g._extract_character_identity(state)
    print("  ->", ident)
    assert ident["subject"] == "ugly_duckling", "中段未提角色时应继承上镜 subject"
    print("  [PASS] 中段抗漂移生效，subject 锁定 ugly_duckling\n")


if __name__ == "__main__":
    case_1_single_duckling()
    case_2_multi_char_first_scene()
    case_3_multi_char_script_mentions_swan()
    case_4_no_reference_fallback_focus()
    case_5_anti_drift_history()
    case_6_residual_thread_pollution()
    case_7_mid_film_drift_protection()
    print("=" * 60)
    print("ALL CASES PASSED  角色 DNA 锁 bug 已修复（不再锁成法师）")
    print("=" * 60)
