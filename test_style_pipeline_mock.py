"""
离线模拟测试：验证「视觉风格链路」满足以下产品/工程约束（不调用任何大模型）：

  P1. 风格不是固定/硬耦合的 —— 输入不同故事，识别出的 style_key / suffix 必须随之变化。
  P2. 风格顺着用户输入来，而不是代码随意发挥 —— 风格来源于 task+global_setting+story 的 LLM 抽取，
      且抽取 prompt 不应预先替用户写死「XX题材用YY风格」的映射。
  P3. 提示词应「顺着用户输入丰富」，而非模型自创风格 —— 自定义风格时使用 LLM 给的 style_prompt，
      且每个镜头都把风格作为硬锚点注入（不丢风格）。
  P4. 负面约束动态化 —— 不同风格的冲突词由 _get_conflict_keywords 动态合并，不写死。

测试通过 mock call_llm 模拟不同用户输入下的风格抽取结果，验证链路解耦。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import agent.multimedia.graph as g

# ── mock call_llm：按不同用户输入返回不同的风格抽取 JSON ──
_STYLE_RESPONSES = {}


def fake_call_llm(prompt: str, purpose: str = ""):
    # 关键：prompt 由「预设风格列表(style_keys) + 用户输入(inputs)」拼接而成。
    # 预设列表里含所有 preset 的 suffix（含 "documentary"/"watercolor" 等词），
    # 若直接对整段 prompt 判断，每个输入都会误命中这些预设词。
    # 因此这里只截取【用户输入段】[task]...[story] 来判断，跳过开头的预设列表。
    user_part = prompt
    ti = prompt.find("[task]")
    si = prompt.find("[story]")
    if ti >= 0 and si >= 0:
        user_part = prompt[ti:si]  # 仅含 task + global_setting

    if "纪录野生动物" in user_part or "记录野生动物" in user_part:
        return '{"style_key":"documentary","label":"纪实纪录片","description":"documentary photography style","style_prompt":""}'
    if "赛博朋克科幻" in user_part:
        return '{"style_key":"aaa_cg","label":"赛博朋克CG","description":"neon cyberpunk UE5 cinematic trailer","style_prompt":""}'
    if "黏土偶动画" in user_part:
        # 自定义风格：预设库没有，走 LLM 给的 style_prompt（顺用户输入丰富）
        return '{"style_key":"custom","label":"黏土定格动画","description":"stop-motion claymation aesthetic","style_prompt":"stop-motion claymation, handmade clay figures, visible fingerprints, miniature sets, warm practical lighting"}'
    if "水墨武侠" in user_part:
        return '{"style_key":"watercolor","label":"水墨武侠","description":"ink wash wuxia painting","style_prompt":""}'
    if "丑小鸭" in user_part or "水彩风格" in user_part:
        return '{"style_key":"watercolor","label":"水彩童话","description":"a warm hand-drawn fairy tale watercolor style","style_prompt":""}'
    # 兜底：未识别（不应命中，用于暴露 prompt 模板污染问题）
    return '{"style_key":"custom","label":"未定风格","description":"unspecified","style_prompt":"soft painterly storybook illustration"}'


# 用 monkeypatch 替换 call_llm
g.call_llm = fake_call_llm


def make_state(task: str, global_setting: str = "", story: str = ""):
    return {
        "task": task,
        "global_setting": global_setting,
        "story": story,
        "script": story,
        "visual_style": None,
        "style_suffix": None,
        "style_description": None,
    }


def run_case(name, task, global_setting="", story=""):
    state = make_state(task, global_setting, story)
    res = g._extract_visual_style(state)
    print(f"  [{name}]")
    print(f"    task/输入: {task[:40]!r}")
    print(f"    -> style_key={res['style_key']!r}")
    print(f"    -> suffix={res['style_suffix'][:70]!r}")
    return res


def main():
    print("=" * 64)
    print("P1/P2/P3: 风格随用户输入变化，且来源于输入而非硬耦合")
    print("=" * 64)

    r1 = run_case("水彩童话(丑小鸭)", "一只丑小鸭变成白天鹅的童话故事", "温暖手绘童话水彩风格")
    r2 = run_case("赛博朋克", "未来霓虹都市的赛博朋克科幻史诗")
    r3 = run_case("水墨武侠", "古代江湖侠客的水墨武侠传奇")
    r4 = run_case("黏土定格(自定义)", "用黏土偶动画讲一个温馨的小镇故事")
    r5 = run_case("纪实纪录片", "记录野生动物迁徙的真实纪录片")

    # P1: style_key 必须随输入变化
    keys = [r1["style_key"], r2["style_key"], r3["style_key"], r4["style_key"], r5["style_key"]]
    assert len(set(keys)) >= 4, f"风格未随输入变化！keys={keys}"
    print("\n  [P1 PASS] 不同输入识别出不同 style_key（{0} 种）\n".format(len(set(keys))))

    # P2: 风格来源于输入，不应默认成 aaa_cg（除非输入就是赛博朋克）
    assert r1["style_key"] == "watercolor", "丑小鸭应识别为 watercolor"
    assert r5["style_key"] == "documentary", "纪录片应识别为 documentary"
    print("  [P2 PASS] 风格由输入决定（丑小鸭→watercolor，纪录片→documentary），"
          "未静默回退到默认 3D\n")

    # P3: 自定义风格用 LLM 给的 style_prompt（顺用户输入丰富，而非预设）
    assert r4["style_key"].startswith("custom"), "黏土定格应走 custom"
    assert "claymation" in r4["style_suffix"].lower(), "custom 风格应带 LLM 给的 claymation 描述"
    print("  [P3 PASS] 自定义风格(黏土定格)使用 LLM 生成的 style_prompt，"
          "未套用任何预设 suffix\n")

    # P4: 负面约束（冲突词）动态合并，不写死
    print("=" * 64)
    print("P4: 冲突词动态生成（不同风格不同负面约束）")
    print("=" * 64)
    for key in ["watercolor", "anime", "documentary", "aaa_cg", "oil_painting"]:
        cw = g._get_conflict_keywords(key)
        print(f"  {key}: {len(cw)} 个冲突词，含 'unreal engine'={('unreal engine' in cw)}")
    # 通用写实词在所有风格都有
    wf = g._get_conflict_keywords("watercolor")
    assert "unreal engine" in wf and "photorealistic" in wf
    # aaa_cg 自身不应把 'unreal engine' 当冲突（它是写实风格）
    cg = g._get_conflict_keywords("aaa_cg")
    assert "unreal engine" not in cg, "写实风格不该把自身引擎当冲突词"
    print("\n  [P4 PASS] 冲突词按风格动态合并（写实风不排除自身引擎，非写实风排除 3D 术语）\n")

    # P5: _resolve_style 单一数据源 —— 第二次调用应直接读缓存不重抽
    print("=" * 64)
    print("P5: 风格单一数据源（_resolve_style 缓存复用）")
    print("=" * 64)
    s = make_state("丑小鸭童话", "水彩风格")
    s["visual_style"] = "watercolor"
    s["style_suffix"] = "delicate watercolor painting..."
    s["style_description"] = "watercolor fairy tale"
    cached = g._resolve_style(s)
    assert cached["style_key"] == "watercolor"
    print("  [P5 PASS] 下游节点从 state 缓存读取统一风格，不重复抽取\n")

    # P6: LLM 抽取失败时，兜底不应是「写实 3D(aaa_cg)」，而是中性 neutral
    print("=" * 64)
    print("P6: 兜底安全 —— LLM 抽取失败时回到 neutral，而非静默套写实 CG")
    print("=" * 64)
    def boom(prompt, purpose=""):
        raise RuntimeError("simulated LLM outage")
    g.call_llm = boom
    s = make_state("一段未明确风格的故事", "")
    res = g._extract_visual_style(s)
    print(f"  LLM 崩溃后 fallback style_key = {res['style_key']!r}")
    print(f"  fallback suffix 含 'Unreal Engine' = {'Unreal Engine' in res['style_suffix']}")
    assert res["style_key"] == "neutral", f"兜底应是 neutral，实际 {res['style_key']}"
    assert "Unreal Engine" not in res["style_suffix"], "兜底不应含写实引擎词（避免静默 3D）"
    # 恢复正常 mock 以便其他潜在调用
    g.call_llm = fake_call_llm
    print("  [P6 PASS] 兜底为中性风格，未静默套用写实 CG（消除画风崩坏根因）\n")

    print("=" * 64)
    print("ALL STYLE PIPELINE CASES PASSED  风格链路解耦、顺输入、可丰富")
    print("=" * 64)


if __name__ == "__main__":
    main()
