# src/agent/multimedia/reference_gen.py
"""
Reference Sheet Generation Module
==================================
在关键帧生成之前，为角色、道具、场景生成一致性参考图。

参考图类型：
  1. Character Turnaround Sheet — 4视图并排（front/3quarter/side/back）
  2. Prop Reference Sheet — 多角度展示（front/side/detail）
  3. Environment Reference Board — 建立镜头（wide establishing shot）

架构位置：
  showrunner_review → 【reference_gen_node】→ visual_context_builder

设计原则：
  - 用 LLM 从 global_setting + scenes 提取元素（不硬编码）
  - 每个元素生成一张参考图，存入 state.reference_sheets
  - 后续 image_gen_node 消费参考图，用 img2img ref_strength 锚定一致性
"""

import json
import os
from typing import Dict, Any, Optional

from .prompts import REFERENCE_EXTRACTION_PROMPT
from .tools.image_gen import generate_keyframe


def _extract_elements(
    global_setting: str,
    scenes: list,
    task: str,
) -> Dict[str, Any]:
    """
    用 LLM 从 global_setting + scenes 中提取需要参考图的关键元素。

    Returns:
        {
            "characters": [{"name": "...", "description": "..."}],
            "props": [{"name": "...", "description": "..."}],
            "environments": [{"name": "...", "description": "..."}],
        }
    """
    from .tools.text_llm import call_llm  # lazy import to avoid circular

    scenes_text = "\n".join(
        f"  Scene {i+1}: {s.get('script', '')}" for i, s in enumerate(scenes)
    )

    prompt = REFERENCE_EXTRACTION_PROMPT.format(
        task=task,
        global_setting=global_setting,
        scenes=scenes_text,
    )

    raw = call_llm(prompt, "参考元素提取")

    # 解析 JSON
    try:
        # 清理可能的 markdown 包裹
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        result = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    [WARN] 元素提取 JSON 解析失败: {e}")
        # 降级：只提取全局环境
        result = {
            "characters": [],
            "props": [],
            "environments": [{"name": "main", "description": global_setting[:200]}],
        }

    # 确保结构完整
    for key in ("characters", "props", "environments"):
        if key not in result:
            result[key] = []

    return result


def _build_character_prompt(char: Dict, style_suffix: str) -> str:
    """为角色转面图构建生成提示词。"""
    name = char.get("name", "character")
    desc = char.get("description", "")

    return (
        f"Character turnaround model sheet of {name}. "
        f"Four views arranged side by side on a clean white background, same scale: "
        f"front view, three-quarter view, side profile, back view. "
        f"Full body visible in all views, standing in a relaxed neutral pose. "
        f"{desc}. "
        f"Consistent design across all four views — same outfit, same hair, same proportions. "
        f"Professional character design sheet, anime concept art style. "
        f"{style_suffix}"
    )


def _build_prop_prompt(prop: Dict, style_suffix: str) -> str:
    """为道具设定图构建生成提示词。"""
    name = prop.get("name", "prop")
    desc = prop.get("description", "")

    return (
        f"Prop reference sheet of {name}. "
        f"Multiple views on a clean white background: front view, side view, and a detail close-up. "
        f"{desc}. "
        f"Professional concept art prop sheet, showing material texture and key design details. "
        f"{style_suffix}"
    )


def _build_environment_prompt(env: Dict, style_suffix: str) -> str:
    """为场景环境参考图构建生成提示词。"""
    name = env.get("name", "environment")
    desc = env.get("description", "")

    return (
        f"Wide establishing shot of {name}. "
        f"{desc}. "
        f"Cinematic environment concept art, showing the full scope of the location, "
        f"dominant lighting conditions, color palette, and atmospheric mood. "
        f"No characters visible — pure environment showcase. "
        f"{style_suffix}"
    )


def generate_reference_sheets(
    global_setting: str,
    scenes: list,
    task: str,
    style_suffix: str,
    style_key: str = "",
) -> Dict[str, Any]:
    """
    核心函数：提取关键元素 → 生成参考图 → 返回结构化结果。

    Returns:
        {
            "characters": {"name": {"url": "...", "description": "..."}},
            "props": {"name": {"url": "...", "description": "..."}},
            "environments": {"name": {"url": "...", "description": "..."}},
        }
    """
    elements = _extract_elements(global_setting, scenes, task)

    sheets: Dict[str, Any] = {"characters": {}, "props": {}, "environments": {}}

    # ── 生成角色转面图 ──
    for char in elements.get("characters", []):
        name = char.get("name", "character")
        desc = char.get("description", "")
        name_cn = char.get("name_cn", "")
        print(f"    [*] Generating character turnaround: {name}" + (f" ({name_cn})" if name_cn else ""))

        prompt = _build_character_prompt(char, style_suffix)
        try:
            url = generate_keyframe(
                prompt,
                size="1024*1024",  # square format for turnaround sheet
            )
            sheets["characters"][name] = {"url": url, "description": desc, "name_cn": name_cn}
            print(f"    [OK] Character sheet: {name}")
        except Exception as e:
            print(f"    [WARN] Character sheet failed for {name}: {e}")

    # ── 生成道具设定图 ──
    for prop in elements.get("props", []):
        name = prop.get("name", "prop")
        desc = prop.get("description", "")
        name_cn = prop.get("name_cn", "")
        print(f"    [*] Generating prop sheet: {name}" + (f" ({name_cn})" if name_cn else ""))

        prompt = _build_prop_prompt(prop, style_suffix)
        try:
            url = generate_keyframe(
                prompt,
                size="1024*1024",
            )
            sheets["props"][name] = {"url": url, "description": desc, "name_cn": name_cn}
            print(f"    [OK] Prop sheet: {name}")
        except Exception as e:
            print(f"    [WARN] Prop sheet failed for {name}: {e}")

    # ── 生成场景环境参考图 ──
    for env in elements.get("environments", []):
        name = env.get("name", "environment")
        desc = env.get("description", "")
        name_cn = env.get("name_cn", "")
        print(f"    [*] Generating environment reference: {name}" + (f" ({name_cn})" if name_cn else ""))

        prompt = _build_environment_prompt(env, style_suffix)
        try:
            url = generate_keyframe(
                prompt,
                size="1024*1024",
            )
            sheets["environments"][name] = {"url": url, "description": desc, "name_cn": name_cn}
            print(f"    [OK] Environment reference: {name}")
        except Exception as e:
            print(f"    [WARN] Environment reference failed for {name}: {e}")

    return sheets
