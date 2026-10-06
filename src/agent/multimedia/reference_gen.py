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
import shutil
import logging
from typing import Dict, Any, Optional

import requests
from pathlib import Path

logger = logging.getLogger(__name__)

from .prompts import REFERENCE_EXTRACTION_PROMPT
from .tools.image_gen import generate_keyframe
from .tools.media_paths import local_to_media_url, media_url_to_local

# 项目根目录：reference_gen.py -> multimedia -> agent -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# 参考图本地落盘根目录（相对于 PROJECT_ROOT）
REFERENCE_SHEETS_DIR = PROJECT_ROOT / "output" / "reference_sheets"


def _ext_from_url(url: str, default: str = ".png") -> str:
    path = url.split("?")[0].split("#")[0]
    ext = os.path.splitext(path)[1].lower()
    return ext or default


def _local_media_path(src: Path) -> Optional[str]:
    """若本地文件位于项目内，返回其 /media URL；否则 None。

    统一走 media_paths.local_to_media_url：output/ 内产出 output-relative 的
    ``/media/<rel>``（static_server 文档约定），output/ 之外退化为带项目根名的
    ``/media/<root>/<rel>``（root_marker 兜底）。修复此前直接 relative_to(PROJECT_ROOT)
    产出 ``/media/output/...``，使下游按 output-relative 解析时二次拼出 output/output/。
    """
    return local_to_media_url(src)


def _ref_cache_enabled() -> bool:
    """参考图缓存开关（默认开）。"""
    return (os.getenv("MULTIMEDIA_REF_CACHE", "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _ref_cache_dir() -> Path:
    d = REFERENCE_SHEETS_DIR / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ref_cache_key(prompt: str, size: str) -> str:
    import hashlib
    return hashlib.sha1(f"{prompt}|{size}".encode("utf-8")).hexdigest()


def _ref_cache_lookup(prompt: str, size: str) -> Optional[Path]:
    """命中则返回缓存文件路径；否则 None。"""
    if not _ref_cache_enabled():
        return None
    f = _ref_cache_dir() / f"{_ref_cache_key(prompt, size)}.png"
    if f.is_file() and f.stat().st_size > 0:
        return f
    return None


def _ref_cache_store(prompt: str, size: str, media_path: str) -> None:
    """把刚生成的参考图（/media 路径）复制进缓存，供后续运行复用。"""
    if not _ref_cache_enabled():
        return
    try:
        if not media_path or not media_path.startswith("/media/"):
            return
        src = media_url_to_local(media_path)
        if src is None:
            return
        if src.is_file() and src.stat().st_size > 0:
            dst = _ref_cache_dir() / f"{_ref_cache_key(prompt, size)}.png"
            if not dst.exists():
                shutil.copyfile(src, dst)
    except Exception as exc:  # noqa: BLE001
        logger.warning("参考图缓存写入失败（忽略）: %s", exc)


def _persist_reference_image(remote_url: str, thread_id: str, category: str, name: str) -> str:
    """
    把参考图归置到本地 output/reference_sheets/<thread_id>/<category>/，
    返回可经 /media 访问的本地路径；失败时回退原值。

    入参可能是：
      - 远程 URL（DashScope 等）→ 下载落盘；
      - 本地文件路径（jimeng 免费后端已落盘）→ 项目内直接转 /media，项目外复制；
      - data: URI / 已是 /media 路径 → 原样透传。
    """
    if not remote_url or remote_url.startswith("/media") or remote_url.startswith("data:"):
        return remote_url
    tid = thread_id or "default"
    safe_cat = "".join(c for c in category if c.isalnum() or c in "-_").strip() or "misc"
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip() or "ref"
    target_dir = REFERENCE_SHEETS_DIR / tid / safe_cat
    try:
        # —— 本地文件（免费后端已落盘）——
        if os.path.isfile(remote_url):
            src = Path(remote_url)
            media = _local_media_path(src)
            if media:
                return media  # 已在项目内，直接由静态服务器提供
            target_dir.mkdir(parents=True, exist_ok=True)
            dest = target_dir / f"{safe_name}{_ext_from_url(remote_url)}"
            shutil.copyfile(src, dest)
            if dest.is_file() and dest.stat().st_size > 0:
                return local_to_media_url(dest) or remote_url
            return remote_url
        # —— 远程 URL ——
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = _ext_from_url(remote_url)
        dest = target_dir / f"{safe_name}{ext}"
        resp = requests.get(remote_url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        if dest.is_file() and dest.stat().st_size > 0:
            return local_to_media_url(dest) or remote_url
    except Exception as exc:  # noqa: BLE001
        logger.warning("参考图本地持久化失败，回退原值: %s (%s)", remote_url, exc)
    return remote_url


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
    thread_id: str = "",
) -> Dict[str, Any]:
    """
    核心函数：提取关键元素 → 生成参考图 → 返回结构化结果。

    生成后会将远程链接下载到本地 output/reference_sheets/<thread_id>/，
    并把 url 改写为可经 /media 稳定访问的本地路径（远程链接过期也不影响前端显示）。

    Args:
        thread_id: 当前运行所属 thread_id，用于在本地按线程隔离落盘。

    Returns:
        {
            "characters": {"name": {"url": "...", "description": "..."}},
            "props": {"name": {"url": "...", "description": "..."}},
            "environments": {"name": {"url": "...", "description": "..."}},
        }
    """
    elements = _extract_elements(global_setting, scenes, task)

    sheets: Dict[str, Any] = {"characters": {}, "props": {}, "environments": {}}

    # S1（P0-2 子项）：参考图生成并行化——角色/道具/环境三组循环原本串行逐张生成，
    # 即梦单张生图数秒~数十秒，多角色任务累计耗时显著。此处把全部条目扁平化后用
    # 线程池并发提交（max_workers=3，避免触发即梦并发限流），结果按条目回收写回。
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = []
    for char in elements.get("characters", []):
        jobs.append(("characters", char, _build_character_prompt(char, style_suffix)))
    for prop in elements.get("props", []):
        jobs.append(("props", prop, _build_prop_prompt(prop, style_suffix)))
    for env in elements.get("environments", []):
        jobs.append(("environments", env, _build_environment_prompt(env, style_suffix)))

    _REF_SIZE = "1024*1024"

    def _gen_one(category: str, item: dict, prompt: str):
        name = item.get("name", category.rstrip("s"))
        # 缓存命中：同 prompt 的参考图跨运行复用，省一次即梦生图额度
        _hit = _ref_cache_lookup(prompt, _REF_SIZE)
        if _hit is not None:
            _media = _local_media_path(_hit) or _persist_reference_image(
                str(_hit), thread_id, category, name
            )
            print(f"    [CACHE] 参考图命中缓存（跳过生图，省额度）: {category}/{name}")
            return category, name, _media
        url = generate_keyframe(prompt, size=_REF_SIZE)
        url = _persist_reference_image(url, thread_id, category, name)
        _ref_cache_store(prompt, _REF_SIZE, url)
        return category, name, url

    max_workers = max(1, min(3, len(jobs)))
    if jobs:
        print(f"    [*] 参考图并行生成：{len(jobs)} 张（max_workers={max_workers}）")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_gen_one, cat, item, pr): (cat, item) for cat, item, pr in jobs}
        for fut in as_completed(futures):
            cat, item = futures[fut]
            name = item.get("name", cat.rstrip("s"))
            desc = item.get("description", "")
            name_cn = item.get("name_cn", "")
            try:
                _category, _name, url = fut.result()
                sheets[_category][_name] = {"url": url, "description": desc, "name_cn": name_cn}
                print(f"    [OK] {_category} sheet: {_name}")
            except Exception as e:  # noqa: BLE001 — 单张失败不拖垮其余参考图
                print(f"    [WARN] {cat} sheet failed for {name}: {e}")

    return sheets
