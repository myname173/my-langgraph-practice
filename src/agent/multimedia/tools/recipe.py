# src/agent/multimedia/tools/recipe.py
"""P2-4 镜头配方卡（Shot Recipe）。

把「一个镜头是怎么被造出来的」完整记录下来，成为一份**可复现、可分享、可微调**
的结构化配方：

    {
      "schema": "multimedia.shot_recipe.v1",
      "thread_id": "...",
      "created_at": "2026-10-03T02:30:00+08:00",
      "task": "...",
      "global_setting": "...",
      "style": {"visual_style": ..., "description": ..., "suffix": ...},
      "post": {"color_preset": ..., "subtitle_theme": ..., "title_card": ...},
      "shots": [
        {
          "index": 0,
          "script": "...",
          "image":   {"backend","prompt","negative_prompt","size","ref_images","ref_strength","model"},
          "end_frame": {"prompt","ref_strength","ref_images"},
          "video":   {"backend","quality_tier","duration","prompt","use_first_last_frame","negative_prompt","reference_images","model"},
          "fix":     {"fix_type","extend_applied"},
          "artifacts": {"image_url","last_image_url","final_video_url"}
        }, ...
      ]
    }

设计原则：
- **只读、纯函数**：`build_shot_recipe` / `build_task_recipe` 不改任何 state，容错缺字段。
- **向后兼容**：P1-3 已把 `render_params` 写在 scene 上；本模块在其上做「补全成完整配方」，
  旧字段（backend/quality_tier/duration/video_prompt/use_first_last_frame）原样保留。
- **可微调**：`apply_shot_patch` 深合并一份补丁到指定镜头的配方（仅白名单字段），
  产出新配方；调用方据此把改动写回 scene 再重跑。
- **零额度**：任何函数都不触发生图/生视频，仅序列化与读写本地 JSON。
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

RECIPE_SCHEMA = "multimedia.shot_recipe.v1"

# 项目根：tools/ -> multimedia/ -> agent/ -> src/ -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_RECIPE_DIR = _PROJECT_ROOT / "output" / "recipes"

# 允许被「微调」写回的字段白名单（prompt / 强度 / 时长 / 后处理）。
# 结构：{段落: {字段: 类型}}；类型仅做轻量校验，不做强制强转（容错优先）。
PATCHABLE: Dict[str, Dict[str, type]] = {
    "image": {"prompt": str, "negative_prompt": str, "ref_strength": (int, float), "size": str},
    "end_frame": {"prompt": str, "ref_strength": (int, float)},
    "video": {"prompt": str, "duration": (int, float), "quality_tier": str, "negative_prompt": str},
    "post": {"color_preset": str, "subtitle_theme": str, "title_card": bool},
    "fix": {"fix_type": str},
}


def _now_iso() -> str:
    """带本地时区偏移的 ISO 时间戳（Asia/Shanghai = +08:00）。"""
    try:
        tz = timezone(timedelta(hours=8))
        return datetime.now(tz).isoformat(timespec="seconds")
    except Exception:
        return datetime.now().isoformat(timespec="seconds")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def _as_str_list(v: Any) -> list:
    if not v:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x]
    return [str(v)]


def build_shot_recipe(scene: Dict[str, Any], idx: int, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把单个 scene（+ 可选的顶层 state 背景）提炼成一份镜头配方。纯函数、容错。"""
    scene = scene or {}
    state = state or {}
    rp = scene.get("render_params") or {}
    if not isinstance(rp, dict):
        rp = {}
    img_rp = rp.get("image") if isinstance(rp.get("image"), dict) else {}
    end_rp = rp.get("end_frame") if isinstance(rp.get("end_frame"), dict) else {}
    vid_rp = rp.get("video") if isinstance(rp.get("video"), dict) else {}

    image = {
        "backend": img_rp.get("backend") or _env("MULTIMODIA_BACKEND", "jimeng"),
        "model": img_rp.get("model") or _env("JIMENG_MODEL", ""),
        "prompt": img_rp.get("prompt") or scene.get("image_prompt") or "",
        "negative_prompt": img_rp.get("negative_prompt") or "",
        "size": img_rp.get("size") or scene.get("image_size") or "2K",
        "ref_images": _as_str_list(img_rp.get("ref_images") or scene.get("used_reference_images")),
        "ref_strength": img_rp.get("ref_strength", scene.get("image_ref_strength")),
    }
    end_frame = {
        "prompt": end_rp.get("prompt") or scene.get("last_image_prompt") or "",
        "ref_strength": end_rp.get("ref_strength", scene.get("last_image_ref_strength")),
        "ref_images": _as_str_list(end_rp.get("ref_images") or ([scene.get("image_url")] if scene.get("image_url") else [])),
    }
    video = {
        "backend": vid_rp.get("backend") or rp.get("backend") or "",
        "model": vid_rp.get("model") or _env("AGNES_VIDEO_MODEL", ""),
        "quality_tier": vid_rp.get("quality_tier") or rp.get("quality_tier") or state.get("quality_tier") or "",
        "duration": vid_rp.get("duration", rp.get("duration", scene.get("duration_seconds"))),
        "prompt": vid_rp.get("prompt") or rp.get("video_prompt") or scene.get("video_prompt") or "",
        "negative_prompt": vid_rp.get("negative_prompt") or "",
        "use_first_last_frame": bool(
            vid_rp.get("use_first_last_frame", rp.get("use_first_last_frame", state.get("use_first_last_frame")))
        ),
        "reference_images": _as_str_list(vid_rp.get("reference_images")),
    }
    fix = {
        "fix_type": scene.get("fix_type") or (rp.get("fix") or {}).get("fix_type") or "",
        "extend_applied": bool(scene.get("extend_applied")),
    }
    return {
        "index": int(idx),
        "script": scene.get("script") or "",
        "action_beat": scene.get("action_beat") or "",
        "image": image,
        "end_frame": end_frame,
        "video": video,
        "fix": fix,
        "artifacts": {
            "image_url": scene.get("image_url") or "",
            "last_image_url": scene.get("last_image_url") or "",
            "final_video_url": scene.get("final_video_url") or scene.get("raw_video_url") or "",
        },
    }


def build_task_recipe(state: Dict[str, Any]) -> Dict[str, Any]:
    """把整个 run 的 state 提炼成一份「任务配方」（含所有镜头）。纯函数。"""
    state = state or {}
    scenes = state.get("scenes") or []
    shots = [
        build_shot_recipe(sc, i, state)
        for i, sc in enumerate(scenes)
        if isinstance(sc, dict)
    ]
    return {
        "schema": RECIPE_SCHEMA,
        "thread_id": state.get("thread_id") or "",
        "created_at": _now_iso(),
        "task": state.get("task") or "",
        "global_setting": state.get("global_setting") or "",
        "style": {
            "visual_style": state.get("visual_style") or "",
            "description": state.get("style_description") or "",
            "suffix": state.get("style_suffix") or "",
        },
        "post": {
            "color_preset": _env("MULTIMEDIA_COLOR_PRESET", ""),
            "subtitle_theme": _env("MULTIMEDIA_SUBTITLE_THEME", "default"),
            "title_card": bool(state.get("title_card")),
        },
        "shots": shots,
    }


# ── 可读渲染（供前端「复制配方」与人眼阅读）────────────────────────────────

def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def recipe_to_markdown(recipe: Dict[str, Any]) -> str:
    """把一份配方渲染成人类可读的 Markdown 卡片文本。"""
    recipe = recipe or {}
    out: list[str] = []
    out.append(f"# 镜头配方卡 · {recipe.get('task') or '(未命名)'}")
    out.append("")
    out.append(f"- schema: `{recipe.get('schema', RECIPE_SCHEMA)}`")
    out.append(f"- thread: `{recipe.get('thread_id') or '—'}`")
    out.append(f"- 生成于: {recipe.get('created_at') or '—'}")
    style = recipe.get("style") or {}
    out.append(f"- 视觉风格: {_fmt(style.get('visual_style'))} · {_fmt(style.get('description'))}")
    post = recipe.get("post") or {}
    out.append(
        f"- 后处理: 调色预设 {_fmt(post.get('color_preset'))} · 字幕主题 {_fmt(post.get('subtitle_theme'))}"
        f" · 包装卡 {'开' if post.get('title_card') else '关'}"
    )
    out.append("")
    shots = recipe.get("shots") or []
    out.append(f"共 {len(shots)} 个镜头")
    for sh in shots:
        img = sh.get("image") or {}
        end = sh.get("end_frame") or {}
        vid = sh.get("video") or {}
        fix = sh.get("fix") or {}
        out.append("")
        out.append(f"## 镜头 {int(sh.get('index', 0)) + 1}")
        out.append(f"- 剧本: {_fmt(sh.get('script'))}")
        out.append(f"- 动作节拍: {_fmt(sh.get('action_beat'))}")
        out.append(
            f"- 首帧: 后端 {_fmt(img.get('backend'))} · 尺寸 {_fmt(img.get('size'))}"
            f" · 参考强度 {_fmt(img.get('ref_strength'))} · 参考图 {len(img.get('ref_images') or [])} 张"
        )
        out.append(f"  - image_prompt: {_fmt(img.get('prompt'))}")
        if end.get("prompt"):
            out.append(f"- 尾帧: 参考强度 {_fmt(end.get('ref_strength'))}")
            out.append(f"  - last_image_prompt: {_fmt(end.get('prompt'))}")
        out.append(
            f"- 视频: 后端 {_fmt(vid.get('backend'))} · 档位 {_fmt(vid.get('quality_tier'))}"
            f" · 时长 {_fmt(vid.get('duration'))}s · 首尾帧双控 {'是' if vid.get('use_first_last_frame') else '否'}"
        )
        out.append(f"  - video_prompt: {_fmt(vid.get('prompt'))}")
        if fix.get("fix_type") or fix.get("extend_applied"):
            out.append(f"- 修正痕迹: {_fmt(fix.get('fix_type'))} · 已延长 {'是' if fix.get('extend_applied') else '否'}")
    return "\n".join(out)


# ── 微调：深合并补丁 ────────────────────────────────────────────────────────

def apply_shot_patch(recipe: Dict[str, Any], idx: int, patch: Dict[str, Any]) -> Dict[str, Any]:
    """把一份补丁深合并到指定镜头的配方上，返回**新的**配方（不改原对象）。

    仅接受 PATCHABLE 白名单内的字段；空值（None/"") 视为「不修改」，避免误清空。
    """
    out = copy.deepcopy(recipe or {})
    shots = out.get("shots") or []
    target = None
    for sh in shots:
        if int(sh.get("index", -1)) == int(idx):
            target = sh
            break
    if target is None:
        raise KeyError(f"shot index {idx} not found in recipe")
    patch = patch or {}
    for section, fields in PATCHABLE.items():
        incoming = patch.get(section)
        if not isinstance(incoming, dict):
            continue
        # 顶层修正类型可直接挂在 fix 段
        dst = target.setdefault(section, {})
        if not isinstance(dst, dict):
            dst = {}
            target[section] = dst
        for k, v in incoming.items():
            if k not in fields:
                continue
            if v is None or v == "":
                continue
            dst[k] = v
    # 传递「本次微调」的时间戳标记，便于前端/日志区分
    target.setdefault("fix", {})
    if isinstance(target["fix"], dict):
        target["fix"]["recipe_patched_at"] = _now_iso()
    return out


def shot_patch_to_scene_fields(patch: Dict[str, Any]) -> Dict[str, Any]:
    """把镜头配方补丁翻译成 scene 上的字段（供重跑时写回 checkpoint）。

    返回值形如 {"image_prompt":..., "last_image_prompt":..., "video_prompt":...,
                "duration_seconds":..., "fix_type":..., "render_params": {...}}
    仅包含补丁里实际给出的项。
    """
    patch = patch or {}
    fields: Dict[str, Any] = {}
    rp: Dict[str, Any] = {}
    img = patch.get("image") or {}
    end = patch.get("end_frame") or {}
    vid = patch.get("video") or {}
    post = patch.get("post") or {}
    fix = patch.get("fix") or {}
    if isinstance(img.get("prompt"), str) and img["prompt"].strip():
        fields["image_prompt"] = img["prompt"].strip()
        rp["image"] = {"prompt": img["prompt"].strip()}
    if isinstance(end.get("prompt"), str) and end["prompt"].strip():
        fields["last_image_prompt"] = end["prompt"].strip()
        rp.setdefault("end_frame", {})["prompt"] = end["prompt"].strip()
    if isinstance(vid.get("prompt"), str) and vid["prompt"].strip():
        fields["video_prompt"] = vid["prompt"].strip()
        rp["video"] = {"prompt": vid["prompt"].strip()}
    if isinstance(vid.get("duration"), (int, float)):
        fields["duration_seconds"] = float(vid["duration"])
        rp.setdefault("video", {})["duration"] = float(vid["duration"])
    if isinstance(vid.get("quality_tier"), str) and vid["quality_tier"].strip():
        rp.setdefault("video", {})["quality_tier"] = vid["quality_tier"].strip()
    if isinstance(fix.get("fix_type"), str) and fix["fix_type"].strip():
        fields["fix_type"] = fix["fix_type"].strip()
    if post:
        rp["post"] = {k: v for k, v in post.items() if v not in (None, "")}
    if rp:
        fields["render_params"] = rp
    return fields


# ── 落盘 / 读取 ────────────────────────────────────────────────────────────

def recipe_dir(out_dir: Optional[Any] = None) -> Path:
    return Path(out_dir) if out_dir else _DEFAULT_RECIPE_DIR


def recipe_paths(thread_id: str, out_dir: Optional[Any] = None) -> Dict[str, Path]:
    base = recipe_dir(out_dir) / (thread_id or "_unknown")
    return {"dir": base, "latest": base / "latest.json"}


def save_task_recipe(recipe: Dict[str, Any], out_dir: Optional[Any] = None) -> Dict[str, str]:
    """把任务配方落盘为 <dir>/<thread>/<timestamp>.json + latest.json。返回两个路径。"""
    tid = (recipe or {}).get("thread_id") or "_unknown"
    paths = recipe_paths(tid, out_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    text = json.dumps(recipe, ensure_ascii=False, indent=2)
    stamp = _now_iso().replace(":", "").replace("-", "").replace("+", "_")
    snapshot = paths["dir"] / f"recipe_{stamp}.json"
    snapshot.write_text(text, encoding="utf-8")
    paths["latest"].write_text(text, encoding="utf-8")
    return {"dir": str(paths["dir"]), "snapshot": str(snapshot), "latest": str(paths["latest"])}


def load_task_recipe(thread_id: str, out_dir: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """读取某 thread 最近一次落盘的配方（latest.json）。不存在返回 None。"""
    p = recipe_paths(thread_id, out_dir)["latest"]
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
