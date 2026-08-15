"""
统一配置加载器（配置与代码分离）

复用 shot_strategy._load_shot_config 的范式：
  - 与 shot_strategy 共享同一 _CONFIG_DIR（vision_knowledge/config）
  - 模块级缓存（文件只读取一次，零运行时开销）
  - 文件缺失时抛出明确错误，便于排错

承载两类配置：
  - thresholds.json   质量/重试/一致性等数值阈值
  - visual_rules.json 情绪能量表、可接受弧线、焦点白名单、转场阈值

不引入 pydantic-settings 全量改造，避免大改动；仅保持
CHARACTER_CONSISTENCY_THRESHOLD / MAX_VIDEO_GEN_FAILURES 的 os.getenv 覆盖点。
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

_CONFIG_DIR = Path(__file__).parent / "vision_knowledge" / "config"

_THRESHOLDS_FILE = _CONFIG_DIR / "thresholds.json"
# 注意：vision_knowledge/config/visual_rules.json 已被 visual_context.py 占用（含
# intent_visual_rules/genre_modifiers 等键），为避免覆盖破坏其加载，本项目新增的
# 规则键（情绪能量表/可接受弧线/焦点白名单/转场阈值）放在独立的 multimedia_rules.json。
_VISUAL_RULES_FILE = _CONFIG_DIR / "multimedia_rules.json"


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    """从 JSON 配置文件加载字典。文件缺失时抛出明确错误。"""
    if not path.exists():
        raise FileNotFoundError(
            f"{label}配置文件缺失: {path}\n"
            f"请确保 {path.name} 存在于 vision_knowledge/config/ 目录。"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_thresholds() -> Dict[str, Any]:
    """加载阈值配置（模块级缓存）。"""
    return _load_json(_THRESHOLDS_FILE, "阈值")


@lru_cache(maxsize=1)
def load_visual_rules() -> Dict[str, Any]:
    """加载视觉规则配置（模块级缓存）。"""
    return _load_json(_VISUAL_RULES_FILE, "视觉规则")


# ── 便捷访问器：把最易变的阈值暴露为函数，便于 env 覆盖与单测 mock ──

def character_consistency_threshold() -> float:
    """角色一致性阈值，env CHARACTER_CONSISTENCY_THRESHOLD 可覆盖。"""
    env = os.getenv("CHARACTER_CONSISTENCY_THRESHOLD")
    if env is not None:
        try:
            return float(env)
        except ValueError:
            pass
    return float(load_thresholds().get("character_consistency_threshold", 0.5))


def max_video_gen_failures() -> int:
    """视频生成最大失败次数，env MAX_VIDEO_GEN_FAILURES 可覆盖。"""
    env = os.getenv("MAX_VIDEO_GEN_FAILURES")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    return int(load_thresholds().get("max_video_gen_failures", 2))


def max_rewrites() -> int:
    """critic 重写上限，env MAX_REWRITES 可覆盖。"""
    env = os.getenv("MAX_REWRITES")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    return int(load_thresholds().get("max_rewrites", 2))


def quality_score_thresholds() -> Dict[str, float]:
    """质量评分三档阈值：excellent / pass / fail。"""
    return load_thresholds().get("quality_score", {
        "excellent": 0.8, "pass": 0.6, "fail": 0.15,
    })


def emotion_energy_table(profile: str = "default") -> Dict[str, float]:
    """情绪能量表。profile='editor' 返回 editor_agent 的副本。"""
    rules = load_visual_rules()
    if profile == "editor":
        return rules.get("emotion_energy_editor", rules.get("emotion_energy", {}))
    return rules.get("emotion_energy", {})


def acceptable_arcs() -> list:
    """可接受的情绪弧线集合（trailer 含 anti_crescendo 的 opening 例外）。"""
    return load_visual_rules().get("acceptable_arcs", ["crescendo", "arc"])


def anti_crescendo_relax_phases() -> list:
    """anti_crescendo 在哪些阶段按配置放行（如 opening）。"""
    return load_visual_rules().get("anti_crescendo_relax_phases", ["opening"])


def focus_patterns() -> Dict[str, list]:
    """焦点对象抽取白名单（category -> 关键词列表）。"""
    return load_visual_rules().get("focus_patterns", {})


def transition_threshold() -> Dict[str, float]:
    """跨场景转场能量跳变阈值。"""
    return load_visual_rules().get("transition_threshold", {
        "energy_jump_large": 0.4, "energy_jump_small": 0.15,
    })


# ── 与 graph 解耦的纯工具（无 moviepy/edge_tts 重依赖，便于单测与复用）──

def make_placeholder_clip(duration: float = 2.0, size=(1280, 720), out_path: str = "", idx: int = -1, label: str = "") -> str:
    """生成占位降级视频，返回本地文件路径（失败则返回空字符串）。

    内部 lazy import moviepy：依赖缺失时不抛异常，返回 ""（降级不可用但不阻断主流程）。
    视觉上采用深灰底 + 居中文字标注（默认"额度耗尽·跳过"，可传 label 覆盖），
    使成片中"因额度耗尽被降级的镜头"可辨识，而非突兀纯黑。
    """
    import os as _os
    if not out_path:
        clips_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "output", "clips")
        _os.makedirs(clips_dir, exist_ok=True)
        suffix = f"_{idx}" if idx >= 0 else ""
        out_path = _os.path.join(clips_dir, f"placeholder_scene{suffix}.mp4")
    try:
        try:
            from moviepy.editor import ColorClip
        except ImportError:
            from moviepy import ColorClip
    except Exception as e:  # pragma: no cover - 依赖缺失时降级不可用
        print(f"    [WARN] [降级] 占位黑场生成失败（moviepy 不可用）: {e}")
        return ""
    try:
        # 【修复】占位卡改为明显"风格化提示卡"而非近黑深灰底（原 (32,32,36) 观感接近黑屏）。
        # 采用仙侠暗蓝紫底 + 暖金文字，使降级镜头在成片中可被辨识为"占位提示卡"，
        # 而非突兀黑屏（stitcher 已默认跳过 shot_failed 占位卡，此处仅作视觉兜底）。
        clip = ColorClip(size=size, color=(20, 18, 34), duration=duration)
        clip = clip.with_audio(None)  # 静音底，避免 audio_mixer 拼接报错
        # 尝试叠加文字标注；TextClip 依赖字体，缺失时静默降级为纯底
        txt_clip = None
        try:
            caption = label or f"镜头 {idx + 1}\n额度耗尽 · 已跳过"
            # moviepy 2.x TextClip 需用 text= 关键字 + 绝对字体路径，避免误用字体名
            from .tools.fonts import make_text_clip
            txt_clip = (
                make_text_clip(
                    caption, font_size=48, color=(230, 200, 140),
                    size=(None, None), method="caption",
                )
                .set_duration(duration)
                .resize(height=int(size[1] * 0.5))
                .set_position("center")
            )
            clip = clip.set_duration(duration).fx(lambda c: c) if hasattr(clip, "fx") else clip
            from moviepy.editor import CompositeVideoClip  # type: ignore
            clip = CompositeVideoClip([clip, txt_clip])
        except Exception as te:  # pragma: no cover - 字体/依赖缺失时降级为纯深灰底
            print(f"    [INFO] [降级] 占位文字标注不可用，使用纯深灰底: {te}")
        clip.write_videofile(out_path, fps=24, codec="libx264", audio=False, logger=None)
        clip.close()
        if txt_clip is not None:
            try:
                txt_clip.close()
            except Exception:
                pass
        return out_path
    except Exception as e:  # pragma: no cover
        print(f"    [WARN] [降级] 占位黑场写入失败: {e}")
        return ""


def persist_studio_output(studio_result: Dict[str, Any], idx: int) -> str:
    """将完整 studio 结果落盘到 data/studio_cache/，返回本地路径（异常安全）。

    用于大 state 瘦身：完整 studio 结果不随 state 入库，仅存路径指针。
    """
    import os as _os
    import time as _time
    cache_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "studio_cache")
    try:
        _os.makedirs(cache_dir, exist_ok=True)
        fname = f"studio_{int(_time.time() * 1000)}_{idx}.json"
        fpath = _os.path.join(cache_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(studio_result, f, ensure_ascii=False, default=str)
        return fpath
    except Exception as e:  # pragma: no cover - 落盘失败不阻断主流程
        print(f"    ⚠️ [瘦身] studio_output 落盘失败（已回退为仅存 state）: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Google AI Studio（Gemini）接入配置读取
# 用于图像/视频生成的替代后端（Imagen / Veo），当 DashScope 免费额度耗尽时启用。
# 仅读取 .env 中的 GEMINI_* 变量，不在此处发起任何网络请求。
# ─────────────────────────────────────────────────────────────────────────────
def get_google_config() -> Dict[str, Any]:
    """读取 Google AI Studio 配置，返回字典。

    返回字段：
      - enabled:   MULTIMODIA_BACKEND == "google" 且 GEMINI_API_KEY 已填（非占位符）
      - api_key:   GEMINI_API_KEY（占位符 __FILL_ME_* 视为未填）
      - backend:   MULTIMODIA_BACKEND（dashscope / google）
      - image_model: GEMINI_IMAGE_MODEL
      - video_model: GEMINI_VIDEO_MODEL
      - base_url:  GEMINI_BASE_URL
    """
    raw_key = os.getenv("GEMINI_API_KEY", "").strip()
    backend = os.getenv("MULTIMODIA_BACKEND", "dashscope").strip().lower()
    # 占位符视为未填：__FILL_ME_GOOGLE_AI_STUDIO_KEY__
    key_filled = bool(raw_key) and not raw_key.startswith("__FILL_ME_")
    return {
        "enabled": backend == "google" and key_filled,
        "api_key": raw_key if key_filled else "",
        "backend": backend,
        "image_model": os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image").strip(),
        "video_model": os.getenv("GEMINI_VIDEO_MODEL", "veo-3.1-generate-preview").strip(),
        "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 即梦 AI Free（jimeng-api）接入配置读取（白嫖即梦网页版每日免费积分）
# 把即梦官网每日赠送的免费积分暴露为 OpenAI 兼容本地 API（生图 / 生视频 / Seedance）。
# 仅读取 .env 中的 JIMENG_* 变量，不在此处发起任何网络请求。
# 用作 DashScope / Google / SiliconFlow 之外的补充免费后端（缓解百炼额度紧张）。
# ─────────────────────────────────────────────────────────────────────────────
def get_jimeng_config() -> Dict[str, Any]:
    """读取即梦免费后端配置，返回字典。

    返回字段：
      - enabled:     MULTIMODIA_BACKEND == "jimeng" 且 JIMENG_SESSION_ID 已填
      - has_session: JIMENG_SESSION_ID 是否非空（用于 fallback 判断，即使 backend 非 jimeng）
      - session_ids: 逗号分隔解析后的 sessionid 列表
      - base_url:    JIMENG_BASE_URL
      - backend:     MULTIMODIA_BACKEND
    """
    raw = os.getenv("JIMENG_SESSION_ID", "").strip()
    backend = os.getenv("MULTIMODIA_BACKEND", "dashscope").strip().lower()
    ids = [s.strip() for s in raw.split(",") if s.strip()] if raw else []
    return {
        "enabled": backend == "jimeng" and bool(ids),
        "has_session": bool(ids),
        "session_ids": ids,
        "base_url": os.getenv("JIMENG_BASE_URL", "http://localhost:8080/v1").strip().rstrip("/"),
        "backend": backend,
    }


def get_visual_backend() -> str:
    """统一返回当前激活的多媒体后端名称（小写）。

    取值：dashscope（默认）| google | jimeng | siliconflow
    供 image_gen / video_gen 在入口处判断是否走对应分支。
    """
    return os.getenv("MULTIMODIA_BACKEND", "dashscope").strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# SiliconFlow（硅基流动）接入配置读取（img2img 图生图后端）
# 用素材参考图作为锚点，生成本镜头专属首/尾帧（而非把素材图直接当首尾帧）。
# 仅读取 .env 中的 SILICONFLOW_* 变量，不在此处发起任何网络请求。
# ─────────────────────────────────────────────────────────────────────────────
def get_siliconflow_config() -> Dict[str, Any]:
    """读取 SiliconFlow 后端配置，返回字典。

    返回字段：
      - enabled:   SILICONFLOW_API_KEY 已填（非占位符）
      - api_key:   SILICONFLOW_API_KEY
      - base_url:  SILICONFLOW_BASE_URL
      - image_model: SILICONFLOW_IMAGE_MODEL（img2img 模型）
    """
    raw_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    key_filled = bool(raw_key) and not raw_key.startswith("__FILL_ME_")
    return {
        "enabled": key_filled,
        "api_key": raw_key if key_filled else "",
        "base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").strip().rstrip("/"),
        "image_model": os.getenv("SILICONFLOW_IMAGE_MODEL", "Kwai-Kolors/Kolors").strip(),
    }
