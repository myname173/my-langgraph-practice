# src/agent/multimedia/tools/vision_eval.py
import os
import json
import base64
import tempfile
from io import BytesIO
from typing import Optional, List, Dict, Any

import httpx
import requests
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

http_client = httpx.Client(trust_env=False, timeout=120.0)

# ── 视觉通道 env 覆盖（P0-3 后续）──────────────────────────
# DashScope 视觉模型不可用（Key 失效 / 额度耗尽）时，可用 env 切到任意
# OpenAI 兼容视觉端点（如智谱 GLM-4V）：MULTIMEDIA_VISION_BASE_URL / _API_KEY / _MODEL / _OMNI_MODEL
_VISION_BASE_URL = (os.getenv("MULTIMEDIA_VISION_BASE_URL") or os.getenv("OPENAI_BASE_URL")
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1")
_VISION_API_KEY = (os.getenv("MULTIMEDIA_VISION_API_KEY") or os.getenv("OPENAI_API_KEY")
                   or os.getenv("DASHSCOPE_API_KEY"))
_VISION_MODEL = (os.getenv("MULTIMEDIA_VISION_MODEL") or os.getenv("DASHSCOPE_VISION_MODEL")
                 or "qwen-image-3.0-pro")
_VISION_OMNI_MODEL = (os.getenv("MULTIMEDIA_OMNI_MODEL") or os.getenv("DASHSCOPE_OMNI_MODEL")
                      or "qwen3.5-omni-plus-2026-03-15")

client = OpenAI(
    api_key=_VISION_API_KEY,
    base_url=_VISION_BASE_URL,
    http_client=http_client,
    timeout=120.0,
)


def _image_to_data_url(image: Image.Image, max_size: int = 768, quality: int = 88) -> str:
    img = image.convert("RGB")
    width, height = img.size
    long_side = max(width, height)

    if long_side > max_size:
        scale = max_size / float(long_side)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _resolve_image_input(u: str, max_size: int = 768, quality: int = 88) -> str:
    """把图片输入统一成视觉模型可消费的形式。

    - ``http(s)://`` / ``data:`` 原样返回；
    - 本地文件路径：读取并缩放后转 JPEG data URL。

    背景：即梦（jimeng）等后端产出的是**本地文件路径**，而 OpenAI 兼容视觉端点
    （如智谱 glm-4v-flash）无法解析本地路径，会直接报
    ``400 {... 'code': '1210', 'message': '图片输入格式/解析错误'}``。
    DashScope 时代图片多为 https OSS URL，故该缺陷此前未暴露。
    """
    if not u:
        return u
    s = str(u)
    if s.startswith(("http://", "https://", "data:")):
        return s
    try:
        if os.path.isfile(s):
            with Image.open(s) as im:
                return _image_to_data_url(im, max_size=max_size, quality=quality)
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN][Vision] 本地图转 data URL 失败，原样透传: {e}")
    return s


def _download_to_tempfile(url: str, suffix: str = ".mp4") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    response = requests.get(url, stream=True, timeout=180)
    response.raise_for_status()

    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return tmp_path


def _sample_video_frames(video_path: str, frame_count: int = 4) -> List[str]:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        from moviepy import VideoFileClip

    clip = None
    data_urls: List[str] = []

    try:
        clip = VideoFileClip(video_path)
        duration = float(getattr(clip, "duration", 0.0) or 0.0)

        if duration <= 0.0:
            raise ValueError("视频时长无效，无法抽帧。")

        fractions = [0.15, 0.35, 0.60, 0.85]
        timestamps = []
        for frac in fractions[: max(1, frame_count)]:
            t = duration * frac
            t = max(0.05, min(t, max(0.05, duration - 0.05)))
            timestamps.append(t)

        while len(timestamps) < frame_count:
            timestamps.append(timestamps[-1])

        for t in timestamps[:frame_count]:
            frame = clip.get_frame(t)
            img = Image.fromarray(frame)
            data_urls.append(_image_to_data_url(img))

        return data_urls
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass


def evaluate_image(
    image_url: str,
    script: str,
    prompt_template: str,
    previous_image_url: Optional[str] = None
) -> str:
    """调用视觉模型评估关键帧。"""
    system_msg = prompt_template.format(script=script)

    content = [
        {"type": "image_url", "image_url": {"url": _resolve_image_input(image_url)}},
    ]
    if previous_image_url:
        content.append({"type": "image_url", "image_url": {"url": _resolve_image_input(previous_image_url)}})
        system_msg += "\n\n【重要】第二张图片是前一个镜头的关键帧。请严格检查视觉一致性。"

    content.append({"type": "text", "text": system_msg})

    try:
        response = client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=512
        )

        result = str(response.choices[0].message.content or "").strip()

        if "PASS" in result.upper():
            print("    ✅ 艺术总监 + 安全官审核通过！")
            return "PASS"

        print(f"    ❌ 审核打回: {result[:300]}...")
        return result

    except Exception as e:
        error_str = str(e).lower()

        if "data_inspection_failed" in error_str or "inappropriate content" in error_str:
            fail_msg = (
                "FAIL: Content safety pre-check flagged this image (data_inspection_failed). "
                "Please regenerate with the same subject and composition, but reduce realistic violence detail, "
                "avoid explicit weapon close-ups or blood effects, and shift toward a more stylized, cinematic, "
                "non-photorealistic artistic expression while preserving the scene's mood and atmosphere."
            )
            print("    ⚠️ [安全拦截] 绿网误判，已转为 FAIL 重绘")
            print(f"    {fail_msg}")
            return fail_msg

        print(f"    ⚠️ [审核员接口异常] {str(e)[:200]}")
        print("    转为人工审核（不再自动放行）")
        # 返回统一标记：审核模型不可用（如免费额度耗尽 403），供上游跳过"一致性强制重生"等无效重试
        return f"REVIEWER_UNAVAILABLE: Reviewer API error — please manually check this image. Error: {str(e)[:150]}"


# ── P1-4：VLM 成对「角色一致性」判断（结构化输出）──────────────────────
# 背景：原一致性闭环依赖 DashScope multimodal-embedding 计算余弦相似度，
# 但该通道 Key 失效（401），使闭环静默降级。P1-4 改为用视觉模型对
# 【当前关键帧 vs 参照帧(上一镜尾帧/源头肖像)】直接做「是否同一角色」判断。
# 关键约束（实测）：flash 档模型正文推理可用，但自由文本结论标签会飘，
# 故必须强制结构化 JSON 输出，解析失败即视为不可用（优雅降级，不阻断主流程）。
# P1-4 前置：比较前先做 mediapipe 人脸对齐（摆正 + 裁剪），降低姿态噪声。
# env MULTIMEDIA_FACE_ALIGN（默认 "1" 开启）；mediapipe 缺失/无脸时自动回退原图。
_FACE_ALIGN_ENABLED = os.getenv("MULTIMEDIA_FACE_ALIGN", "1") != "0"


def _prepare_for_compare(u: str) -> tuple:
    """把图片输入整理为「视觉模型可消费」的形式：优先对齐后的 data URL。

    返回 ``(url, aligned_bool)``。任何异常都回退到 ``_resolve_image_input``，绝不阻断。
    """
    resolved = _resolve_image_input(u)
    if not _FACE_ALIGN_ENABLED or not u:
        return resolved, False
    try:
        from .face_align import align_face  # 惰性导入：mediapipe 非硬依赖
        r = align_face(u)
        if r.get("available") and r.get("aligned_data_url"):
            return r["aligned_data_url"], True
    except Exception as e:  # noqa: BLE001
        print(f"    [P1-4][FaceAlign] 对齐跳过(不阻断): {str(e)[:120]}")
    return resolved, False


_CONSISTENCY_PROMPT = (
    "你是影视制作流水线中的【角色一致性审片员】。下面有两张图片：图1 与 图2。\n"
    "任务：判断图1 与图2 是否刻画【同一个角色】。\n"
    "判据（只看角色身份）：面部特征、发型、服装、体型、配饰/装备是否一致。\n"
    "不计入判据：镜头角度、景别、光线、背景、动作差异（这些不同属正常）。\n"
    "【关键】只比较图1 与图2 这两张图彼此之间的关系；不要用任何文字描述作为判据，"
    "也不要评价画面内容是否合理或是否符合某段剧情。\n"
    "必须严格只输出一个 JSON 对象，不要解释文字、不要 markdown 代码块：\n"
    '{"same_character": true 或 false, "score": 0到1之间的小数, "reason": "一句话中文理由"}\n'
    "score 表示『图1 与图2 是同一角色』的置信度：1=完全确定同一角色，0=完全确定不同角色。"
)


def _extract_first_json(text: str) -> Optional[dict]:
    """从模型自由文本中提取第一个 JSON 对象；失败返回 None。"""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:  # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


def assess_character_consistency(
    image_url: str,
    reference_url: str,
    context: str = "",
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """P1-4：用视觉模型对两张图做「角色一致性」成对判断（结构化输出）。

    替代失效的 embedding 余弦：把当前关键帧与参照帧一起喂给 VLM，要求返回
    严格 JSON ``{"same_character": bool, "score": 0-1, "reason": str}``。

    Args:
        image_url: 当前镜头关键帧（http(s)/data/本地路径均可）。
        reference_url: 参照帧（上一镜尾帧或源头角色肖像）。
        context: 镜头剧本上下文，帮助模型理解场景。
        threshold: 判定漂移的分数阈值（score 低于它且 same_character=false 才算漂移）。

    Returns:
        字典：``{"available", "same_character", "score", "drifting",
        "reason", "method", "threshold"}``。接口/解析失败时 available=False、
        drifting=False（优雅降级，绝不阻断主流程）。
    """
    result: Dict[str, Any] = {
        "available": False,
        "same_character": None,
        "score": None,
        "drifting": False,
        "reason": "",
        "method": "vlm_pair",
        "threshold": float(threshold),
        "aligned": False,
    }
    if not image_url or not reference_url:
        result["reason"] = "缺少图片输入"
        return result

    # 注意：不注入 context —— 实测弱模型会把文字描述当成判据，去比对「画面 vs 描述」
    # 而非「图1 vs 图2」，导致同角色被误判为漂移（进而误触发重生烧额度）。
    prompt = _CONSISTENCY_PROMPT

    cur_url, cur_aligned = _prepare_for_compare(image_url)
    ref_url, ref_aligned = _prepare_for_compare(reference_url)
    result["aligned"] = bool(cur_aligned and ref_aligned)

    content = [
        {"type": "image_url", "image_url": {"url": cur_url}},
        {"type": "image_url", "image_url": {"url": ref_url}},
        {"type": "text", "text": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=256,
        )
        raw = str(response.choices[0].message.content or "").strip()
        data = _extract_first_json(raw)
        if not data:
            result["available"] = True  # 接口通、但输出非结构化
            result["reason"] = f"无法解析一致性 JSON: {raw[:120]}"
            return result

        same = data.get("same_character")
        score = data.get("score")
        try:
            score = float(score)
            if score > 1.0:  # 容错：模型可能返回 0-100
                score = score / 100.0
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = None

        result["available"] = True
        result["same_character"] = bool(same) if same is not None else None
        result["score"] = score
        result["reason"] = str(data.get("reason", ""))[:200]
        # 仅当「分数低」且「模型明确判不同角色」两个信号同时成立才算漂移，降低误杀。
        if score is not None:
            result["drifting"] = bool((score < float(threshold)) and (same is False))
        return result

    except Exception as e:  # noqa: BLE001
        result["available"] = False
        result["reason"] = f"一致性判断接口异常: {str(e)[:150]}"
        print(f"    [P1-4][Vision] 一致性判断失败(不阻断): {str(e)[:150]}")
        return result


_FINAL_CUT_PROMPT = (
    "你是影视成片阶段的【成片审片员】。下面是从同一部无声粗剪成片中按时间顺序抽取的画面。\n"
    "请从三个维度复审：\n"
    "1) 节奏（pacing）：相邻画面是否长时间雷同、有无明显拖沓或断裂感；\n"
    "2) 连续性（continuity）：场景/角色/光线是否有跳变穿帮、相邻画面是否像同一部片；\n"
    "3) 画面质量（quality）：有无明显黑场/占位帧/花屏/糊帧。\n"
    "必须严格只输出一个 JSON 对象，不要解释文字、不要 markdown 代码块：\n"
    '{"overall_score": 0到1的小数, "issues": [{"type": "pacing或continuity或quality",'
    ' "desc": "一句话中文问题描述", "severity": "low或medium或high"}]}\n'
    "issues 最多列 6 条，只列真实可从画面判断的问题；没有问题则给空数组。"
)


def critic_final_cut(video_path: str, max_frames: int = 4) -> Dict[str, Any]:
    """P1-2：成片级 Critic —— 对无声粗剪按时间顺序抽帧，交给视觉模型复审。

    与镜头级 evaluate_video 不同：本函数审的是「整片」——节奏断裂、跨镜头
    连续性穿帮、占位/黑场。产出结构化问题清单，随草稿审片卡呈现给用户，
    为「是否打回重拼 / 哪些镜头要修」提供依据。

    Returns:
        {"available", "score", "issues", "raw", "method"}。接口/解析失败时
        available=False（绝不阻断主流程）。
    """
    result: Dict[str, Any] = {
        "available": False,
        "score": None,
        "issues": [],
        "raw": "",
        "method": "vlm_final_cut",
    }
    if not video_path or not os.path.exists(video_path):
        result["raw"] = "成片文件不存在"
        return result

    try:
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            from moviepy import VideoFileClip

        clip = None
        frames: List[str] = []
        try:
            clip = VideoFileClip(video_path)
            duration = float(getattr(clip, "duration", 0.0) or 0.0)
            if duration <= 0:
                raise ValueError("视频时长无效")
            count = max(3, min(int(max_frames), 8))
            fracs = [ (i + 0.5) / count for i in range(count) ]
            for frac in fracs:
                t = max(0.05, min(duration * frac, max(0.05, duration - 0.05)))
                img = Image.fromarray(clip.get_frame(t))
                frames.append(_image_to_data_url(img, max_size=512, quality=82))
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass

        def _invoke(fr: List[str]) -> str:
            content = [{"type": "image_url", "image_url": {"url": u}} for u in fr]
            content.append({"type": "text", "text": _FINAL_CUT_PROMPT})
            response = client.chat.completions.create(
                model=_VISION_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=512,
            )
            return str(response.choices[0].message.content or "").strip()

        try:
            raw = _invoke(frames)
        except Exception as _ce:  # noqa: BLE001
            # glm-4v-flash 等视觉模型对单次多图数量有上限（实测 >4 张报 1210
            # 「输入图片数量超过限制」）。降到 3 帧重试一次；仍失败则按降级处理。
            _msg = str(_ce)
            if len(frames) > 3 and ("1210" in _msg or "数量超过" in _msg or "limit" in _msg.lower()):
                print(f"    [P1-2][FinalCritic] 图片数超模型上限，降为 3 帧重试")
                raw = _invoke(frames[:3])
                frames = frames[:3]
            else:
                raise
        result["raw"] = raw[:400]
        data = _extract_first_json(raw)
        if not data:
            result["available"] = True  # 接口通但输出非结构化
            return result

        score = data.get("overall_score")
        try:
            score = float(score)
            if score > 1.0:
                score = score / 100.0
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = None

        issues = []
        for it in (data.get("issues") or [])[:8]:
            if isinstance(it, dict) and it.get("desc"):
                issues.append({
                    "type": str(it.get("type", "other"))[:24],
                    "desc": str(it.get("desc", ""))[:200],
                    "severity": str(it.get("severity", "low"))[:12],
                })

        result["available"] = True
        result["score"] = score
        result["issues"] = issues
        return result
    except Exception as exc:  # noqa: BLE001
        result["raw"] = f"成片 Critic 异常: {str(exc)[:150]}"
        print(f"    [P1-2][FinalCritic] 失败(不阻断): {str(exc)[:150]}")
        return result


def evaluate_video(
    video_url: str,
    script: str,
    prompt_template: str,
    previous_image_url: Optional[str] = None,
    frame_count: int = 4
) -> str:
    """调用视觉模型审核视频片段：抽关键帧后做视频级审核。"""
    system_msg = prompt_template.format(script=script)

    tmp_path = None
    try:
        if video_url.startswith("http://") or video_url.startswith("https://"):
            tmp_path = _download_to_tempfile(video_url, suffix=".mp4")
            local_video_path = tmp_path
        else:
            local_video_path = video_url

        frame_data_urls = _sample_video_frames(local_video_path, frame_count=frame_count)

        content = []
        for frame_url in frame_data_urls:
            content.append({"type": "image_url", "image_url": {"url": frame_url}})

        if previous_image_url:
            content.append({"type": "image_url", "image_url": {"url": _resolve_image_input(previous_image_url)}})
            system_msg += "\n\n【重要】最后一张图片是前一个镜头的关键帧。请严格检查跨镜头视觉一致性。"

        system_msg += "\n\n这些图片按时间顺序表示同一个视频片段，请重点审核连续性、动作自然度、主体稳定性、镜头运动合理性。"
        content.append({"type": "text", "text": system_msg})

        response = client.chat.completions.create(
            model=_VISION_OMNI_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=512
        )

        result = str(response.choices[0].message.content or "").strip()

        if "PASS" in result.upper():
            print("    ✅ 视频审核通过！")
            return "PASS"

        print(f"    ❌ 视频审核打回: {result[:300]}...")
        return result

    except Exception as e:
        error_str = str(e).lower()

        if "data_inspection_failed" in error_str or "inappropriate content" in error_str:
            fail_msg = (
                "FAIL: Video content safety pre-check flagged this clip (data_inspection_failed). "
                "Please regenerate with the same subject and motion, but reduce realistic violence detail, "
                "avoid explicit weapon close-ups or blood effects, and shift toward a more stylized, cinematic, "
                "non-photorealistic artistic expression while preserving the scene's mood and atmosphere."
            )
            print("    ⚠️ [安全拦截] 视频审核绿网误判，已转为 FAIL 重绘")
            print(f"    {fail_msg}")
            return fail_msg

        print(f"    ⚠️ [视频审核员接口异常] {str(e)[:200]}")
        print("    转为人工审核（不再自动放行）")
        # 返回统一标记：审核模型不可用（如免费额度耗尽 403），供上游跳过无效重试
        return f"REVIEWER_UNAVAILABLE: Video reviewer API error — please manually check this clip. Error: {str(e)[:150]}"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def design_camera_movement(
    image_url: str,
    script: str,
    prompt_template: str,
    critique: str = "无",
    camera_context: str = "",
    cross_scene_context: str = "",
    **context: str,
) -> str:
    """设计视频运镜/运动提示词。

    由全模态模型改为复用 `text_llm.call_llm`（文本 qwen fallback 链，带超时/重试/
    多模型切换），并把输出上限提到 1536（对齐首尾帧的导演角色）。由于文本模型
    看不到首帧图，调用方需通过 `context` 注入首帧英文描述 `image_prompt` 作为
    「画面锚点」，并注入 global_setting / style_suffix / action_beat / visual_elements
    / first_frame_prompt / last_frame_prompt 等上下文，以弥补视觉缺失。

    Args:
        image_url: 首帧图 URL（单帧 i2v 模式），仅作日志/兜底展示，不再喂给多模态。
        script: 场景剧本。
        prompt_template: 视频 prompt 模板（含 ``{script}``/``{critique}`` 等占位符）。
        critique: 上一轮视频审核反馈。
        camera_context: 景别/动作焦点等摄影参数上下文。
        cross_scene_context: 跨场景连续性上下文。
        **context: 模板所需的其他上下文（image_prompt/global_setting/style_suffix/
            action_beat/visual_elements/first_frame_prompt/last_frame_prompt 等）。

    Returns:
        纯英文视频 motion prompt；异常时返回一个保守兜底运镜描述。
    """
    from .text_llm import call_llm

    # 用画面锚点 + 风格上下文补齐文本模型缺失的视觉信息
    fill = dict(
        script=script or "",
        critique=critique or "无",
        image_prompt=context.get("image_prompt", ""),
        global_setting=context.get("global_setting", ""),
        style_suffix=context.get("style_suffix", ""),
        action_beat=context.get("action_beat", "（无，请从 script 与首帧锚点中推断主体可见动作）"),
        visual_elements=context.get("visual_elements", ""),
        first_frame_prompt=context.get("first_frame_prompt", context.get("image_prompt", "")),
        last_frame_prompt=context.get("last_frame_prompt", ""),
        environment_anchor=context.get("environment_anchor", ""),
    )
    system_msg = prompt_template.format(**fill)

    # Inject camera context (shot distance, action focus, angle)
    if camera_context:
        system_msg += f"\n\n{camera_context}"

    # Inject cross-scene continuity context
    if cross_scene_context:
        system_msg += f"\n\n{cross_scene_context}"

    try:
        # 显式 1536，与首尾帧导演角色对齐；文本 qwen 链自带 fallback/重试
        return call_llm(system_msg, "摄影师", max_tokens=1536).strip()
    except Exception as e:
        print(f"    ⚠️ [摄影师接口异常] {str(e)}")
        return "Slow pan right, cinematic lighting, highly detailed."