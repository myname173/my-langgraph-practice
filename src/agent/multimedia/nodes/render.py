# src/agent/multimedia/nodes/render.py
"""渲染阶段：关键帧 / 尾帧 / 视频生成 + 各审核闸口（外部模型调用）。

P2-3：由原 graph.py（单体）按 stage 机械拆分而来，逻辑逐字节保留；
仅新增本文件头注释与模块导入，函数体 / 常量 / 注释均未改动。
"""
import os
import os as _os
import shutil
import time
import traceback
from langchain_core.runnables import RunnableConfig
try:
    from ..tools.embedding import compute_similarity, get_image_embedding
except Exception:
    compute_similarity = None
    get_image_embedding = None
from .. import run_metrics
from ..config_loader import make_placeholder_clip as _make_placeholder_clip
from ..prompts import REVIEWER_SYSTEM_PROMPT, VIDEO_REVIEWER_SYSTEM_PROMPT
from ..shot_strategy import get_consistency_threshold
from ..state import MultimediaState
from ..tools.image_gen import generate_keyframe
from ..tools.video_gen import FreeTierQuotaExhaustedError, _MOTION_NEGATIVE_PROMPT, generate_video_from_image
from ..tools.video_stitcher import download_video
from ..tools.media_paths import output_dir as _output_dir
from ..tools.vision_eval import assess_character_consistency, evaluate_image, evaluate_video
from .common import APPROVE_ACTIONS, CHARACTER_CONSISTENCY_THRESHOLD, FIX_EXTEND, FIX_KEEP_FIRST_FRAME, FIX_PROMPT_ONLY, FIX_REIMAGE, MAX_VIDEO_GEN_FAILURES, VLM_CONSISTENCY_THRESHOLD, _CONSISTENCY_VLM_ENABLED, _DEFAULT_STYLE_KEY, _STYLE_QUALITY_BAR, _decide, _is_review_pass, _normalize_fix_type
from .design import _filter_reference_images, _first_url, _is_turnaround_sheet, _match_reference_elements


# 防穿模块：以英文结构化块形式强制注入关键帧 prompt，不依赖 LLM 是否书写。
# 英文写法可让 _translate_to_english 直接跳过（省调用），并由 jimeng._fit_prompt
# 标记为最高优先级块（永不截断丢弃），从根本上抑制角色肢体畸变/穿模/多余肢体。
# 注意：只约束「画面质量」（解剖合理、不穿模、武器实体可辨），不约束角色身份/情节，
# 留给文本模型按用户输入自由发挥。措辞以引导为主，避免过死的禁令式定义。
_ANTI_DEFORMATION_BLOCK = (
    "[Visual Quality: keep the human figure anatomically natural — one head, two arms, "
    "two legs, hands and feet in believable proportion; avoid extra or missing limbs and "
    "grotesque mutations. Let characters and objects occupy space plausibly so nothing "
    "clips through scenery, terrain or props, and feet rest naturally on the ground. "
    "Render weapons as clear, tangible objects held in a natural grip, with light effects "
    "enhancing rather than swallowing them.]"
)


def image_gen_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- [Art] [Shot {idx+1}] Rendering keyframe ---")

    # ── 幂等守卫：续跑保留关键帧 ──
    # 若当前镜头已存在合法首帧（http/data URL/本地文件），说明关键帧已生成，
    # 续跑（input=None）时直接复用，避免重复消耗生图额度与重画构图。
    # 【修复死循环】例外：一致性闭环（P1-4 VLM / embedding 肖像相似度）判定需重生时
    # （needs_realign=True），必须跳过复用、强制重绘；否则会陷入
    # 「复用旧图 → 再审 → 仍判漂移 → 再复用」的死循环（iterations 永不递增）。
    _existing_img = str(scene.get("image_url") or "").strip()
    _rerender_requested = bool(scene.get("needs_realign"))
    if _existing_img and not _rerender_requested and (
        _existing_img.startswith("http")
        or _existing_img.startswith("data:image")
        or _existing_img.startswith("file://")
        or (len(_existing_img) > 1 and _existing_img[1] == ":" and os.path.isabs(_existing_img))
    ):
        print(f"    ⏭️ [Shot {idx+1}] 关键帧已存在，续跑复用（跳过生图）: {_existing_img[:80]}")
        return {"scenes": state["scenes"]}
    if _rerender_requested:
        # 消费重生请求（本次进入重绘流程）；reviewer 会依据新图重新判定并按需再次置位。
        scene.pop("needs_realign", None)
        print(f"    ♻️ [Shot {idx+1}] 一致性闭环要求重生，忽略已缓存关键帧，重新生图")

    scenes = state["scenes"].copy()

    # ── 复用素材库资产（assets_imported）──
    # 素材参考图(reference_sheets) 本质只是"参考/锚点"，不应直接当首尾帧。
    # 两种用法：
    #   A) MULTIMODIA_BACKEND != "siliconflow"：跳过关键帧生图，image_url 留空 →
    #      video_gen_node 走"无首帧 + 素材参考图"的 Agnes keyframes 模式，不消耗生图额度。
    #   B) MULTIMODIA_BACKEND == "siliconflow"：用素材参考图作 img2img 锚点，生成本镜头
    #      专属首/尾帧（SiliconFlow images/edits）。首尾帧是镜头专属构图，素材图仅作风格/身份锚定，
    #      符合大厂视频 Agent 把参考图融入生成的做法。
    if state.get("assets_imported"):
        from ..tools.image_gen import get_visual_backend
        _backend = get_visual_backend()
        # 后端支持 img2img 锚点生图（SiliconFlow / 即梦）时，复用素材参考图作锚点，
        # 生成本镜头专属首/尾帧——关键帧必须有内容，否则视频黑场。
        if _backend in ("siliconflow", "jimeng"):
            # 方案 B：img2img 生成镜头专属首/尾帧
            reference_sheets = state.get("reference_sheets") or {}
            script_text = scene.get("script", "").lower()
            ref_match = _match_reference_elements(reference_sheets, script_text)
            # 构建「按角色名 → 专属肖像」映射：每个角色固定一张非 turnaround 立绘，
            # 保证同一角色在任意镜头都锚定到同一张脸（修复"每镜脸不同"的核心 bug）。
            _character_portraits = {}
            for _cn, _cd in (reference_sheets.get("characters") or {}).items():
                if isinstance(_cd, dict) and not _is_turnaround_sheet(_cd):
                    _u = _first_url(_cd.get("urls") or _cd.get("url"))
                    if _u:
                        _character_portraits[_cn] = _u

            # 锚点优先级：命中角色专属肖像 > 命中角色全部图 > 环境 > 全局统一肖像 > 全部素材
            # 回退不再按镜头索引轮转（旧 idx%len 会让同一角色跨镜锚到不同立绘 → 脸漂移），
            # 改为统一全局肖像，确保身份一致优先于首帧差异化。
            _matched_name = ref_match.get("matched_char_name")
            _anchor_by_name = _character_portraits.get(_matched_name) if _matched_name else None
            anchor_urls = []
            if _anchor_by_name:
                anchor_urls = [_anchor_by_name]
            elif ref_match.get("char_urls"):
                anchor_urls = (ref_match.get("char_urls") or [])[:1]
            elif ref_match.get("env_urls"):
                anchor_urls = (ref_match.get("env_urls") or [])[:1]
            if not anchor_urls:
                if state.get("character_portrait_url"):
                    anchor_urls = [state["character_portrait_url"]]
                else:
                    _all = []
                    for _cat in ("characters", "props", "environments"):
                        for _v in (reference_sheets.get(_cat) or {}).values():
                            if _v and not _is_turnaround_sheet(_v):
                                _all.append(_first_url(_v.get("urls") or _v.get("url")))
                    anchor_urls = [u for u in _all if u][:3]

            # 排除整张 turnaround/多视图网格图，避免转面网格渗入首帧。
            # 【实测修正】ref_strength 0.5 时产物与素材图相似度仅 0.2469，
            # 甚至低于"完全不带参考图的纯文生图"(0.2582)，等于参考图白费。
            # 实测 jimeng img2img：strength=0.5→0.2469，strength=0.9→0.2995。
            # 统一提至 0.9 上限，让参考图（尤其角色专属肖像）真正产生身份约束力
            # （jimeng 免费档无更强 img2img 后端，0.9 为可用峰值）。
            anchor_urls = [u for u in (anchor_urls or []) if u and not _is_turnaround_sheet({
                "url": u,
            })]
            # 分镜视觉承接：并入上一镜结尾帧，使每个新镜头开头承接上一镜（修复"分镜无联系"）。
            if idx > 0:
                _sa = state.get("scene_anchors") or {}
                _prev = _sa.get("last_frame_url")
                if _prev and _prev not in anchor_urls:
                    anchor_urls = [_prev] + list(anchor_urls)
            # 【修复-C】用户要求：关键帧生图失败时应退避重试，而非静默降级为无首帧/纯文模式。
            # generate_keyframe 内部已对即梦「积分不足/权益不足」(-2001 伪装) 做 4 次长退避重试
            # (8/20/40/60s) 等免费权益恢复；此处外层仅做 2 次兜底（间隔更长），避免内层重试窗口
            # 仍不够时彻底失败。仅当重试耗尽仍失败才依次降级（先纯文保底、再无首帧）。
            # 【修复-穿模】prompt 强制追加防穿模块（_ANTI_DEFORMATION_BLOCK），且 img2img 重试时
            # 降强度（0.85→0.5）缓解限流；纯文/无首帧失败均真实计入 video_gen_failures，
            # 不允许静默无首帧放行（避免视频阶段退化成素材图直入帧、角色面目全非/穿模）。
            def _gen_with_backoff(urls, strength=None, attempts=2, extra_prompt=""):
                _delay = 15
                _last = None
                _base = (scene["image_prompt"] or "").strip()
                if extra_prompt:
                    _base = _base + "\n" + extra_prompt
                for _i in range(attempts):
                    try:
                        if urls:
                            return generate_keyframe(_base, reference_image_urls=urls,
                                                      ref_strength=strength, thread_id=state.get("thread_id", ""))
                        return generate_keyframe(_base, thread_id=state.get("thread_id", ""))
                    except Exception as _e:
                        _last = _e
                        if _i < attempts - 1:
                            print(f"    [RETRY] 镜头 {idx+1} 关键帧生图第{_i+1}次失败，{_delay:.1f}s 后重试: {_e}")
                            time.sleep(_delay)
                            _delay *= 2
                raise _last

            if anchor_urls:
                print(f"    [Asset+SiliconFlow] 以 {len(anchor_urls)} 张素材参考图作 img2img 锚点（已排除 turnaround），"
                      f"生成镜头 {idx+1} 专属首/尾帧...")
                try:
                    first_url = _gen_with_backoff(anchor_urls, strength=0.9,
                                                  extra_prompt=_ANTI_DEFORMATION_BLOCK)
                    scenes[idx]["image_url"] = first_url
                    # 取消首尾帧强制同源：尾帧不再锁死为首帧，
                    # 留空后交由 end_frame_director → end_frame_generator 基于 end shot 独立生成，
                    # 解锁首尾帧之间的运动区间（首帧=hero shot 构图，尾帧=end shot 构图，二者不同源）。
                    scenes[idx].pop("last_image_url", None)
                    scenes[idx]["iterations"] += 1
                    print(f"    [OK] 镜头 {idx+1} 首帧已生成(img2img)，尾帧交由 end_frame_director 独立生成")
                    return {"scenes": scenes, "use_first_last_frame": True}
                except Exception as e:
                    print(f"    [WARN] img2img(强度0.85)生图重试耗尽，降强度0.5 重试(镜头 {idx+1}): {e}")
                    try:
                        first_url = _gen_with_backoff(anchor_urls, strength=0.5,
                                                      extra_prompt=_ANTI_DEFORMATION_BLOCK)
                        scenes[idx]["image_url"] = first_url
                        scenes[idx].pop("last_image_url", None)
                        scenes[idx]["iterations"] += 1
                        print(f"    [OK] 镜头 {idx+1} 首帧已生成(img2img 降强度0.5保底)")
                        return {"scenes": scenes, "use_first_last_frame": True}
                    except Exception as e1:
                        print(f"    [WARN] img2img 降强度也失败，降级为纯文生图保底(镜头 {idx+1}): {e1}")
                        try:
                            first_url = _gen_with_backoff(None, extra_prompt=_ANTI_DEFORMATION_BLOCK)
                            scenes[idx]["image_url"] = first_url
                            scenes[idx].pop("last_image_url", None)
                            scenes[idx]["iterations"] += 1
                            print(f"    [OK] 镜头 {idx+1} 首/尾帧已生成(纯文生图保底)")
                            return {"scenes": scenes, "use_first_last_frame": True}
                        except Exception as e2:
                            # 真实计入失败计数，交由路由节点按 MAX_VIDEO_GEN_FAILURES 决定补跑/跳过。
                            # 绝不静默无首帧放行（否则视频阶段退化成素材图直入帧 → 角色面目全非/穿模）。
                            print(f"    ❌ [Art] [Shot {idx+1}] 关键帧生图彻底失败: {e2}")
                            scenes[idx]["video_gen_failures"] = scenes[idx].get("video_gen_failures", 0) + 1
                            return {"scenes": scenes}
            # 【修复】后端支持生图（jimeng/siliconflow）但无锚点图时，仍须为本镜头生成专属首帧，
            # 否则 image_url 留空 → agnes 退化成「仅素材参考图+雷同prompt」生视频，导致分镜雷同/原图入帧。
            # 不再 fall-through 到清空首帧，而是纯文生图保底，确保每个镜头有差异化关键帧。
            print(f"    [Asset] 无素材锚点，纯文生图保底生成镜头 {idx+1} 专属首帧（jimeng 后端）...")
            try:
                first_url = _gen_with_backoff(None, extra_prompt=_ANTI_DEFORMATION_BLOCK)
                scenes[idx]["image_url"] = first_url
                scenes[idx].pop("last_image_url", None)
                scenes[idx]["iterations"] += 1
                print(f"    [OK] 镜头 {idx+1} 首帧已生成(纯文生图保底)，视频阶段以此为专属首帧")
                return {"scenes": scenes, "use_first_last_frame": True}
            except Exception as e2:
                print(f"    ❌ [Art] [Shot {idx+1}] 纯文生图保底也失败: {e2}")
                scenes[idx]["video_gen_failures"] = scenes[idx].get("video_gen_failures", 0) + 1
                return {"scenes": scenes}
        # 方案 A（默认）
        print(f"    [Asset] 复用素材库模式：跳过即梦关键帧生图(镜头 {idx+1})，视频阶段以素材参考图作锚点")
        # 清空任何残留的首尾帧，确保不会把素材图当作首尾帧注入视频
        scenes[idx].pop("image_url", None)
        scenes[idx].pop("last_image_url", None)
        return {"scenes": scenes, "use_first_last_frame": False}

    # ── 选择参考图（reference image）用于 img2img 一致性控制 ──
    # 优先级: 上一场景尾帧(FLF连续性) > 匹配角色参考 > 环境参考 > 无参考
    #
    # FLF 模式核心理念（参考 CoAgent GCM + Script-is-All-You-Need 帧锚定方案）:
    #   prev_end_frame 作为 img2img 参考图(ref_strength=0.5) -> 导演 prompt 主导构图
    #   -> end_frame_gen 从 keyframe 生成尾帧 -> kf2v(keyframe, end_frame) 渲染视频
    # 这样既保持场景间像素级连续性，又不放弃导演对每个镜头起始构图的精确控制
    reference_url = None
    ref_strength = 1.0

    use_flf = state.get("use_first_last_frame", False)
    scene_anchors = state.get("scene_anchors") or {}
    prev_end_frame = scene_anchors.get("last_frame_url") if idx > 0 else None

    reference_sheets = state.get("reference_sheets") or {}
    script_text = scene.get("script", "").lower()

    # 源头角色肖像（showrunner 通过后生成的标准角色 ID 基准）。
    # 关键：所有场景对齐"同一个源头基准"，抑制链式(逐场景)传递导致的角色漂移(drift)。
    character_portrait_url = state.get("character_portrait_url")

    # ── 使用 helper 匹配当前场景涉及的角色/道具/环境 ──
    ref_match = _match_reference_elements(reference_sheets, script_text)
    matched_char_urls = ref_match["char_urls"]
    matched_char_desc = ref_match["char_desc"]
    matched_prop_desc = ref_match["prop_desc"]
    if ref_match["matched_char_name"]:
        print(f"    [*] Matched character reference: {ref_match['matched_char_name']} ({len(matched_char_urls or [])} ref image(s))")
    if ref_match["matched_prop_name"]:
        print(f"    [*] Matched prop reference: {ref_match['matched_prop_name']}")

    # ── 材质匹配可见化：把当前镜头命中信息累加进 asset_match_report ──
    # image_gen_node 每个镜头调用一次，从 state 取已有报告并就地累加命中镜头数，
    # 使前端能展示「这张用户参考图被多少个镜头实际用上」。
    _report = dict(state.get("asset_match_report") or {})
    _report.setdefault("matched", {})
    _report.setdefault("unmatched", [])
    for _m in (ref_match.get("matched_char_name"), ref_match.get("matched_prop_name")):
        if _m:
            _report["matched"][_m] = int(_report["matched"].get(_m, 0)) + 1
            # 一旦被镜头实际命中即移出「未命中」名单
            if _m in _report["unmatched"]:
                _report["unmatched"].remove(_m)

    if use_flf and prev_end_frame and idx > 0:
        # FLF 模式: 上一场景尾帧 -> img2img 参考，并并入命中角色专属肖像（双锚点）。
        # 上一镜结尾帧保证"场景首尾视觉承接"，角色肖像保证"同一角色跨镜同脸"，
        # 二者共同修复"分镜之间无联系 / 每镜脸不同"。ref_strength 0.6 略强于旧 0.5，
        # 但仍低于角色肖像侧，避免被上一镜背景带偏构图。
        reference_urls = [prev_end_frame] + (matched_char_urls or [])
        ref_strength = 0.6
        print(f"    [*] FLF mode: prev scene end frame + character portrait as img2img refs (ref_strength={ref_strength})")
    elif matched_char_urls:
        # 匹配到的角色参考图（多张全部注入做融合，比单张更精准一致）
        reference_urls = matched_char_urls
        ref_strength = 0.5
        print(f"    [*] Using matched character reference ({len(matched_char_urls)} image(s), ref_strength={ref_strength})")
    elif character_portrait_url:
        # 源头基准兜底：无 FLF 尾帧 / 无匹配角色参考时，用源头角色肖像做 img2img，
        # 保证角色身份对齐源头基准，而非退化为无参考(漂移风险最高)。
        reference_urls = [character_portrait_url]
        ref_strength = 0.4
        print(f"    [*] Using source character portrait as anchor reference (ref_strength={ref_strength})")
    elif ref_match["env_urls"]:
        # Fallback: 环境参考图（多张融合）
        reference_urls = ref_match["env_urls"]
        ref_strength = 0.35
        print(f"    [*] Using environment reference ({len(ref_match['env_urls'])} image(s), ref_strength={ref_strength})")
    else:
        reference_urls = None

    # ── 文本增强：在 prompt 中注入角色/道具描述，强化一致性 ──
    enhanced_prompt = scene["image_prompt"]
    identity_anchors = []
    if matched_char_desc:
        identity_anchors.append(f"[Character Identity Anchor: {matched_char_desc}]")
    if matched_prop_desc:
        identity_anchors.append(f"[Prop Identity Anchor: {matched_prop_desc}]")
    # 角色肖像文本级恒定锚点：即便本帧参考图用的是"上一场景尾帧"(链式)，
    # 也始终提示模型对齐源头角色外观，进一步抑制跨场景漂移。
    if character_portrait_url and reference_urls != [character_portrait_url]:
        identity_anchors.append(
            "[Character Consistency: keep the SAME character identity as the "
            "established source portrait — same face, hairstyle, outfit, and equipment.]"
        )
    if identity_anchors:
        enhanced_prompt = enhanced_prompt + "\n" + "\n".join(identity_anchors)
        print(f"    [*] Injected {len(identity_anchors)} identity anchor(s) into prompt")
    # 画面质量约束（仅质量，不约束角色/情节）：抑制肢体畸变与穿模，引导武器实体呈现。
    enhanced_prompt = enhanced_prompt + "\n" + _ANTI_DEFORMATION_BLOCK
    print(f"    [*] Injected visual-quality block into keyframe prompt")

    # 本地关键帧注入标记：当 KEYFRAME_LOCAL_DIR 设置时，jimeng.generate_image 会按
    # [KF#idx] 定位外部（AI 生图/人工）预生成的镜头关键帧，跳过即梦额度消耗。
    kf_tag = f" [KF#{idx}]"
    try:
        if reference_urls:
            image_url = generate_keyframe(enhanced_prompt + kf_tag,
                                          reference_image_urls=reference_urls,
                                          ref_strength=ref_strength,
                                          thread_id=state.get("thread_id", ""))
        else:
            image_url = generate_keyframe(enhanced_prompt + kf_tag,
                                          thread_id=state.get("thread_id", ""))
    except Exception as e:
        # 生图失败（网络瞬断/接口错误）不再直接 FATAL，改为计入失败计数，
        # 交由路由节点按 MAX_VIDEO_GEN_FAILURES 决定重试或跳过该镜头。
        print(f"    ❌ [Art] [Shot {idx+1}] 关键帧生图失败: {e}")
        scenes[idx]["video_gen_failures"] = scenes[idx].get("video_gen_failures", 0) + 1
        return {"scenes": scenes}

    scenes[idx]["image_url"] = image_url
    # F-1：候选历史 —— 每次生成把关键帧登记进候选列表，供审核卡 A/B 挑选
    # （重绘 fix_type=reimage 会再次进入本节点，从而积累第 2、3 版候选）。
    _cands = [u for u in (scenes[idx].get("image_candidates") or []) if u]
    if image_url and image_url not in _cands:
        _cands.append(image_url)
    scenes[idx]["image_candidates"] = _cands[-6:]
    # P2-4：记录首帧渲染参数（配方卡）——含后端/尺寸/参考图/参考强度，供复现与微调。
    _rp_img = dict(scenes[idx].get("render_params") or {})
    _rp_img["image"] = {
        "backend": "jimeng",
        "model": os.environ.get("JIMENG_MODEL", ""),
        "prompt": enhanced_prompt,
        "size": "2K",
        "ref_images": [str(u) for u in (reference_urls or []) if u],
        "ref_strength": ref_strength,
    }
    scenes[idx]["render_params"] = _rp_img
    scenes[idx]["iterations"] += 1
    return {"scenes": scenes, "asset_match_report": _report}


def reviewer_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🧐 [镜头 {idx+1}] 艺术总监 + 一致性引擎审核 ---")

    # 【修复】显式获取上一镜头的图片，优先取尾帧
    previous_image_url = None
    if idx > 0:
        prev_scene = state["scenes"][idx - 1]
        previous_image_url = prev_scene.get("last_image_url") or prev_scene.get("image_url")
        if previous_image_url:
            print(f"    [*] 正在对比上一镜头关键帧: {previous_image_url[-30:]}...")

    similarity_score = None
    if previous_image_url and compute_similarity is not None:
        try:
            similarity_score = compute_similarity(scene["image_url"], previous_image_url)
            print(f"    [*] embedding 相似度(相邻镜头): {similarity_score:.4f}")
        except Exception as e:
            print(f"    ⚠️ embedding 计算失败，继续人工审核: {str(e)}")

    # ── 角色一致性闭环度量：与"源头角色肖像"的相似度 ──
    # 源头肖像是全局角色 ID 基准，用它度量角色漂移比"与上一镜头比"更抗累积误差。
    portrait_similarity = None
    portrait_url = state.get("character_portrait_url")
    if portrait_url and compute_similarity is not None:
        try:
            portrait_similarity = compute_similarity(scene["image_url"], portrait_url)
            print(f"    [*] embedding 相似度(源头肖像): {portrait_similarity:.4f}")
        except Exception as e:
            print(f"    ⚠️ 肖像相似度计算失败(不阻断): {str(e)}")

    # 连贯性可观测分数：相邻镜头相似度 + 源头肖像相似度（任一可缺失）
    continuity_score = {
        "neighbor_similarity": similarity_score,
        "portrait_similarity": portrait_similarity,
    }

    # ── P1-4：一致性强制层 —— embedding 缺失时用 VLM 成对判断兜底 ──
    # embedding 通道（DashScope multimodal-embedding）失效时，similarity 恒为 None，
    # 下游一致性闭环会静默降级。这里补一条 VLM 数据源：把当前帧与参照帧（源头肖像
    # 优先，其次上一镜尾帧）喂给视觉模型，结构化输出「是否同一角色」。
    vlm_consistency = None
    if _CONSISTENCY_VLM_ENABLED and (similarity_score is None or portrait_similarity is None):
        ref_for_vlm = portrait_url or previous_image_url
        if not ref_for_vlm:
            # 分支机制（并行/链式）下 scenes[idx-1] 常为空——快照期其余镜头尚未生成图像，
            # previous_image_url 恒 None。前镜尾帧经由 ctx.scene_anchors.last_frame_url 传递
            # （P1-1 链式承接通道，image_gen 亦读此处），这里作为兜底参照帧。
            ref_for_vlm = (state.get("scene_anchors") or {}).get("last_frame_url")
        if ref_for_vlm and scene.get("image_url"):
            try:
                thr_vlm, relax_vlm = get_consistency_threshold(
                    VLM_CONSISTENCY_THRESHOLD, scene.get("script", "")
                )
                vlm_consistency = assess_character_consistency(
                    scene["image_url"], ref_for_vlm, threshold=thr_vlm,
                )
                vlm_consistency["relax_reason"] = relax_vlm
                if vlm_consistency.get("score") is not None:
                    note = f"，已按 {relax_vlm} 放宽" if relax_vlm else ""
                    print(
                        f"    [P1-4] VLM 角色一致性 {vlm_consistency['score']:.3f} "
                        f"(阈值 {thr_vlm:.3f}{note}) · drifting={vlm_consistency['drifting']}"
                    )
            except Exception as e:  # noqa: BLE001
                print(f"    [P1-4] VLM 一致性判断失败(不阻断): {str(e)[:150]}")

    # 预注入风格质量标准（evaluate_image 内部会继续 format script）
    style_key = state.get("visual_style") or _DEFAULT_STYLE_KEY
    quality_bar = _STYLE_QUALITY_BAR.get(style_key, _STYLE_QUALITY_BAR[_DEFAULT_STYLE_KEY])
    reviewer_prompt = REVIEWER_SYSTEM_PROMPT.replace("{quality_bar}", quality_bar)

    feedback = evaluate_image(
        scene["image_url"],
        scene["script"],
        reviewer_prompt,
        previous_image_url=previous_image_url
    )

    scenes = state["scenes"].copy()
    scenes[idx]["image_auto_feedback"] = feedback
    scenes[idx]["embedding_similarity"] = similarity_score
    scenes[idx]["portrait_similarity"] = portrait_similarity
    scenes[idx]["continuity_score"] = continuity_score
    scenes[idx]["critique"] = feedback
    scenes[idx]["is_perfect"] = _is_review_pass(feedback)

    # P1-4：记录 VLM 一致性结论（needs_realign 统一在下方按 VLM/embedding 判定后置位）
    if vlm_consistency is not None:
        scenes[idx]["vlm_consistency"] = vlm_consistency

    # 审核视觉模型不可用（如免费额度耗尽 403）：记录全局标志，供路由跳过"一致性强制重生"等无效重试
    reviewer_unavailable = feedback.startswith("REVIEWER_UNAVAILABLE:")
    if reviewer_unavailable:
        print("    ⚠️ [审核模型不可用] 后续镜头的一致性强制重生将被跳过（重生也无法提分，避免空耗额度）")

    # ── 一致性强制重生的统一置位与硬上界 ──
    # 触发源二选一：P1-4 VLM 判定漂移，或 embedding 肖像相似度低于阈值。
    # consistency_attempts 每判定一次「需重生」就 +1（reviewer 每个循环 pass 执行一次），
    # 作为死循环硬上界：即使重绘持续失败，decide 侧也会随计数增长而收敛退出。
    if not reviewer_unavailable:
        _cons_thr, _ = get_consistency_threshold(
            CHARACTER_CONSISTENCY_THRESHOLD, scene.get("script", "")
        )
        _drift_vlm = bool((vlm_consistency or {}).get("drifting"))
        _drift_emb = (
            portrait_similarity is not None and _cons_thr > 0
            and portrait_similarity < _cons_thr
        )
        if _drift_vlm or _drift_emb:
            scenes[idx]["needs_realign"] = True
            scenes[idx]["consistency_attempts"] = int(scenes[idx].get("consistency_attempts", 0)) + 1
        else:
            scenes[idx]["needs_realign"] = False

    return {
        "scenes": scenes,
        "reviewer_unavailable": reviewer_unavailable or bool(state.get("reviewer_unavailable")),
    }


def image_review_gate_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    # 【修复】显式获取上一镜头的图片
    previous_image_url = None
    if idx > 0:
        prev_scene = state["scenes"][idx - 1]
        previous_image_url = prev_scene.get("last_image_url") or prev_scene.get("image_url")

    payload = {
        "stage": "image_review",
        "title": f"镜头 {idx + 1} 关键帧审片",
        "scene_index": idx + 1,
        "script": scene["script"],
        "global_setting": state.get("global_setting", ""),
        "image_prompt": scene.get("image_prompt", ""),
        "image_url": scene.get("image_url", ""),
        "reference_image_url": previous_image_url,
        "embedding_similarity": scene.get("embedding_similarity"),
        "portrait_similarity": scene.get("portrait_similarity"),
        "continuity_score": scene.get("continuity_score"),
        "auto_feedback": scene.get("image_auto_feedback", scene.get("critique", "")),
        "message": "请审核关键帧：可通过、重写，或修改 image_prompt 后继续。",
        "actions":["approve", "rewrite", "edit_prompt"],
        # P1-3：外科手术可选修正类型（前端渲染为按钮）
        "fix_types": ["reimage", "keep_first_frame", "prompt_only", "extend"],
        # F-1：候选关键帧（最近 4 张，含当前）—— 审核卡可挑选后随 approve 提交 pick_candidate
        "candidates": [u for u in (scene.get("image_candidates") or []) if u][-4:],
    }

    decision = _decide("image_review", payload, state)
    action = decision.get("action", "approve")

    scenes = state["scenes"].copy()

    if decision.get("image_prompt") is not None:
        scenes[idx]["image_prompt"] = str(decision["image_prompt"]).strip()

    # P1-3：记录外科手术修正类型；未通过时默认为 reimage（重绘关键帧，等价旧全量行为）。
    _fix = _normalize_fix_type(decision.get("fix_type"))
    scenes[idx]["fix_type"] = _fix or FIX_REIMAGE
    if _fix in (FIX_KEEP_FIRST_FRAME, FIX_PROMPT_ONLY, FIX_EXTEND):
        print(f"    [P1-3] 镜头 {idx+1} 修正类型={_fix}（保留关键帧，跳过重绘，仅修视频侧）")

    # F-1：候选挑选 —— 用户在审核卡上从历史候选中选定某一版关键帧后通过。
    # 必须在 approve 写 reference_images/embeddings 之前落位，保证基准一致性。
    _cands = [u for u in (scene.get("image_candidates") or []) if u][-4:]
    _pick = decision.get("pick_candidate")
    if (
        isinstance(_pick, int)
        and not isinstance(_pick, bool)
        and 0 <= _pick < len(_cands)
        and _cands[_pick]
    ):
        if scenes[idx].get("image_url") != _cands[_pick]:
            print(f"    [F-1] 镜头 {idx+1} 选用候选 #{_pick + 1} 关键帧")
        scenes[idx]["image_url"] = _cands[_pick]
        scenes[idx]["last_image_url"] = scenes[idx].get("last_image_url") or _cands[_pick]

    if action in APPROVE_ACTIONS:
        scenes[idx]["is_perfect"] = True
        scenes[idx]["critique"] = "无"

        ref_images = state.get("reference_images",[]).copy()
        ref_images.append(scenes[idx]["image_url"])

        updates = {
            "scenes": scenes,
            "reference_images": ref_images,
        }

        if get_image_embedding is not None:
            try:
                ref_embs = state.get("reference_embeddings",[]).copy()
                ref_embs.append(get_image_embedding(scenes[idx]["image_url"]))
                updates["reference_embeddings"] = ref_embs
            except Exception as e:
                print(f"    ⚠️ 记录 embedding 失败，但不影响主流程: {str(e)}")

        return updates

    scenes[idx]["is_perfect"] = False
    scenes[idx]["critique"] = decision.get("reason") or scene.get("image_auto_feedback") or "Human requested rewrite"
    return {"scenes": scenes}


def end_frame_gen_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🎨[镜头 {idx+1}] 画师渲染【尾帧】 ---", flush=True)

    # ── 幂等守卫：续跑保留尾帧 ──
    # 若当前镜头已存在合法尾帧（http/data URL/本地文件），说明尾帧已生成，
    # 续跑（input=None）时直接复用，避免重复消耗生图额度与重画构图。
    _existing_last = scene.get("last_image_url")
    if _existing_last and (
        _existing_last.startswith("http") or os.path.isfile(_existing_last)
        or _existing_last.startswith("data:")
    ):
        print(f"    [*] 复用已有尾帧（{_existing_last[:60]}...），跳过重复生图", flush=True)
        return {"scenes": state["scenes"]}

    # 动态计算 ref_strength：尾帧需要与首帧产生足够大的视觉差异，
    # 这样 kf2v 视频模型才有足够的插值空间来生成动态运动。
    # ref_strength 越低 → 图编模型越自由 → 尾帧与首帧差异越大 → 视频动作幅度越大
    #
    # 【实测修正】原默认 0.08 / 最低 0.02，经 jimeng clamp 到下限 0.1 后参考图几无约束力，
    # 尾帧角色与首帧漂移严重。实测 jimeng img2img 强度-相似度：
    #   0.5 → 0.2469（低于纯文生图 0.2582，等于无效）
    #   0.9 → 0.2995（该后端实测上限）
    # 为保证"尾帧角色与首帧一致"，将整体区间上移：默认 0.45，动作场景 0.35，静态场景 0.6。
    ref_strength = 0.45  # 默认：保留首帧角色/构图，同时留出运动差异空间
    shot_plan = state.get("shot_plan", {})
    script_lower = scene.get("script", "").lower()

    # 从 shot_plan 的 pacing_weight 推断运动幅度
    if shot_plan and shot_plan.get("shots"):
        hero_idx = shot_plan.get("hero_shot_index", 0)
        pacing = shot_plan["shots"][hero_idx].get("pacing_weight", 0.25)
        if pacing >= 0.35:
            ref_strength = 0.35  # 高能量动作场景：较低参考，允许较剧烈姿态变化
        elif pacing <= 0.15:
            ref_strength = 0.6   # 静态/情感场景：高参考，保持角色与构图稳定
        else:
            ref_strength = 0.45  # 中等

    # 剧本关键词微调
    action_kws = ["战斗", "冲锋", "追逐", "挥剑", "爆炸", "combat", "chase", "attack", "explosion", "slash"]
    calm_kws = ["回忆", "告别", "沉默", "微笑", "祈祷", "tears", "farewell", "silent", "whisper", "hope"]
    if any(kw in script_lower for kw in action_kws):
        ref_strength = min(ref_strength, 0.35)
    elif any(kw in script_lower for kw in calm_kws):
        ref_strength = max(ref_strength, 0.6)

    # 尾帧负面提示词：抑制与首帧相同姿态的静态输出
    end_frame_negative = (
        "same pose as reference, identical composition, static, frozen, "
        "no movement, copy of reference image, unchanged position, "
        "stiff body, no expression change, duplicate"
    )

    # ── 注入 identity anchors 到尾帧 prompt，防止低 ref_strength 下角色外貌漂移 ──
    end_frame_prompt = scene["last_image_prompt"]
    reference_sheets = state.get("reference_sheets") or {}
    script_text = scene.get("script", "").lower()
    ref_match = _match_reference_elements(reference_sheets, script_text)
    end_frame_anchors = []
    if ref_match["char_desc"]:
        end_frame_anchors.append(f"[Character Identity Anchor: {ref_match['char_desc']}]")
    if ref_match["prop_desc"]:
        end_frame_anchors.append(f"[Prop Identity Anchor: {ref_match['prop_desc']}]")
    if end_frame_anchors:
        end_frame_prompt = end_frame_prompt + "\n" + "\n".join(end_frame_anchors)

    scenes = state["scenes"].copy()
    # 【修复-C】尾帧生图同样优先退避重试（覆盖代理瞬时抖动 / 免费档限流），而非直接判失败。
    # 但严格限制重试耗时（timeout=60s × 2次 + 短退避），避免单镜尾帧卡死整条流水线。
    last_image_url = None
    _delay = 5
    for _i in range(2):
        try:
            last_image_url = generate_keyframe(
                end_frame_prompt,
                reference_image_urls=[scene["image_url"]],
                ref_strength=ref_strength,
                negative_prompt=end_frame_negative,
                thread_id=state.get("thread_id", ""),
                timeout=60,
            )
            break
        except Exception as _e:
            traceback.print_exc()
            if _i < 1:
                print(f"    [RETRY] 镜头 {idx+1} 尾帧生图第{_i+1}次失败，{_delay:.1f}s 后重试: {_e}", flush=True)
                time.sleep(_delay)
                _delay *= 2
            else:
                # 尾帧生图重试耗尽不 FATAL，计入失败计数交由路由处理（重试/跳过该镜头）
                print(f"    ❌ [尾帧] [Shot {idx+1}] 尾帧生图重试耗尽失败: {_e}", flush=True)
                scenes[idx]["video_gen_failures"] = scenes[idx].get("video_gen_failures", 0) + 1
                return {"scenes": scenes}
    if last_image_url is None:
        print(f"    ❌ [尾帧] [Shot {idx+1}] 尾帧生图未产出")
        scenes[idx]["video_gen_failures"] = scenes[idx].get("video_gen_failures", 0) + 1
        return {"scenes": scenes}
    print(f"    [*] 尾帧参考强度: {ref_strength}（{'极低→大差异' if ref_strength <= 0.12 else '偏低→中等差异' if ref_strength <= 0.2 else '中等→保持连续'}）")

    scenes[idx]["last_image_url"] = last_image_url
    # P2-4：记录尾帧渲染参数（配方卡）。
    _rp_end = dict(scenes[idx].get("render_params") or {})
    _rp_end["end_frame"] = {
        "prompt": end_frame_prompt,
        "ref_strength": ref_strength,
        "ref_images": [scene["image_url"]] if scene.get("image_url") else [],
    }
    scenes[idx]["render_params"] = _rp_end
    scenes[idx]["last_image_iterations"] += 1
    return {"scenes": scenes}


def end_frame_review_gate_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    payload = {
        "stage": "end_frame_review",
        "title": f"镜头 {idx + 1} 【尾帧】审片",
        "scene_index": idx + 1,
        "script": scene["script"],
        "image_prompt": scene.get("last_image_prompt", ""),
        "image_url": scene.get("last_image_url", ""),
        "reference_image_url": scene.get("image_url", ""), # 首帧作为参考图
        "message": "请审核尾帧：需确保与首帧视觉一致。可通过、重写，或修改 prompt 后继续。",
        "actions": ["approve", "rewrite", "edit_prompt"],
    }

    decision = _decide("end_frame_review", payload, state)
    action = decision.get("action", "approve")
    scenes = state["scenes"].copy()

    if decision.get("image_prompt") is not None:
        scenes[idx]["last_image_prompt"] = str(decision["image_prompt"]).strip()

    if action in APPROVE_ACTIONS:
        scenes[idx]["last_image_is_perfect"] = True
        scenes[idx]["last_image_critique"] = "无"

        # 注意：尾帧是 jimeng 生成的镜头产物，属于“关键帧”而非“参考图素材”。
        # 不得追加进 reference_images（参考图只来自用户素材库），避免关键帧被
        # 当作后续镜头的身份锚点，造成视觉雷同与素材库污染。

        updates = {
            "scenes": scenes,
        }

        return updates

    scenes[idx]["last_image_is_perfect"] = False
    scenes[idx]["last_image_critique"] = decision.get("reason") or "Human requested rewrite"
    return {"scenes": scenes}


def video_gen_node(state: MultimediaState, config: RunnableConfig | None = None):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🎞️ [镜头 {idx+1}] 动画师生成原片 ---")

    # 按 thread_id 隔离 clips 目录，避免多次运行相互覆盖导致前端“张冠李戴”
    thread_id = ""
    if config:
        thread_id = (config.get("configurable") or {}).get("thread_id", "") or ""
    # 【修复 P2-3 回归】同上：nodes/ 多一层，旧 dirname×3 会落到 <root>/src/agent。
    base_output = str(_output_dir())
    if thread_id:
        clips_dir = os.path.join(base_output, "clips", thread_id)
    else:
        clips_dir = os.path.join(base_output, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    scenes = state["scenes"].copy()

    # S1（P0-1）：锁定镜头跳过再生成——审核通过并「锁定」的镜头在重跑/续跑时
    # 直接复用既有视频，不重复消耗视频额度（关键帧复用由 image_gen_node 幂等守卫覆盖）。
    _locked_scenes = state.get("locked_scenes") or []
    if idx in _locked_scenes and str(scene.get("raw_video_url") or "").strip():
        print(f"    [LOCKED] [镜头 {idx+1}] 已锁定，复用既有视频（跳过视频生成）")
        return {"scenes": state["scenes"], "aborted": False, "abort_reason": None}

    # 【修复-一张图占位】agnes 免费档限流为「每分钟 1 次」(HTTP 429)，若多镜紧挨着提交，
    # 第二镜起会连续 429 → 退避 65s 起步、逐次加长，极易超过轮询上限而整体失败 → 降级成
    # 静态占位黑场（视觉上即"一张图重复播放"）。此处加镜头级提交间隔退避：距上次视频
    # 提交不足 AGNES_MIN_SUBMIT_GAP 秒则 sleep 补齐，从源头降低 429 连发导致的超时占位。
    _gap = float(os.getenv("AGNES_MIN_SUBMIT_GAP", "60"))
    _last = getattr(video_gen_node, "_last_submit_ts", 0.0)
    _now = time.monotonic()
    _wait = _gap - (_now - _last)
    # S1：仅当实际生效的视频后端为 agnes 时才做提交间隔退避；
    # fast/cinema 档位可能选择 zhipu/jimeng/dashscope，无需吃 agnes 的分钟级限速。
    from ..tools.video_gen import effective_primary_provider as _eff_provider
    if _last > 0 and _wait > 0 and _eff_provider(state.get("quality_tier") or "") == "agnes":
        print(f"    ⏱️ [镜头 {idx+1}] 距上次视频提交 {_gap:.0f}s 间隔退避，sleep {_wait:.1f}s 避免 agnes 429 连发")
        time.sleep(_wait)
    video_gen_node._last_submit_ts = time.monotonic()

    # 【修复】守卫：resume / 改写后运行等路径可能绕过 image_gen 与路由守卫，
    # 使空的 image_url 直接进入本节点，提交 DashScope 时报 'Field required: input.media'。
    # 此处复用"跳过该镜头"机制，不影响其他正常镜头与整体流程。
    img_url = scene.get("image_url", "")
    # 【修复-A】t2v 默认链不吃首帧图，关键帧图失败/缺失不应连累视频出片。
    # 仅当 image_url 既非空、又非 http、也非 data URL、也非本地文件时才跳过；
    # 为空则走 t2v 无图兜底。
    # 注意：部分后端可能返回 base64 data URL（data:image/...;base64,...），
    # 以及可能的本地绝对路径，二者均可作为首帧使用，不能判为"非法"而跳过。
    _img_url_str = str(img_url).strip()
    _is_data_url = _img_url_str.startswith("data:image")
    _is_local_file = _img_url_str.startswith("file://") or (
        len(_img_url_str) > 1 and _img_url_str[1] == ":" and os.path.isabs(_img_url_str)
    )
    if _img_url_str and not (_img_url_str.startswith("http") or _is_data_url or _is_local_file):
        # 【修复-B】此前该分支只清场而不递增 video_gen_failures / 不置 shot_failed，
        # 导致 decide_after_video_generation 看到空 raw_video_url + 空 shot_failed，
        # 既不算失败也不算降级占位，从而陷入静默丢镜头（最终拼接被判"无视频"跳过）。
        # 【用户硬性要求：视频生成绝不降级】首帧非法属于上游产物异常，
        # 计入失败交由路由重试（不写占位黑场，避免"一张图持续一个分镜"）。
        print(f"    ⚠️ [镜头 {idx+1}] 首帧图非法（非 http/data URL/本地文件），计入失败待重试")
        scenes[idx]["video_gen_failures"] = scenes[idx].get("video_gen_failures", 0) + 1
        scenes[idx]["video_is_perfect"] = False
        scenes[idx]["raw_video_url"] = ""
        scenes[idx]["final_video_url"] = ""
        scenes[idx]["video_critique"] = "SKIP: 首帧图非法，待重试（已禁用占位降级）"
        _f = scenes[idx]["video_gen_failures"]
        if _f >= MAX_VIDEO_GEN_FAILURES:
            return {
                "scenes": scenes,
                "aborted": True,
                "abort_reason": f"镜头 {idx+1} 首帧图非法且重试 {_f} 次仍不可用（已禁用占位降级）",
                "error_log": f"镜头 {idx+1} 首帧图非法",
                "shot_failed": True,
            }
        return {
            "scenes": scenes,
            "aborted": False,
            "abort_reason": None,
            "error_log": f"镜头 {idx+1} 首帧图非法，待重试",
        }

    # 【修复】部分后端返回 base64 data URL 作为首帧图，需解码为本地 png
    # 文件再传给视频生成（图生视频 / 本地渲染均接受本地文件路径）。
    # 这样既复用已生成的关键帧（保证首帧一致），又不破坏"首帧校验"语义。
    _first_frame_path: str = ""
    if _is_data_url:
        import base64 as _b64
        try:
            _header, _b64data = _img_url_str.split(",", 1)
            _png_bytes = _b64.b64decode(_b64data)
            _first_frame_path = os.path.join(clips_dir, f"scene_{idx:02d}_firstframe.png")
            with open(_first_frame_path, "wb") as _ff:
                _ff.write(_png_bytes)
            print(f"    [*] 首帧 data URL 解码为本地文件: {_first_frame_path} ({len(_png_bytes)}B)")
        except Exception as _e:
            print(f"    [WARN] 首帧 data URL 解码失败，改无图兜底: {_e}")
            _first_frame_path = ""
    elif _is_local_file:
        _first_frame_path = _img_url_str.replace("file://", "", 1) if _img_url_str.startswith("file://") else _img_url_str
    # 若首帧是远程 http，则仍交给底层视频生成链决定是否下载（t2v 不吃首帧，i2v 会下载）

    try:
        # 【验证用】GRAPH_DRY_RUN=1 时跳过真实即梦视频生成（零积分消耗），
        # 直接走占位黑场降级路径，仅用于验证 graph 节点链路本身不崩 KeyError('v')。
        # 不影响正常流程；环境变量未开启时完全走原逻辑。
        if _os.getenv("GRAPH_DRY_RUN") == "1":
            print(f"    🔬 [DRY-RUN] 镜头 {idx+1} 跳过真实视频生成，写入占位黑场（验证链路）")
            duration = float(scene.get("duration_seconds") or os.getenv("VIDEO_DURATION", "8"))
            placeholder = _make_placeholder_clip(
                duration=duration,
                idx=idx,
                out_path=os.path.join(clips_dir, f"scene_{idx:02d}.mp4"),
                label=f"镜头 {idx + 1}\nDRY-RUN 验证 · 已跳过",
            )
            scenes[idx]["final_video_url"] = placeholder or ""
            scenes[idx]["raw_video_url"] = placeholder or ""
            scenes[idx]["shot_failed"] = False
            scenes[idx]["video_is_perfect"] = False
            scenes[idx]["video_critique"] = "DRY_RUN_SKIP"
            return {"scenes": scenes, "aborted": False, "abort_reason": None}

        last_url = scene.get("last_image_url", "") if state.get("use_first_last_frame") else ""

        # ── 素材参考图融入视频生成（agnes 多图 keyframes / 其他后端 prompt 锚点）──
        # 【修复】排除整张 turnaround/多视图网格图（仅作离线身份参考，禁止作为帧内容注入，
        # 否则多视图网格会残留在画面里），仅保留角色肖像 + 道具/环境锚点，限制 1~2 张，
        # 遵循 Veo/Runway/Seedance「参考图作身份锚点、禁止网格入帧」规范，抑制身份漂移与构图杂乱。
        _portrait = state.get("character_portrait_url")
        _sheets = state.get("reference_sheets") or {}
        # 【修复-B】agnes 免费档弱运动：若把参考图当 keyframe 内容注入，会退化成
        # "人物定格的素材原图"直接入帧（静态无动作）。故 agnes 下仅取角色肖像作「身份文本约束」，
        # 不把任何参考图作为帧内容传给底层生成（ref 锚定由首帧图 ff 承担），彻底消除原图入帧。
        _video_provider = _os.getenv("VIDEO_PROVIDER", "agnes").strip().lower()
        # 角色命中信息（供后续 video_prompt 注入身份/一致性约束复用）
        _char_name = ""
        _char_desc = ""
        _prop_desc = ""
        if _video_provider == "agnes":
            # 【修复-角色漂移】此前此处强制 `_ref_imgs = []`，导致 agnes 只收到首尾关键帧
            # 与文本 prompt，完全没有角色身份锚点 → 视频中角色被任意重画（魔尊被画成
            # 白衣仙尊、女剑仙与仙尊服装雷同、同角色跨分镜换脸）。
            # agnes 的 keyframes 多图模式天然支持「参考图仅作身份锚点、不强制入帧」，
            # 因此这里改回：按当前分镜剧本匹配出场角色，取其单视图肖像作为身份锚点注入。
            #
            # 关键设计：命中角色时【不注入】全局 character_portrait_url。
            # 若所有分镜都锚定同一张全局肖像，会把魔尊也锚定成仙尊的样子，
            # 恰恰是「角色混淆」的成因之一。仅当本分镜未匹配到任何角色时才回退全局肖像。
            _script_text = " ".join([
                str(scene.get("script") or ""),
                str(scene.get("dialogue") or ""),
            ]).lower()
            _matched = _match_reference_elements(_sheets, _script_text)
            _char_urls = _matched.get("char_urls") or []
            _char_desc = _matched.get("char_desc") or ""
            _prop_desc = _matched.get("prop_desc") or ""
            _char_name = _matched.get("matched_char_name") or ""

            # 构造「命中角色优先」的子集，复用 _filter_reference_images 的
            # turnaround 多视图网格图过滤逻辑（避免网格原图入帧 / 静态定格回归）。
            _subset = {}
            if _char_urls:
                _subset["characters"] = {
                    (_char_name or f"char_{_i}"): {
                        "urls": [_u],
                        "description": _char_desc,
                    }
                    for _i, _u in enumerate(_char_urls[:2])
                }

            _ref_imgs, _ref_notes = _filter_reference_images(
                _subset if _subset else _sheets,
                "" if _char_urls else _portrait,
                max_refs=1,
            )
            if _char_name:
                _ref_notes.append(
                    f"[Character Identity] The character appearing in this shot is '{_char_name}'. "
                    f"Preserve this exact character identity — same face, same hairstyle, "
                    f"same costume colors and silhouette, same props and temperament. "
                    f"Do NOT replace them with any other character, and do NOT change costume between shots."
                )
            if _char_desc:
                _ref_notes.append(f"[Character Visual Anchor] {_char_desc}")
            if _prop_desc:
                _ref_notes.append(f"[Prop Visual Anchor] {_prop_desc}")
            print(
                f"    [*] Agnes 后端：按分镜注入角色参考图 {len(_ref_imgs)} 张"
                f"（命中角色: {_char_name or '无'}，已排除 turnaround 网格图）"
            )
        else:
            _ref_imgs, _ref_notes = _filter_reference_images(_sheets, _portrait, max_refs=2)
            if _portrait:
                _ref_notes.append("[Character Consistency] Keep the SAME character identity as the established source portrait — same face, hairstyle, outfit, equipment.")
        _ref_note = "\n".join(_ref_notes)
        if _ref_imgs:
            print(f"    [*] 注入 {len(_ref_imgs)} 张素材参考图到视频生成（已排除 turnaround 多视图网格）")

        # 【修复-D】强化镜头运动：把"抑制静态/增强运动"负向约束透传给底层，
        # 并在 video_prompt 末尾显式追加运镜/动作强调（agnes 弱运动模型尤其需要），
        # 避免成片出现"人物定格、画面静止、镜头重复"的弱运动观感。
        _vp = scene["video_prompt"] or ""
        # 【修复-动作跳跃】Agnes 官方 keyframes 最佳实践句式：显式声明「首帧 → 尾帧
        # 平滑过渡 + 保持角色身份与镜头稳定」。此前缺失这层过渡语义，模型自行在
        # 两端点间脑补，导致姿态瞬移（scene_00 的"挥剑→落地"跳变）。
        # 判断一律基于原始 _vp，避免追加内容影响后续关键字判定。
        _vp_lower = _vp.lower()
        if "keyframe transition" not in _vp_lower:
            _vp = (_vp + "\n[Keyframe Transition] Create a smooth cinematic transition from the first keyframe to the last keyframe, maintaining character identity, consistent camera angle, and natural continuous motion.").strip()
        if not any(k in _vp_lower for k in ("motion", "moving", "camera move", "swing", "push-in", "tracking")):
            _vp = (_vp + "\n[Camera Motion] Smooth continuous camera movement with the subject performing a clear, visible action — flowing robes, drifting hair, shifting light. Avoid any static or frozen frame.").strip()
        # 【修复-穿模】正向强调解剖正确，与 _MOTION_NEGATIVE_PROMPT 的负向约束双管齐下。
        # 约束已收敛为短句：过长的 prompt 会稀释模型对核心语义的注意力。
        if not any(k in _vp_lower for k in ("anatomy", "anatomically", "limbs")):
            _vp = (_vp + "\n[Anatomy] Anatomically correct figure — two arms, two legs, natural hands, no extra or floating limbs, physically plausible pose.").strip()
        # 【修复-过曝】发光道具/法术光效过强会把背景漂成纯白、吞掉手部与面部细节。
        if not any(k in _vp_lower for k in ("exposure", "overexposed", "glow")):
            _vp = (_vp + "\n[Exposure] Soft controlled glow only — no overexposure, no blown highlights, keep environment, fabric and facial detail visible.").strip()
        # 【修复-手部崩坏】握持武器的手必须与武器有真实接触：手指可见、自然包裹握柄。
        if not any(k in _vp_lower for k in ("grip", "gripping", "fingers wrapped")):
            _vp = (_vp + "\n[Hands & Grip] Hand grips the weapon naturally with five distinct fingers in real physical contact; the weapon never floats unheld.").strip()
        # 【修复-角色漂移】动态过程中角色身份必须与首尾关键帧一致，禁止中途换人换装。
        if _char_name and not any(k in _vp_lower for k in (_char_name.lower(), "character identity")):
            _vp = (_vp + f"\n[Identity Lock] This shot features '{_char_name}'. Same character from first to last frame — identical face, hairstyle, costume and props; never swap person or alter costume mid-shot.").strip()
            if _char_desc:
                _vp = (_vp + f"\n[Character Appearance] {_char_desc}").strip()
        raw_video_url = generate_video_from_image(
            img_url, _vp, last_url,
            first_frame_path=_first_frame_path, thread_id=thread_id,
            reference_images=_ref_imgs, reference_note=_ref_note,
            negative_prompt=_MOTION_NEGATIVE_PROMPT,
            duration=scene.get("duration_seconds") or None,
            quality_tier=str(state.get("quality_tier") or ""),
        )

        # 【修复】DashScope 返回的是临时远程 URL（有时效/防盗链），前端 <video> 直接播放会报
        # “没有找到支持的视频格式和 mime 类型”。这里在产出后立即下载到本地（按 thread 隔离的目录），
        # 并将 final_video_url/raw_video_url 改写为本地绝对路径，由静态服务器以正确 MIME 提供。
        # 注：部分后端直接返回本地 mp4 路径（非 URL），此时跳过网络下载，直接落盘复用。
        local_path = None
        try:
            local_path = os.path.join(clips_dir, f"scene_{idx:02d}.mp4")
            if os.path.isfile(raw_video_url):
                # 已是本地文件（本地渲染后端）：复制到统一 clips 目录
                shutil.copyfile(raw_video_url, local_path)
                raw_video_url = local_path
                print(f"    💾 镜头 {idx+1} 视频已本地落盘（免费后端）: {local_path}")
            elif download_video(raw_video_url, local_path):
                raw_video_url = local_path
                print(f"    💾 镜头 {idx+1} 视频已本地化: {local_path}")
            else:
                print(f"    ⚠️ 镜头 {idx+1} 视频本地化失败，回退使用远程 URL: {raw_video_url}")
        except Exception as dl_err:
            print(f"    ⚠️ 镜头 {idx+1} 视频本地化异常（回退远程 URL）: {dl_err}")

        # 仅在成功产出后递增质量重试计数（用于审片不通过时的重生循环上限）
        scenes[idx]["video_iterations"] = scenes[idx].get("video_iterations", 0) + 1
        scenes[idx]["raw_video_url"] = raw_video_url
        scenes[idx]["final_video_url"] = raw_video_url
        scenes[idx]["video_is_perfect"] = False
        # P1-3：记录本次渲染参数（外科手术时供 prompt_only/extend 冻结「其它条件、仅改一处」）。
        # P2-4：改为「合并写入」，保留 image_gen_node / end_frame_gen_node 已记录的首/尾帧参数，
        #       使 scene["render_params"] 成为一份完整配方（image + end_frame + video）。
        _rp_vid = dict(scenes[idx].get("render_params") or {})
        _rp_vid.update({
            "backend": _eff_provider(state.get("quality_tier") or ""),
            "quality_tier": state.get("quality_tier"),
            "duration": scene.get("duration_seconds"),
            "video_prompt": scene.get("video_prompt"),
            "use_first_last_frame": bool(state.get("use_first_last_frame")),
        })
        _rp_vid["video"] = {
            "backend": _eff_provider(state.get("quality_tier") or ""),
            "model": os.environ.get("AGNES_VIDEO_MODEL", ""),
            "quality_tier": state.get("quality_tier") or "",
            "duration": scene.get("duration_seconds"),
            "prompt": scene.get("video_prompt") or "",
            "use_first_last_frame": bool(state.get("use_first_last_frame")),
            "reference_images": [str(u) for u in (_ref_imgs or []) if u],
        }
        scenes[idx]["render_params"] = _rp_vid

        print(f"    🌟 镜头 {idx+1} 制作完成！URL: {raw_video_url}")
        return {"scenes": scenes, "aborted": False, "abort_reason": None}

    except FreeTierQuotaExhaustedError as e:
        # 【用户硬性要求：视频生成绝不降级】
        # agnes 的"额度耗尽"多为免费档限流/队列满（可恢复），一律按失败计入重试，
        # 严禁写占位黑场（会产生"一张图持续一个分镜"）。重试耗尽则中止并明确报错。
        msg = str(e)
        fails = scenes[idx].get("video_gen_failures", 0) + 1
        scenes[idx]["video_gen_failures"] = fails
        scenes[idx]["raw_video_url"] = ""
        scenes[idx]["final_video_url"] = ""
        scenes[idx]["video_is_perfect"] = False
        scenes[idx]["video_critique"] = "QUOTA_FAIL: " + msg
        quota_exhausted = list(state.get("quota_exhausted_scenes") or [])
        if idx not in quota_exhausted:
            quota_exhausted.append(idx)
        run_metrics._event("video_quota_exhausted", scene_index=idx, degraded=False, placeholder=False)

        if fails >= MAX_VIDEO_GEN_FAILURES:
            print(f"    ⛔ [镜头 {idx+1}] 视频生成(额度/限流)重试 {fails} 次仍失败，中止（不降级占位）: {msg}")
            return {
                "scenes": scenes,
                "aborted": True,
                "abort_reason": f"镜头 {idx+1} 视频生成重试 {fails} 次仍失败（已禁用占位降级）: {msg}",
                "quota_exhausted_scenes": quota_exhausted,
                "error_log": msg,
                "shot_failed": True,
            }

        print(f"    ⚠️ [镜头 {idx+1}] 视频生成(额度/限流)失败，准备重试 ({fails}/{MAX_VIDEO_GEN_FAILURES}): {msg}")
        return {
            "scenes": scenes,
            "aborted": False,
            "abort_reason": None,
            "quota_exhausted_scenes": quota_exhausted,
            "error_log": msg,
        }

    except Exception as e:
        # 【用户硬性要求：视频生成绝不降级】
        # 失败一律走重试（由路由 decide_after_video_generation 按次数决策），
        # 严禁写入占位黑场/静态图（那会导致"一张图持续一个分镜"的观感）。
        # 重试次数耗尽仍失败 -> 中止整条 run 并明确报错，绝不用静态图顶替真实视频。
        msg = f"视频生成失败: {str(e)}"
        fails = scenes[idx].get("video_gen_failures", 0) + 1
        scenes[idx]["video_gen_failures"] = fails
        scenes[idx]["video_is_perfect"] = False
        scenes[idx]["video_critique"] = "FAIL: " + msg
        # 关键：不写任何占位，槽位保持为空，让路由识别为"未产出"从而重试。
        scenes[idx]["raw_video_url"] = ""
        scenes[idx]["final_video_url"] = ""

        if fails >= MAX_VIDEO_GEN_FAILURES:
            print(f"    ⛔ [镜头 {idx+1}] 视频生成重试 {fails} 次仍失败，中止（不降级占位）: {msg}")
            return {
                "scenes": scenes,
                "aborted": True,
                "abort_reason": f"镜头 {idx+1} 视频生成重试 {fails} 次仍失败（已禁用占位降级）: {msg}",
                "error_log": msg,
                "shot_failed": True,
            }

        print(f"    ⚠️ [镜头 {idx+1}] 视频生成失败，准备重试 ({fails}/{MAX_VIDEO_GEN_FAILURES}): {msg}")
        return {
            "scenes": scenes,
            "aborted": False,
            "abort_reason": None,
            "error_log": msg,
        }


def video_reviewer_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🧐 [镜头 {idx+1}] 视频总监 + 安全官审核 ---")

    # 【修复】显式获取上一镜头的图片，优先取尾帧
    previous_image_url = None
    if idx > 0:
        prev_scene = state["scenes"][idx - 1]
        previous_image_url = prev_scene.get("last_image_url") or prev_scene.get("image_url")
        if previous_image_url:
            print(f"    [*] 正在对比上一镜头关键帧: {previous_image_url[-30:]}...")

    feedback = evaluate_video(
        scene["raw_video_url"],
        scene["script"],
        VIDEO_REVIEWER_SYSTEM_PROMPT,
        previous_image_url=previous_image_url,
        frame_count=4
    )

    scenes = state["scenes"].copy()
    scenes[idx]["video_auto_feedback"] = feedback
    scenes[idx]["video_critique"] = feedback
    scenes[idx]["video_is_perfect"] = _is_review_pass(feedback)

    # 视频审核模型不可用（如免费额度耗尽 403）：记录全局标志，供路由跳过依赖 VLM 的无效重试
    reviewer_unavailable = feedback.startswith("REVIEWER_UNAVAILABLE:")
    if reviewer_unavailable:
        print("    ⚠️ [视频审核模型不可用] 后续镜头的视频审片重试将被跳过（仅作占位保底，不空耗额度）")

    return {
        "scenes": scenes,
        "reviewer_unavailable": reviewer_unavailable or bool(state.get("reviewer_unavailable")),
    }


def video_review_gate_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    # 【修复】显式获取上一镜头的图片
    previous_image_url = None
    if idx > 0:
        prev_scene = state["scenes"][idx - 1]
        previous_image_url = prev_scene.get("last_image_url") or prev_scene.get("image_url")

    # ── 动作幅度自检：防止"静止空镜 / 看不出在动" ──
    action_beat = scene.get("action_beat", "")
    if len(action_beat.strip()) < 8:
        motion_hint = "⚠️ 动作空洞：本镜头缺少明确的可见动作（action_beat 过短），建议重写时补充主体从起始到结束的具体动作，否则视频模型易生成静止空镜。"
    else:
        motion_hint = f"✓ 已锚定可见动作：{action_beat[:60]}..."

    payload = {
        "stage": "video_review",
        "title": f"镜头 {idx + 1} 视频审片",
        "scene_index": idx + 1,
        "script": scene["script"],
        "global_setting": state.get("global_setting", ""),
        "video_prompt": scene.get("video_prompt", ""),
        "video_url": scene.get("raw_video_url", ""),
        "reference_image_url": previous_image_url,
        "embedding_similarity": scene.get("embedding_similarity"),
        "auto_feedback": scene.get("video_auto_feedback", scene.get("video_critique", "")),
        "action_beat": action_beat,
        "motion_hint": motion_hint,
        "message": "请审核视频成片：可通过、重写，或修改 video_prompt 后再生成。注意检查主体是否有可见动作、画面是否看得出在演什么。",
        "actions": ["approve", "rewrite", "edit_prompt"],
        # P1-3：外科手术可选修正类型（改词/保首帧/重绘/延长）
        "fix_types": ["prompt_only", "keep_first_frame", "reimage", "extend"],
    }

    decision = _decide("video_review", payload, state)
    action = decision.get("action", "approve")

    scenes = state["scenes"].copy()

    if decision.get("video_prompt") is not None:
        scenes[idx]["video_prompt"] = str(decision["video_prompt"]).strip()

    # P1-3：记录外科手术修正类型；默认 keep_first_frame（保首帧重滚视频，等价旧行为）。
    _fix = _normalize_fix_type(decision.get("fix_type")) or FIX_KEEP_FIRST_FRAME
    scenes[idx]["fix_type"] = _fix
    if _fix == FIX_EXTEND:
        _cur = float(scene.get("duration_seconds") or 8)
        _new = min(12.0, round(_cur * 1.5, 1))
        scenes[idx]["duration_seconds"] = _new
        scenes[idx]["extend_applied"] = True
        print(f"    [P1-3] 镜头 {idx+1} 修正类型=extend（尾帧续接延长：{_cur}s → {_new}s）")
    else:
        print(f"    [P1-3] 镜头 {idx+1} 修正类型={_fix}")

    if action in APPROVE_ACTIONS:
        scenes[idx]["video_is_perfect"] = True
        scenes[idx]["video_critique"] = "无"
        scenes[idx]["final_video_url"] = scene["raw_video_url"]
        _ret: Dict[str, Any] = {"scenes": scenes}
        # S1（P0-1）：审核通过时可勾选「锁定此镜头」——重跑/续跑不再重新生成该镜头
        if decision.get("lock"):
            _locked = list(state.get("locked_scenes") or [])
            if idx not in _locked:
                _locked.append(idx)
            _ret["locked_scenes"] = _locked
            print(f"    [LOCKED] [镜头 {idx+1}] 已锁定，后续重跑将复用该视频")
        return _ret

    scenes[idx]["video_is_perfect"] = False
    scenes[idx]["video_critique"] = decision.get("reason") or scene.get("video_auto_feedback") or "Human requested rewrite"
    return {"scenes": scenes}
