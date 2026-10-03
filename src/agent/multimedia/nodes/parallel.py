# src/agent/multimedia/nodes/parallel.py
"""并行 / 链式扇出机制：Send fan-out、镜头子图、FLF 链式衔接。

P2-3：由原 graph.py（单体）按 stage 机械拆分而来，逻辑逐字节保留；
仅新增本文件头注释与模块导入，函数体 / 常量 / 注释均未改动。
"""
import inspect
import os
from typing import Any, Dict, List, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from ..state import MultimediaState, ShotState
from .director import director_node, director_refine_node, end_frame_director_node, end_frame_prompt_preview_node, prompt_preview_node, videographer_node
from .render import end_frame_gen_node, end_frame_review_gate_node, image_gen_node, image_review_gate_node, reviewer_node, video_gen_node, video_review_gate_node, video_reviewer_node
from .routing import decide_after_critic, decide_after_end_frame_generation, decide_after_image_generation, decide_after_video_generation, decide_end_frame_quality, decide_image_quality, decide_video_quality


# ================= 组装 Graph =================


# ═════════════════════════════════════════════════════════════════════════
# P0-2（架构评审）：镜头级并行化 — LangGraph Send fan-out map-reduce
#
# 设计（对齐评审方案）：
#   1. cinematic_critic 通过后，若满足并行条件（auto_mode + MULTIMEDIA_PARALLEL_SHOTS
#      未关闭 + 镜头数 ≥ 2），用 Send 按镜头扇出 N 个 shot_chain 子图分支；
#   2. 每个分支持有【私有 ShotState】（scene 单镜头工作副本 + ctx 只读共享快照），
#      分支内【原样复用】现有节点函数（director → image → end_frame → video → review），
#      经 _shot_rehydrate 获得 MultimediaState 视图，写回只取自己的镜头槽位；
#   3. 分支末尾 _shot_collect 折叠为 {shot_index: 结果}，经主图
#      scene_results / branch_results 通道（merge_dicts_shallow reducer）无锁归并；
#   4. shots_collected 汇合节点把结果折叠回 scenes 并聚合 aborted/error 等全局旗标，
#      之后统一进 stitcher（评审所述「video_review 汇合后统一过 critic」的等价落点：
#      critic 已在扇出前对全片 shot_plan 通过，分支内 VLM 审片照常逐镜执行）。
#
# 已知取舍（评审已接受）：
#   - 并行分支看不到前序镜头的 scene_anchors（角色 DNA / 跨镜承接指令），
#     一致性由 reference_sheets / character_portrait 锚点兜底；串行模式不受影响；
#   - video_reviewer 的「与上一镜关键帧对比」在并行模式下退化为无对比（prev 未生成）；
#   - 并行要求 auto_mode：6 道人工闸口在分支内自动放行（人审工作流仍走串行路径）；
#   - agnes 分钟级限速由「多后端候选 + AGNES_MIN_SUBMIT_GAP」共同错峰，
#     并发下 gap 校验存在竞态（方向性限速），429 由既有退避重试兜底。
# ═════════════════════════════════════════════════════════════════════════

_PARALLEL_SHOTS_ENV = "MULTIMEDIA_PARALLEL_SHOTS"

# 扇出时注入分支 ctx 的只读共享上下文（MultimediaState 键的子集）
_SHOT_CTX_KEYS = (
    "task", "global_setting", "visual_style", "style_description", "style_suffix",
    "content_type", "visual_context", "shot_plan", "sequence_graph", "critic_eval",
    "quality_gates", "rag_context", "narrative_arc", "reference_sheets",
    "reference_images", "reference_embeddings", "character_portrait_url",
    "assets_imported", "thread_id", "quality_tier", "asset_match_report",
    "scene_anchors",
)

# 节点返回值中需要映射回分支私有通道的标量键
_SHOT_BRANCH_SCALARS = (
    "aborted", "abort_reason", "shot_failed", "error_log",
    "reviewer_unavailable", "rewrite_count", "use_first_last_frame",
    "asset_match_report",
)


def _parallel_shots_enabled(state: MultimediaState) -> bool:
    """并行扇出开关：默认开启，但必须 auto_mode（人审工作流恒走串行路径）。"""
    if os.getenv(_PARALLEL_SHOTS_ENV, "1") == "0":
        return False
    if not (state.get("auto_mode") or os.getenv("AUTO_MODE") == "1"):
        return False
    return len(state.get("scenes") or []) >= 2


def _shot_ctx_snapshot(state: MultimediaState) -> Dict[str, Any]:
    """构建分支共享只读上下文（含全片 scenes 快照，供跨镜只读与槽位还原）。"""
    ctx: Dict[str, Any] = {}
    for k in _SHOT_CTX_KEYS:
        v = state.get(k)
        if v is not None:
            ctx[k] = v
    ctx["scenes_snapshot"] = [dict(sc) if isinstance(sc, dict) else {} for sc in (state.get("scenes") or [])]
    return ctx


def _shot_rehydrate(branch: Dict[str, Any]) -> Dict[str, Any]:
    """ShotState → MultimediaState 视图：现有节点函数在分支内零改动复用的关键。

    - scenes = 快照 + 本分支 scene 覆盖自己的槽位（其余镜头仅供只读，并行期
      可能尚未生成图像 → video_reviewer 的 prev 对比自然退化为 None）；
    - current_scene_index 恒为分支的 shot_index；
    - auto_mode 强制 True：分支内人工闸口自动放行（并行模式前提）。
    """
    ctx = branch.get("ctx") or {}
    i = int(branch.get("shot_index", 0))
    scenes = [dict(sc) for sc in (ctx.get("scenes_snapshot") or [])]
    while len(scenes) <= i:
        scenes.append({})
    scenes[i] = dict(branch.get("scene") or {})
    full = dict(ctx)
    full.pop("scenes_snapshot", None)
    full.update({
        "scenes": scenes,
        "current_scene_index": i,
        "auto_mode": True,
        "aborted": bool(branch.get("aborted")),
        "abort_reason": branch.get("abort_reason"),
        "shot_failed": bool(branch.get("shot_failed")),
        "error_log": branch.get("error_log"),
        "reviewer_unavailable": branch.get("reviewer_unavailable"),
        "rewrite_count": int(branch.get("rewrite_count") or 0),
        "use_first_last_frame": bool(branch.get("use_first_last_frame")),
    })
    return full


def _shot_node(fn):
    """把现有节点函数包装成分支节点：读走还原视图，写回只取自己的槽位。

    按 fn 签名决定是否透传 config（如 video_gen_node 需要 configurable.thread_id
    做 clips 目录隔离；其余节点仅接受 state）。
    """
    _accepts_config = len(inspect.signature(fn).parameters) >= 2

    def run(branch: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
        view = _shot_rehydrate(branch)
        result = (fn(view, config) if _accepts_config else fn(view)) or {}
        out: Dict[str, Any] = {}
        if "scenes" in result:
            i = int(branch.get("shot_index", 0))
            lst = result.get("scenes") or []
            if i < len(lst) and isinstance(lst[i], dict):
                out["scene"] = dict(lst[i])
        for k in _SHOT_BRANCH_SCALARS:
            if k in result:
                out[k] = result[k]
        return out
    return run


def _shot_router(fn):
    """把现有路由函数包装成分支路由（路由只读，不写 state）。"""
    def run(branch: Dict[str, Any]):
        return fn(_shot_rehydrate(branch))
    return run


def _decide_after_prompt_preview_shot(state: MultimediaState):
    """（分支版）prompt_preview 路由：与主图 inline 版本语义一致。"""
    scene = state["scenes"][state["current_scene_index"]]
    return "image_generator" if scene.get("is_perfect") else "director"


def _decide_after_end_frame_prompt_preview_shot(state: MultimediaState):
    """（分支版）end_frame_prompt_preview 路由：与主图 inline 版本语义一致。"""
    scene = state["scenes"][state["current_scene_index"]]
    return "end_frame_generator" if scene.get("last_image_is_perfect") else "end_frame_director"


def _diff_asset_reports(base: Optional[Dict], report: Optional[Dict]) -> Dict[str, Any]:
    """计算分支相对扇出时基线的资产命中增量（join 时累加回主图报告）。"""
    base = base or {}
    rep = report or {}
    base_matched = base.get("matched") or {}
    delta_matched: Dict[str, int] = {}
    for k, v in (rep.get("matched") or {}).items():
        inc = int(v) - int(base_matched.get(k, 0))
        if inc > 0:
            delta_matched[k] = inc
    base_unmatched = set(base.get("unmatched") or [])
    removed = [m for m in base_unmatched if m not in set(rep.get("unmatched") or [])]
    if delta_matched or removed:
        return {"matched": delta_matched, "unmatched_removed": removed}
    return {}


def _shot_collect(branch: Dict[str, Any]) -> Dict[str, Any]:
    """分支末节点：折叠私有状态为主图可安全归并的窄载荷。"""
    i = int(branch.get("shot_index", 0))
    delta = _diff_asset_reports(
        (branch.get("ctx") or {}).get("asset_match_report"),
        branch.get("asset_match_report"),
    )
    return {
        "scene_results": {str(i): dict(branch.get("scene") or {})},
        "branch_results": {str(i): {
            "aborted": bool(branch.get("aborted")),
            "abort_reason": branch.get("abort_reason"),
            "shot_failed": bool(branch.get("shot_failed")),
            "error_log": branch.get("error_log"),
            "reviewer_unavailable": bool(branch.get("reviewer_unavailable")),
            "quota_exhausted_scenes": list(branch.get("quota_exhausted") or []),
            "use_first_last_frame": bool(branch.get("use_first_last_frame")),
            "asset_match_delta": delta,
        }},
    }


def shots_collected_node(state: MultimediaState):
    """汇合节点：把各分支的镜头结果折叠回 scenes，聚合全局旗标。"""
    results = state.get("scene_results") or {}
    flags = state.get("branch_results") or {}
    scenes = [dict(sc) for sc in (state.get("scenes") or [])]

    applied = 0
    for k, sc in results.items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(scenes) and isinstance(sc, dict):
            scenes[i] = {**scenes[i], **sc}
            applied += 1

    aborted = any(bool(f.get("aborted")) for f in flags.values())
    shot_failed = any(bool(f.get("shot_failed")) for f in flags.values())
    reviewer_unavailable = any(bool(f.get("reviewer_unavailable")) for f in flags.values())
    quota = sorted({int(x) for f in flags.values() for x in (f.get("quota_exhausted_scenes") or [])})
    _ordered = [flags[k] for k in sorted(flags, key=lambda x: int(x) if str(x).isdigit() else 10**9)]
    reason = next((f.get("abort_reason") for f in _ordered if f.get("abort_reason")), None)
    errors = [str(f.get("error_log")) for f in _ordered if f.get("error_log")]

    # 资产命中报告：基线 + Σ分支增量
    report = dict(state.get("asset_match_report") or {})
    report.setdefault("matched", {})
    report.setdefault("unmatched", [])
    for f in flags.values():
        delta = f.get("asset_match_delta") or {}
        for m, inc in (delta.get("matched") or {}).items():
            report["matched"][m] = int(report["matched"].get(m, 0)) + int(inc)
        for m in (delta.get("unmatched_removed") or []):
            if m in report["unmatched"]:
                report["unmatched"].remove(m)

    done = sum(1 for sc in scenes if str(sc.get("final_video_url") or "").strip())
    print(f"\n--- [汇合] {applied}/{len(flags)} 个并行分支返回，"
          f"成片槽位 {done}/{len(scenes)}，aborted={aborted} ---")

    out: Dict[str, Any] = {
        "scenes": scenes,
        "aborted": aborted,
        "abort_reason": reason,
        "shot_failed": shot_failed,
        "reviewer_unavailable": reviewer_unavailable,
        "quota_exhausted_scenes": quota,
        "asset_match_report": report,
    }
    if errors:
        out["error_log"] = " | ".join(errors)
    return out


def _decide_after_parallel_join(state: MultimediaState):
    """汇合后路由：任一分支中止 → abort；否则统一进剪辑。"""
    if state.get("aborted"):
        return "abort"
    return "stitcher"


def _build_shot_chain_graph() -> "StateGraph":
    """单镜头子图：镜像串行链的节点与路由（含重试环），终点的 skip/abort/advance
    一律落到 branch_collect 折叠返回。节点函数与主图完全同一份实现。"""
    g = StateGraph(ShotState)
    g.add_node("director", _shot_node(director_node))
    g.add_node("director_refine", _shot_node(director_refine_node))
    g.add_node("prompt_preview", _shot_node(prompt_preview_node))
    g.add_node("image_generator", _shot_node(image_gen_node))
    g.add_node("reviewer", _shot_node(reviewer_node))
    g.add_node("image_review", _shot_node(image_review_gate_node))
    g.add_node("end_frame_director", _shot_node(end_frame_director_node))
    g.add_node("end_frame_prompt_preview", _shot_node(end_frame_prompt_preview_node))
    g.add_node("end_frame_generator", _shot_node(end_frame_gen_node))
    g.add_node("end_frame_review", _shot_node(end_frame_review_gate_node))
    g.add_node("videographer", _shot_node(videographer_node))
    g.add_node("video_generator", _shot_node(video_gen_node))
    g.add_node("video_reviewer", _shot_node(video_reviewer_node))
    g.add_node("video_review", _shot_node(video_review_gate_node))
    g.add_node("branch_collect", _shot_collect)

    g.set_entry_point("director")
    g.add_edge("director", "director_refine")
    g.add_edge("director_refine", "prompt_preview")
    g.add_conditional_edges(
        "prompt_preview", _shot_router(_decide_after_prompt_preview_shot),
        {"image_generator": "image_generator", "director": "director"},
    )
    g.add_conditional_edges(
        "image_generator", _shot_router(decide_after_image_generation),
        {
            "reviewer": "reviewer",
            "video_generator": "video_generator",
            "retry": "image_generator",
            "skip": "branch_collect",
            "abort": "branch_collect",
        },
    )
    g.add_edge("reviewer", "image_review")
    g.add_conditional_edges(
        "image_review", _shot_router(decide_image_quality),
        {"end_frame_director": "end_frame_director", "videographer": "videographer", "director": "director"},
    )
    g.add_edge("end_frame_director", "end_frame_prompt_preview")
    g.add_conditional_edges(
        "end_frame_prompt_preview", _shot_router(_decide_after_end_frame_prompt_preview_shot),
        {"end_frame_generator": "end_frame_generator", "end_frame_director": "end_frame_director"},
    )
    g.add_conditional_edges(
        "end_frame_generator", _shot_router(decide_after_end_frame_generation),
        {
            "end_frame_review": "end_frame_review",
            "video_generator": "video_generator",
            "retry": "end_frame_generator",
            "skip": "branch_collect",
            "abort": "branch_collect",
        },
    )
    g.add_conditional_edges(
        "end_frame_review", _shot_router(decide_end_frame_quality),
        {"videographer": "videographer", "end_frame_director": "end_frame_director"},
    )
    g.add_edge("videographer", "video_generator")
    g.add_conditional_edges(
        "video_generator", _shot_router(decide_after_video_generation),
        {
            "video_reviewer": "video_reviewer",
            "retry": "video_generator",
            "skip": "branch_collect",
            "abort": "branch_collect",
        },
    )
    g.add_edge("video_reviewer", "video_review")
    g.add_conditional_edges(
        "video_review", _shot_router(decide_video_quality),
        {
            "videographer": "videographer",
            "video_generator": "video_generator",
            "director": "director",
            "advance_scene": "branch_collect",
        },
    )
    g.add_edge("branch_collect", END)
    return g


_compiled_shot_chain = _build_shot_chain_graph().compile()


def _run_shot_chain_node(branch_state: ShotState, config: RunnableConfig | None = None):
    """父图节点包装：手动 invoke 编译子图，白名单取回结果。

    - 分支持有完全私有的 ShotState（branch_state 即初始 state）；
    - 只把父图 schema 认识的键返回给主图，ShotState 独有通道（scene/ctx/shot_index
      等）在此处被过滤，避免过期副本流入主图（scenes 归并只走 scene_results）；
    - config（含 configurable.thread_id）透传进分支，保证 clips 落盘按 thread 隔离。
    """
    _cfg = dict(config or {})
    _cfg.setdefault("configurable", {})
    _cfg["configurable"] = dict(_cfg["configurable"])
    _cfg["configurable"].setdefault(
        "thread_id",
        str((branch_state.get("ctx") or {}).get("thread_id") or branch_state.get("shot_index", "")),
    )
    final = _compiled_shot_chain.invoke(dict(branch_state), config=_cfg)
    return {
        "scene_results": dict(final.get("scene_results") or {}),
        "branch_results": dict(final.get("branch_results") or {}),
        "aborted": bool(final.get("aborted")),
        "abort_reason": final.get("abort_reason"),
        "reviewer_unavailable": bool(final.get("reviewer_unavailable")),
        "use_first_last_frame": bool(final.get("use_first_last_frame")),
        "rewrite_count": int(final.get("rewrite_count") or 0),
        "error_log": final.get("error_log"),
    }


# ── P1-1：FLF2V 链式衔接 × P0-2 并行共存（相邻镜头成链、跨链并行）──────────
# 串行模式已由 advance_scene → scene_anchors.last_frame_url 完成「上一镜尾帧 →
# 下一镜首帧」的像素级承接；但 P0-2 并行扇出下所有分支并发、彼此看不到前序镜头
# 的尾帧，FLF 链式在并行模式下退化为「无承接」。本节点把镜头按【连续片段】切成
# 若干「链」：链内逐镜串行（前镜尾帧喂给后镜首帧，保证相邻镜头像素级连续），
# 链间 Send 并发（跨链并行）。链数由 MULTIMEDIA_FLF_CHAINS 控制：0=关闭（回退
# 逐镜并行旧行为）；正整数=指定链数；未设置=连续性优先（单链，全部镜头串行承接尾帧）。

_FLF_CHAINS_ENV = "MULTIMEDIA_FLF_CHAINS"


def _flf_chain_plan(state: MultimediaState) -> Optional[List[List[int]]]:
    """返回连续链切分（List[链]，链内为连续镜头索引）；None=不启用链模式。"""
    if not state.get("use_first_last_frame"):
        return None
    n = len(state.get("scenes") or [])
    if n < 2:
        return None
    env = os.getenv(_FLF_CHAINS_ENV, "").strip()
    if env == "0":
        return None
    if env.isdigit() and int(env) >= 1:
        k = int(env)
    else:
        # 连续性优先（默认）：单链 = 全部镜头串行承接尾帧，相邻镜头像素级连续。
        # 需要并行提速时显式设 MULTIMEDIA_FLF_CHAINS=N（N 条链，链间并行、链缝不承接）。
        k = 1
    k = max(1, min(n, k))
    if k >= n:
        return None  # 每链仅 1 镜，与逐镜并行等价，无需链模式
    base, rem = divmod(n, k)
    sizes = [base + 1] * rem + [base] * (k - rem)  # 均衡连续切分
    chains, start = [], 0
    for size in sizes:
        chains.append(list(range(start, start + size)))
        start += size
    return chains


def _chain_all_locked(chain: List[int], scenes: List[Dict[str, Any]], locked: set) -> bool:
    """整链是否全部「已锁定且已有成片」——是则无需重跑。"""
    if not chain:
        return True
    for i in chain:
        if i not in locked:
            return False
        sc = scenes[i] if 0 <= i < len(scenes) else {}
        if not str((sc or {}).get("final_video_url") or "").strip():
            return False
    return True


def _run_chain_node(chain: Dict[str, Any], config: RunnableConfig | None = None):
    """父图节点：一条链内逐镜串行（前镜尾帧 → 后镜首帧），链间由 Send 并发。

    复用 _compiled_shot_chain 子图：每镜手动 invoke 一次；上一镜产出后把其
    last_image_url 注入下一镜 ctx.scene_anchors.last_frame_url，从而让 image_gen
    的 FLF 参考与视频首尾帧在并行模式下依然成立。
    """
    ctx = chain.get("ctx") or {}
    shot_indices = [int(i) for i in (chain.get("shot_indices") or [])]
    locked = set(int(i) for i in (chain.get("locked_scenes") or []))
    snapshot = ctx.get("scenes_snapshot") or []
    use_flf = bool(chain.get("use_first_last_frame"))
    base_report = dict(chain.get("asset_match_report") or {})

    _cfg = dict(config or {})
    _cfg.setdefault("configurable", {})
    _cfg["configurable"] = dict(_cfg["configurable"])
    _cfg["configurable"].setdefault("thread_id", str(ctx.get("thread_id") or ""))

    results: Dict[str, Dict[str, Any]] = {}
    flags: Dict[str, Dict[str, Any]] = {}
    agg_aborted = False
    agg_reason = None
    agg_reviewer_unavailable = False
    agg_error = None
    prev_last_frame = chain.get("seed_last_frame")

    for pos, i in enumerate(shot_indices):
        scene_snapshot = (
            dict(snapshot[i]) if 0 <= i < len(snapshot) and isinstance(snapshot[i], dict) else {}
        )

        # 锁定且已有成片的镜头：跳过生成，仅以其快照图承接链内连续性。
        if i in locked and str(scene_snapshot.get("final_video_url") or "").strip():
            _seed = scene_snapshot.get("last_image_url") or scene_snapshot.get("image_url")
            if _seed:
                prev_last_frame = _seed
            flags[str(i)] = {
                "aborted": False, "abort_reason": None, "shot_failed": False,
                "error_log": None, "reviewer_unavailable": False,
                "quota_exhausted_scenes": [], "use_first_last_frame": use_flf,
                "asset_match_delta": {},
            }
            print(f"    [P1-1] 链内镜头 {i+1} 已锁定成片，跳过生成（仅承接尾帧）")
            continue

        ctx_i = dict(ctx)
        sa = dict(ctx.get("scene_anchors") or {})
        if prev_last_frame:
            sa["last_frame_url"] = prev_last_frame
        ctx_i["scene_anchors"] = sa

        branch = {
            "shot_index": i,
            "scene": scene_snapshot,
            "ctx": ctx_i,
            "aborted": False,
            "abort_reason": None,
            "shot_failed": False,
            "error_log": None,
            "reviewer_unavailable": None,
            "rewrite_count": 0,
            "quota_exhausted": [],
            "use_first_last_frame": use_flf,
            "asset_match_report": dict(base_report),
        }
        print(f"    [P1-1] 链内拍摄 镜头 {i+1}（{pos + 1}/{len(shot_indices)}，"
              f"承接前镜尾帧={'有' if prev_last_frame else '无'}）")
        final = _compiled_shot_chain.invoke(branch, config=_cfg)

        sr = dict(final.get("scene_results") or {})
        results.update(sr)
        flags.update(dict(final.get("branch_results") or {}))
        agg_aborted = agg_aborted or bool(final.get("aborted"))
        agg_reviewer_unavailable = agg_reviewer_unavailable or bool(final.get("reviewer_unavailable"))
        if final.get("abort_reason"):
            agg_reason = final.get("abort_reason")
        if final.get("error_log"):
            agg_error = final.get("error_log")

        out_scene = sr.get(str(i)) or {}
        nxt = out_scene.get("last_image_url") or out_scene.get("image_url")
        if nxt:
            prev_last_frame = nxt

        if agg_aborted:
            break

    return {
        "scene_results": results,
        "branch_results": flags,
        "aborted": agg_aborted,
        "abort_reason": agg_reason,
        "reviewer_unavailable": agg_reviewer_unavailable,
        "use_first_last_frame": use_flf,
        "rewrite_count": 0,
        "error_log": agg_error,
    }


def _fan_out_shots(state: MultimediaState):
    """cinematic_critic 之后的路由：并行条件满足且 critic 通过 → Send 扇出；
    否则回退既有串行路由（decide_after_critic），人工工作流零影响。"""
    base = decide_after_critic(state)
    if base != "director":
        return base
    if not _parallel_shots_enabled(state):
        return "director"

    scenes = state.get("scenes") or []
    locked = set(state.get("locked_scenes") or [])
    ctx = _shot_ctx_snapshot(state)
    report = dict(state.get("asset_match_report") or {})
    use_flf = bool(state.get("use_first_last_frame"))

    # ── P1-1：FLF 链式 × 并行共存（相邻镜头成链、跨链并行）──
    chains = _flf_chain_plan(state)
    if chains:
        chain_sends = []
        for chain in chains:
            if _chain_all_locked(chain, scenes, locked):
                continue
            chain_sends.append(Send("shot_chain_seq", {
                "shot_indices": chain,
                "ctx": ctx,
                "seed_last_frame": None,
                "locked_scenes": sorted(locked),
                "use_first_last_frame": use_flf,
                "asset_match_report": report,
            }))
        if chain_sends:
            print(f"\n?? [P1-1] FLF 链式扇出 {len(chain_sends)} 条链"
                  f"（每链镜数 {'＋'.join(str(len(c)) for c in chains)}，链内串行承接尾帧 / 链间并行）")
            return chain_sends
        return "director"

    sends = []
    for i, sc in enumerate(scenes):
        # 锁定且已有成片的镜头：直接复用，不进分支（与串行 video_gen 守卫同规则）
        if i in locked and str(sc.get("final_video_url") or "").strip():
            continue
        sends.append(Send("shot_chain", {
            "shot_index": i,
            "scene": dict(sc if isinstance(sc, dict) else {}),
            "ctx": ctx,
            "aborted": False,
            "abort_reason": None,
            "shot_failed": False,
            "error_log": None,
            "reviewer_unavailable": None,
            "rewrite_count": 0,
            "quota_exhausted": [],
            "use_first_last_frame": bool(state.get("use_first_last_frame")),
            "asset_match_report": dict(state.get("asset_match_report") or {}),
        }))
    if not sends:
        return "director"
    print(f"\n🚀 [P0-2] 并行扇出 {len(sends)}/{len(scenes)} 个镜头分支（Send fan-out）")
    return sends
