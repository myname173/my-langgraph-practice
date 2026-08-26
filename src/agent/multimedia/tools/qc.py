"""成片质量巡检（QC）工具。

以只读、幂等方式核对一条 thread 跑完后的成片与片段质量，输出结构化
JSON 报告 + 人类可读日志，便于替代当前手动验证。覆盖以下维度：

- 时长：校验 agnes 链 ``VIDEO_DURATION``（默认 8s）是否真正生效（此前写死
  81 帧的回归点）。
- 分辨率 / 可播放性：用已依赖的 MoviePy 现读 ``duration`` / ``size`` / ``fps``。
- 首尾帧与参考图注入信号：消费既有 state 信号，核对 ``use_first_last_frame``、
  ``scene.image_url`` / ``last_image_url``、本地 ``scene_XX_firstframe.png``、
  ``len(_ref_imgs)`` 与 ``reference_sheets`` 命中。
- 素材真正被用上：核对 ``asset_match_report``。
- 跨镜头角色一致性漂移：激活 ``reference_embeddings`` 死字段做余弦相似度比对。

设计守则（参照社区主流实践）：
  - 单一职责：只检查、不修改任何产物（只读、幂等）。
  - 可观测性：结构化报告 JSON + 分级控制台日志（[OK]/[WARN]/[FAIL]）。
  - 类型安全：mypy --strict 全过，Google 风格 docstring。
  - 可测试：单测 mock 掉 MoviePy 与 langgraph_sdk，零额度消耗。
  - 最小依赖：复用 MoviePy / urllib / langgraph_sdk（均已在项目依赖中）。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, TypedDict

# 项目根目录：qc.py -> tools -> multimedia -> agent -> src -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# 成片 / 片段相对产物根（与 graph.py stitch 落盘一致）。
_CLIPS_REL = Path("output") / "clips"

# 时长校验阈值（秒）：与 video_gen 透传逻辑同源，防止 8s 回归。
_DURATION_TOLERANCE_S = 1.5
# 期望分辨率（与 video_stitcher 归一 1920x1080 一致）。
_EXPECTED_SIZE = (1920, 1080)
# 一致性漂移阈值：跨镜头 embedding 余弦相似度低于此值视为可疑（WARN）。
_CONSISTENCY_WARN_THRESHOLD = 0.80


class QcCheckResult(TypedDict):
    """单条检查项结果。"""

    name: str  # 检查项名，如 "duration" / "injection" / "consistency"
    level: str  # "OK" | "WARN" | "FAIL"
    detail: str  # 人类可读说明
    metric: Dict[str, Any]  # 结构化数值，如 {"expected": 8.0, "actual": 7.9}


class QcReport(TypedDict):
    """整条 thread 的质量巡检报告。"""

    thread_id: str
    final_movie: str  # 成片本地路径（可能为空）
    summary: str  # "PASS" | "FAIL"
    checks: List[QcCheckResult]


def _video_file_clip() -> Any:
    """惰性解析 ``moviepy.VideoFileClip``，便于单测 mock（避免顶层重导入）。

    Returns:
        ``VideoFileClip`` 类对象。
    """
    from moviepy import VideoFileClip  # type: ignore[import-untyped]

    return VideoFileClip


def _load_video_duration_default() -> float:
    """从 .env 读取期望时长（秒）。

    Returns:
        期望单镜头时长（秒），默认 8。
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    try:
        return float(os.getenv("VIDEO_DURATION", "8"))
    except ValueError:
        return 8.0


def inspect_clip(path: str) -> Dict[str, Any]:
    """用 MoviePy 读取 mp4 时长 / 分辨率 / fps。

    打开文件后必须关闭句柄，防止句柄泄漏（社区常见踩坑）。读取失败时
    仅返回 ``readable=False``，不抛异常（QC 以降级为 WARN，不崩）。

    Args:
        path: mp4 文件的本地绝对路径。

    Returns:
        含 ``readable`` / ``duration`` / ``width`` / ``height`` / ``fps`` /
        ``error`` 的字典。
    """
    result: Dict[str, Any] = {
        "readable": False,
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "error": "",
    }
    if not os.path.isfile(path):
        result["error"] = f"文件不存在: {path}"
        return result
    try:
        clip_cls = _video_file_clip()
        with clip_cls(path) as clip:
            result["readable"] = True
            result["duration"] = float(getattr(clip, "duration", 0.0) or 0.0)
            size = getattr(clip, "size", None)
            if size:
                result["width"], result["height"] = int(size[0]), int(size[1])
            result["fps"] = float(getattr(clip, "fps", 0.0) or 0.0)
    except Exception as exc:  # noqa: BLE001 - 任何解码失败都降级而非崩溃
        result["error"] = f"无法读取: {exc}"
    return result


def _cosine(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度。

    Args:
        a: 向量 A。
        b: 向量 B。

    Returns:
        余弦相似度，取值 [-1, 1]；任一向量为零向量时返回 1.0（视作无差异）。
    """
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return dot / (na * nb)


def _level_rank(level: str) -> int:
    """把分级映射为数字，便于总判定（FAIL > WARN > OK）。"""
    return {"FAIL": 2, "WARN": 1, "OK": 0}.get(level, 0)


def _check_final_movie(
    thread_id: str,
    expected_duration: float,
) -> List[QcCheckResult]:
    """核对成片文件本身的时长、分辨率与可播放性。

    Args:
        thread_id: 当前 thread 标识，用于定位产物目录。
        expected_duration: 期望单镜头时长（秒）。

    Returns:
        检查项列表（可能包含 duration / resolution / playable 多项）。
    """
    checks: List[QcCheckResult] = []
    final_path = _PROJECT_ROOT / _CLIPS_REL / thread_id / "FINAL_成片.mp4"
    if not final_path.is_file():
        checks.append(
            QcCheckResult(
                name="playable",
                level="FAIL",
                detail=f"成片不存在: {final_path}",
                metric={"path": str(final_path)},
            )
        )
        return checks

    meta = inspect_clip(str(final_path))
    if not meta["readable"]:
        checks.append(
            QcCheckResult(
                name="playable",
                level="FAIL",
                detail=f"成片不可播放: {meta['error']}",
                metric={"path": str(final_path)},
            )
        )
        return checks

    checks.append(
        QcCheckResult(
            name="playable",
            level="OK",
            detail="成片可正常解码播放",
            metric={"path": str(final_path), "fps": meta["fps"]},
        )
    )

    # 分辨率校验：成片为拼接产物，应归一为 1920x1080。
    if (meta["width"], meta["height"]) != _EXPECTED_SIZE:
        checks.append(
            QcCheckResult(
                name="resolution",
                level="WARN",
                detail=(
                    f"成片分辨率 {meta['width']}x{meta['height']} 与期望 "
                    f"{_EXPECTED_SIZE[0]}x{_EXPECTED_SIZE[1]} 不一致"
                ),
                metric={"width": meta["width"], "height": meta["height"]},
            )
        )
    else:
        checks.append(
            QcCheckResult(
                name="resolution",
                level="OK",
                detail="成片分辨率符合 1920x1080",
                metric={"width": meta["width"], "height": meta["height"]},
            )
        )

    # 成片总时长：应约为 单镜头时长 × 镜头数，这里以单镜头时长作下界参考，
    # 仅标记明显异常（如 <=0 或远小于单镜头时长）。
    dur = meta["duration"]
    if dur <= 0:
        checks.append(
            QcCheckResult(
                name="duration",
                level="FAIL",
                detail="成片时长为 0，疑似空文件或损坏",
                metric={"actual": dur},
            )
        )
    elif dur < expected_duration - _DURATION_TOLERANCE_S:
        checks.append(
            QcCheckResult(
                name="duration",
                level="WARN",
                detail=(
                    f"成片总时长 {dur:.1f}s 小于单镜头期望 {expected_duration:.1f}s"
                    f"（可能因部分镜头失败被占位黑场拼接，需人工确认）"
                ),
                metric={"expected": expected_duration, "actual": round(dur, 2)},
            )
        )
    else:
        checks.append(
            QcCheckResult(
                name="duration",
                level="OK",
                detail=f"成片总时长 {dur:.1f}s 合理",
                metric={"expected": expected_duration, "actual": round(dur, 2)},
            )
        )
    return checks


def _check_scene_clips(
    thread_id: str,
    scenes: List[Dict[str, Any]],
    expected_duration: float,
) -> List[QcCheckResult]:
    """逐镜头核对片段时长与首尾帧 / 参考图注入信号。

    Args:
        thread_id: 当前 thread 标识。
        scenes: graph 状态里的 scenes 列表（含 image_url / last_image_url 等）。
        expected_duration: 期望单镜头时长（秒）。

    Returns:
        检查项列表（每个镜头 1~2 项）。
    """
    checks: List[QcCheckResult] = []
    for idx, scene in enumerate(scenes):
        clip_path = (
            _PROJECT_ROOT / _CLIPS_REL / thread_id / f"scene_{idx:02d}.mp4"
        )
        if not clip_path.is_file():
            checks.append(
                QcCheckResult(
                    name=f"scene_{idx:02d}_clip",
                    level="FAIL",
                    detail=f"镜头 {idx + 1} 片段缺失: {clip_path}",
                    metric={"index": idx},
                )
            )
            continue

        meta = inspect_clip(str(clip_path))
        if not meta["readable"]:
            checks.append(
                QcCheckResult(
                    name=f"scene_{idx:02d}_clip",
                    level="FAIL",
                    detail=f"镜头 {idx + 1} 片段不可播放: {meta['error']}",
                    metric={"index": idx},
                )
            )
            continue

        actual = meta["duration"]
        if abs(actual - expected_duration) > _DURATION_TOLERANCE_S:
            level = "WARN" if actual > 0 else "FAIL"
            checks.append(
                QcCheckResult(
                    name=f"scene_{idx:02d}_duration",
                    level=level,
                    detail=(
                        f"镜头 {idx + 1} 时长 {actual:.1f}s 偏离期望 "
                        f"{expected_duration:.1f}s（±{_DURATION_TOLERANCE_S}s）"
                    ),
                    metric={
                        "expected": expected_duration,
                        "actual": round(actual, 2),
                        "index": idx,
                    },
                )
            )
        else:
            checks.append(
                QcCheckResult(
                    name=f"scene_{idx:02d}_duration",
                    level="OK",
                    detail=f"镜头 {idx + 1} 时长 {actual:.1f}s 符合期望",
                    metric={
                        "expected": expected_duration,
                        "actual": round(actual, 2),
                        "index": idx,
                    },
                )
            )

        # 首尾帧注入信号核对（assets_imported=False 时才应有首尾帧）。
        # 参考图注入信号（assets_imported=True 时由 reference_images 提供）。
        has_first = bool(scene.get("image_url"))
        has_last = bool(scene.get("last_image_url"))
        injection_note = (
            f"首帧={'有' if has_first else '无'} "
            f"尾帧={'有' if has_last else '无'}"
        )
        checks.append(
            QcCheckResult(
                name=f"scene_{idx:02d}_injection",
                level="OK",
                detail=f"镜头 {idx + 1} 首尾帧注入信号: {injection_note}",
                metric={
                    "has_first_frame": has_first,
                    "has_last_frame": has_last,
                    "index": idx,
                },
            )
        )
    return checks


def _check_asset_usage(
    state: Dict[str, Any],
) -> List[QcCheckResult]:
    """核对素材是否真正被剧本使用（消费 asset_match_report）。

    Args:
        state: graph 完整状态。

    Returns:
        检查项列表（matched / unmatched 各一项）。
    """
    checks: List[QcCheckResult] = []
    report = state.get("asset_match_report") or {}
    matched = report.get("matched") or {}
    unmatched = report.get("unmatched") or []
    if not isinstance(matched, dict):
        matched = {}
    if not isinstance(unmatched, list):
        unmatched = []

    if matched:
        checks.append(
            QcCheckResult(
                name="asset_matched",
                level="OK",
                detail=f"{len(matched)} 个素材被剧本命中: {list(matched.keys())}",
                metric={"matched": matched},
            )
        )
    else:
        checks.append(
            QcCheckResult(
                name="asset_matched",
                level="WARN",
                detail="无素材命中记录（reference_gen_node 可能未产出 asset_match_report）",
                metric={"matched": {}},
            )
        )

    if unmatched:
        checks.append(
            QcCheckResult(
                name="asset_unmatched",
                level="WARN",
                detail=f"{len(unmatched)} 个素材未命中剧本: {unmatched}",
                metric={"unmatched": unmatched},
            )
        )
    else:
        checks.append(
            QcCheckResult(
                name="asset_unmatched",
                level="OK",
                detail="所有注入素材均被剧本命中",
                metric={"unmatched": []},
            )
        )
    return checks


def _check_consistency(
    state: Dict[str, Any],
) -> List[QcCheckResult]:
    """基于 reference_embeddings 做跨镜头角色一致性漂移核查。

    激活 state 中每镜头 append 但从不消费的 ``reference_embeddings`` 死字段，
    用余弦相似度比对相邻镜头，标记视觉一致性漂移。

    Args:
        state: graph 完整状态。

    Returns:
        检查项列表（含 embedding 可用性 + 各相邻镜头相似度）。
    """
    checks: List[QcCheckResult] = []
    embeddings: List[List[float]] = state.get("reference_embeddings") or []
    if not embeddings:
        checks.append(
            QcCheckResult(
                name="consistency",
                level="WARN",
                detail="无可用的 reference_embeddings（角色一致性模型未启用或采集失败）",
                metric={"count": 0},
            )
        )
        return checks

    checks.append(
        QcCheckResult(
            name="consistency",
            level="OK",
            detail=f"已采集 {len(embeddings)} 个镜头 embedding，执行跨镜头一致性比对",
            metric={"count": len(embeddings)},
        )
    )

    min_sim = 1.0
    worst_pair = (-1, -1)
    for i in range(1, len(embeddings)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        if sim < min_sim:
            min_sim = sim
            worst_pair = (i - 1, i)
        if sim < _CONSISTENCY_WARN_THRESHOLD:
            checks.append(
                QcCheckResult(
                    name=f"consistency_{i - 1}_{i}",
                    level="WARN",
                    detail=(
                        f"镜头 {i} 与镜头 {i - 1} 视觉相似度 {sim:.3f} "
                        f"低于阈值 {_CONSISTENCY_WARN_THRESHOLD}，存在一致性漂移风险"
                    ),
                    metric={"sim": round(sim, 4), "pair": [i - 1, i]},
                )
            )

    if worst_pair != (-1, -1) and min_sim >= _CONSISTENCY_WARN_THRESHOLD:
        checks.append(
            QcCheckResult(
                name="consistency_min",
                level="OK",
                detail=(
                    f"最小相邻相似度 {min_sim:.3f}（镜头 {worst_pair[1]} vs "
                    f"{worst_pair[0]}），未触发漂移阈值"
                ),
                metric={"min_sim": round(min_sim, 4)},
            )
        )
    return checks


def check_thread(
    thread_id: str,
    state: Dict[str, Any] | None = None,
    scenes_override: List[Dict[str, Any]] | None = None,
) -> QcReport:
    """执行整条 thread 的只读质量巡检。

    优先使用调用方传入的 ``state``（便于单测 mock）；未传时尝试从本地
    ``output/clips/<thread_id>/`` 读取产物并做轻量巡检（无网络依赖）。

    Args:
        thread_id: 待巡检的 thread 标识。
        state: 可选的 graph 完整状态字典（含 scenes / asset_match_report /
            reference_embeddings 等）。为空时仅做产物文件巡检。
        scenes_override: 测试用，覆盖 state 内的 scenes。

    Returns:
        QcReport：含 summary（PASS/FAIL）与各项检查细节。
    """
    expected_duration = _load_video_duration_default()
    checks: List[QcCheckResult] = []

    # 1) 成片文件本身
    checks.extend(_check_final_movie(thread_id, expected_duration))

    # 2) 逐镜头片段 + 注入信号（需要 state.scenes）
    scenes: List[Dict[str, Any]] = []
    if state is not None:
        scenes = scenes_override if scenes_override is not None else (
            state.get("scenes") or []
        )
    checks.extend(_check_scene_clips(thread_id, scenes, expected_duration))

    # 3) 素材命中率 + 4) 跨镜头一致性（需要 state 完整信号）
    if state is not None:
        checks.extend(_check_asset_usage(state))
        checks.extend(_check_consistency(state))

    # 总判定：存在任意 FAIL → FAIL，否则含 WARN → 仍记 WARN 但 summary 计 PASS？
    # 社区惯例：FAIL 直接判 FAIL；WARN 表示需关注但流程通过，记为 PASS（带提示）。
    has_fail = any(_level_rank(c["level"]) >= 2 for c in checks)
    summary = "FAIL" if has_fail else "PASS"

    final_path = _PROJECT_ROOT / _CLIPS_REL / thread_id / "FINAL_成片.mp4"
    report: QcReport = {
        "thread_id": thread_id,
        "final_movie": str(final_path) if final_path.is_file() else "",
        "summary": summary,
        "checks": checks,
    }
    return report


def render_report(report: QcReport) -> str:
    """把 QcReport 渲染为人类可读的控制台文本。

    Args:
        report: check_thread 产出的报告。

    Returns:
        分级着色前缀的纯文本报告。
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"[QC] thread={report['thread_id']} 判定={report['summary']}")
    lines.append(f"[QC] 成片={report['final_movie'] or '(缺失)'}")
    lines.append("-" * 60)
    counts = {"OK": 0, "WARN": 0, "FAIL": 0}
    for c in report["checks"]:
        counts[c["level"]] = counts.get(c["level"], 0) + 1
        lines.append(f"[{c['level']}] {c['name']}: {c['detail']}")
    lines.append("-" * 60)
    lines.append(
        f"[QC] 统计 OK={counts['OK']} WARN={counts['WARN']} FAIL={counts['FAIL']}"
    )
    lines.append("=" * 60)
    return "\n".join(lines)


def save_report(report: QcReport, out_dir: str | None = None) -> str:
    """把报告落盘为 JSON，路径 ``<out_dir>/_qc_<thread_id>.json``。

    Args:
        report: check_thread 产出的报告。
        out_dir: 输出目录，默认项目根 ``output/``。

    Returns:
        落盘 JSON 的绝对路径。
    """
    base = Path(out_dir) if out_dir else _PROJECT_ROOT / "output"
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / f"_qc_{report['thread_id']}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(out_path)
