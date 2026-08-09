"""
Agent 运行指标埋点（Run Metrics / Observability）。

设计目的（指向大厂 Agent 工程关注的「可观测性」素养）：
  为多媒体生成流水线建立结构化的运行指标采集，把每次任务做成一条
  trace record，记录关键环节事件（自评、人工 gate、角色一致性重生、
  收尾/失败），落盘为 JSONL，便于离线聚合出质量与成本类指标。

特点：
  - 零外部依赖：仅用标准库，不引入任何第三方包，不影响现有流水线。
  - 可开关：通过环境变量 RUN_METRICS_ENABLED / RUN_METRICS_PATH 控制，
    默认开启，文件落在项目根 data/run_metrics.jsonl。
  - 零侵入：graph.py 仅在 6 个真实落点追加一行调用，不改任何现有逻辑。
  - 单例收集器：每个进程一条「当前 run」，showrunner 起、audio_mixer/abort 收。

聚合用法见 scripts/summarize_runs.py。
"""
import json
import os
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_DEFAULT_PATH = os.path.join(_PROJECT_ROOT, "data", "run_metrics.jsonl")

_ENABLED = os.getenv("RUN_METRICS_ENABLED", "1") != "0"
_PATH = os.getenv("RUN_METRICS_PATH", _DEFAULT_PATH)

_lock = threading.Lock()
_current: Optional[Dict[str, Any]] = None


def _now() -> float:
    return time.time()


def _flush(record: Dict[str, Any]) -> None:
    """追加写入 JSONL（线程安全）。"""
    global _PATH
    try:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        with _lock:
            with open(_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # 埋点绝不影响主流程
        print(f"    [run_metrics] 写入失败（已忽略）: {e}")


# ----------------------------------------------------------------
# 任务级生命周期
# ----------------------------------------------------------------
def start_run(task: str, scene_count: int) -> None:
    """showrunner_node 入口调用：开启一条 run record。"""
    global _current
    if not _ENABLED:
        return
    _current = {
        "run_id": uuid.uuid4().hex[:12],
        "started_at": _now(),
        "task": task,
        "scene_count": scene_count,
        "events": [],          # 环节事件列表
        "finalized": False,
        "finalized_at": None,
        "final_movie_path": "",
        "outcome": "",         # ok / abort / fail
    }


def _event(name: str, **fields: Any) -> None:
    """向当前 run 追加一个环节事件。"""
    if not _ENABLED or _current is None:
        return
    ev = {"t": round(_now() - _current["started_at"], 3), "event": name}
    ev.update(fields)
    _current["events"].append(ev)


# 供外部（graph.py 路由/节点）调用的语义化埋点
def record_critic(score: float, pass_gates: bool, rewrite_count: int,
                  forced: bool) -> None:
    _event("cinematic_critic", overall_score=round(score, 3),
           pass_gates=pass_gates, rewrite_count=rewrite_count, forced=forced)


def record_human_gate(gate: str, action: str) -> None:
    """人工审核卡（interrupt）被消费时调用。action ∈ approve/rewrite/edit_prompt。"""
    _event("human_gate", gate=gate, action=action)


def record_portrait_regen(portrait_similarity: float, threshold: float) -> None:
    _event("portrait_regen", portrait_similarity=round(portrait_similarity, 4),
           threshold=round(threshold, 4))


def record_scene_outcome(idx: int, kind: str, detail: str = "") -> None:
    _event("scene_outcome", scene_index=idx, kind=kind, detail=detail)


def finalize_run(outcome: str, final_movie_path: str = "") -> None:
    """收尾节点 / abort 节点调用：落盘当前 run。"""
    global _current
    if not _ENABLED or _current is None:
        return
    _current["finalized"] = True
    _current["finalized_at"] = _now()
    _current["outcome"] = outcome
    _current["final_movie_path"] = final_movie_path
    _current["duration_s"] = round(_current["finalized_at"] - _current["started_at"], 1)
    _flush(_current)
    _current = None
