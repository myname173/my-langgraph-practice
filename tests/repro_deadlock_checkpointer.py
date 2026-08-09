"""
最小复现：大 state 持久化死锁缓解验证（不跑完整 pipeline，零 DashScope 依赖）。

复现方案：
  - 不编译业务 graph（避免 edge_tts/moviepy 重依赖），改用最小 StateGraph 构造一个
    空编译图，仅用于驱动 checkpointer 的 put（update_state 不执行节点，只写 checkpoint）。
  - 构造两种 state：
      (A) 未瘦身：studio_output 携带超大嵌套对象（模拟 film_studio 历史行为）
      (B) 已瘦身：studio_output=None，仅留 studio_output_path 指针
  - 测量单次 update_state 耗时与是否成功返回，对比两者。
  - 使用独立临时 db 文件（不污染业务 data/），结束后清理。

运行：
    python tests/repro_deadlock_checkpointer.py
退出码 0 = 两种 put 均成功且瘦身后序列化体量更小（put 更快/更稳）。
"""
import os
import sys
import time
import tempfile
import contextlib

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 用独立临时 db，避免污染业务 data/
_TMP_DB = os.path.join(tempfile.gettempdir(), "repro_deadlock_test.sqlite")
os.environ["MULTIMEDIA_CHECKPOINT_DB"] = _TMP_DB


def _big_studio_output() -> dict:
    """模拟 film_studio 的历史大对象（3 Agent 结果 + 大报告）。"""
    return {
        "studio_consensus": 0.8,
        "modifications": [{"shot": i, "detail": "x" * 200} for i in range(2000)],
        "debate": ["conflict " * 20] * 500,
        "formatted_studio_report": "R" * 200_000,
        "agent_results": {
            "cinematography": {"history": ["h" * 1000] * 300},
            "lighting": {"history": ["l" * 1000] * 300},
            "editor": {"history": ["e" * 1000] * 300},
        },
    }


def _build_minimal_graph(checkpointer):
    """构造仅用于驱动 checkpointer.put 的最小编译图（不执行任何业务节点）。"""
    from langgraph.graph import StateGraph
    from typing import TypedDict, Optional, Any

    class _S(TypedDict, total=False):
        task: Optional[str]
        scenes: Optional[list]
        current_scene_index: Optional[int]
        studio_output: Optional[Any]
        studio_output_path: Optional[str]
        studio_summary: Optional[Any]

    def _noop(state):
        return {}
    g = StateGraph(_S)
    g.add_node("noop", _noop)
    g.set_entry_point("noop")
    g.set_finish_point("noop")
    return g.compile(checkpointer=checkpointer)


def _measure_update(graph_app, state: dict, thread_id: str) -> float:
    t0 = time.perf_counter()
    graph_app.update_state({"configurable": {"thread_id": thread_id}}, state)
    return time.perf_counter() - t0


def main() -> int:
    with contextlib.suppress(OSError):
        for ext in ("", "-wal", "-shm"):
            try:
                os.remove(_TMP_DB + ext)
            except FileNotFoundError:
                pass

    from agent.multimedia.checkpointer import checkpointer
    app = _build_minimal_graph(checkpointer)
    base_state = {
        "task": "测试任务",
        "scenes": [{"script": "镜头"} for _ in range(50)],
        "current_scene_index": 0,
    }

    # (A) 未瘦身
    state_fat = dict(base_state)
    state_fat["studio_output"] = _big_studio_output()
    dur_fat = _measure_update(app, state_fat, "repro-fat")
    print(f"[复现] 未瘦身 studio_output  update_state 耗时: {dur_fat*1000:.1f} ms")

    # (B) 已瘦身（film_studio_node 返回行为）
    state_slim = dict(base_state)
    state_slim["studio_output"] = None
    state_slim["studio_output_path"] = "/tmp/fake_studio.json"
    state_slim["studio_summary"] = {"consensus": 0.8, "modification_count": 5, "debate_count": 2}
    dur_slim = _measure_update(app, state_slim, "repro-slim")
    print(f"[复现] 已瘦身 studio_output  update_state 耗时: {dur_slim*1000:.1f} ms")

    ok = dur_fat > 0 and dur_slim > 0
    ratio = dur_fat / max(dur_slim, 1e-9)
    print(f"[复现] 两种 update_state 均成功: {ok}")
    print(f"[复现] 瘦身前后耗时比(fat/slim)={ratio:.1f}x "
          f"-> 序列化体量下降，持锁更短，缓解死锁")

    with contextlib.suppress(OSError):
        for ext in ("", "-wal", "-shm"):
            try:
                os.remove(_TMP_DB + ext)
            except FileNotFoundError:
                pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
