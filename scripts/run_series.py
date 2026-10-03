# -*- coding: utf-8 -*-
"""P2-5 跨集编排：把「短剧连续剧化」跑通。

逐集驱动 multimedia_agent，自动 approve HITL；集与集之间用
tools.narrative.export_episode_assets 把本集的 reference_sheets（资产）与
scene_anchors（尾帧/角色/色调锚点）接力注入下一集——于是后一集复用参考图
（省额度、保角色一致）并自然承接上一集结尾。

用法：
    # 读 series.json：{"name":..., "task":"主线", "episodes":[{"task":"第二集"}, ...]}
    LANGGRAPH_API_MODE=0 python scripts/run_series.py --series series.json --episodes 3 --target-duration 40
    # 或直接给目标时长、单集任务
    LANGGRAPH_API_MODE=0 python scripts/run_series.py --task "道士复仇" --episodes 2 --target-duration 40

说明：
    - 每集独立 thread（series_<n>_<ts>）；interrupt 自动 {"action":"approve"}。
    - 目标时长驱动镜头数：见 tools/narrative.plan_scene_count（P2-5）。
    - 零额度验证：加 --dry-run 则不真正 invoke（只打印每集将注入的输入）。
"""
import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 本脚本模式必须带 checkpointer 才能 HITL resume（见 run_stories.py 说明）。
os.environ["LANGGRAPH_API_MODE"] = ""
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.multimedia.graph import multimedia_agent, make_thread_config  # noqa: E402
from src.agent.multimedia.tools.narrative import (  # noqa: E402
    build_series_inputs,
    export_episode_assets,
    plan_scene_count,
    resolve_episode,
)
from langgraph.types import Command  # noqa: E402


def _is_interrupt(snap):
    for t in (getattr(snap, "tasks", None) or ()):
        if getattr(t, "interrupts", None):
            return True
    return False


def run_episode(entry: dict, idx: int) -> dict:
    """跑一集，返回其 end_state.values。"""
    tid = f"series_{idx}_{int(time.time())}"
    cfg = make_thread_config(tid)
    ep = entry.get("episode", idx)
    print(f"\n===== [第 {ep} 集] thread={tid} 启动 =====")
    print(f"  target_duration={entry.get('target_duration')} "
          f"series_assets={'有' if entry.get('series_assets') else '无'} "
          f"episode_anchors={'有' if entry.get('episode_anchors') else '无'}")

    try:
        multimedia_agent.invoke(entry, cfg)
    except Exception as e:
        print(f"  [!] 首调异常：{e}")
        return {}

    loops = 0
    while True:
        loops += 1
        if loops > 80:
            print("  [!] 超过最大循环次数，中止本集")
            break
        snap = multimedia_agent.get_state(cfg)
        if not _is_interrupt(snap):
            if not getattr(snap, "next", None):
                break
        try:
            multimedia_agent.invoke(Command(resume={"action": "approve"}), cfg)
        except Exception as e:
            print(f"  [!] resume 异常：{e}")
            break
        time.sleep(0.3)

    final = multimedia_agent.get_state(cfg)
    values = getattr(final, "values", {}) or {}
    fm = values.get("final_movie_with_audio") or values.get("final_movie_path")
    scenes = values.get("scenes") or []
    print(f"  -> 第 {ep} 集镜头数: {len(scenes)}  final_movie: {fm}")
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", help="剧集 JSON：{name, task, episodes:[{task}...]}")
    ap.add_argument("--task", help="单任务（未给 --series 时用）")
    ap.add_argument("--episodes", type=int, default=1, help="集数")
    ap.add_argument("--target-duration", type=float, default=None, help="每集目标时长（秒）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将注入的输入，不真正运行")
    args = ap.parse_args()

    base_task = args.task or ""
    episode_tasks = []
    if args.series:
        with open(args.series, "r", encoding="utf-8") as f:
            data = json.load(f)
        base_task = data.get("task", base_task)
        episode_tasks = [e.get("task", "") for e in (data.get("episodes") or [])]
        if args.episodes <= 1 and episode_tasks:
            args.episodes = len(episode_tasks) + 1

    if not base_task:
        print("✗ 需要 --series 或 --task")
        return 2

    inputs = build_series_inputs(base_task, episode_tasks, args.episodes, args.target_duration)
    print(f"共 {len(inputs)} 集；目标时长={args.target_duration}s")
    for i, e in enumerate(inputs, 1):
        n = plan_scene_count(args.target_duration, None)
        print(f"  第 {i} 集: task={e['task'][:30]}... 规划 {n} 镜（无 content_type 时按默认弧线）")

    if args.dry_run:
        print("\n[dry-run] 不实际运行。以下为每集将注入的输入键：")
        for i, e in enumerate(inputs, 1):
            print(f"  第 {i} 集: {sorted(e.keys())}")
        return 0

    prev_values = {}
    for i, entry in enumerate(inputs, 1):
        # 接力：把上一集的资产 + 锚点注入本集
        if prev_values:
            relay = export_episode_assets(
                prev_values.get("reference_sheets"),
                prev_values.get("scene_anchors"),
            )
            if relay.get("series_assets"):
                entry["series_assets"] = relay["series_assets"]
            if relay.get("episode_anchors"):
                entry["episode_anchors"] = relay["episode_anchors"]
        entry["episode"] = i
        prev_values = run_episode(entry, i)

    print("\n===== 连续剧完成 =====")
    print(f"  共 {len(inputs)} 集；末集 final_movie: "
          f"{prev_values.get('final_movie_with_audio') or prev_values.get('final_movie_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
