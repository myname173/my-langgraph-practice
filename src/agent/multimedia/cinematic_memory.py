"""
Cinematic Memory Bank（Phase 5 / P0-3 持久化版）
=================================================
职责：维护跨镜头的电影风格一致性。
      记录已处理镜头的摄影风格、光照风格、剪辑风格，供后续镜头的专业 agent 参考。

架构位置：
    全局状态（非节点），被 film_studio_orchestrator 读写。

P0-3（架构评审）持久化语义：
    - 记忆库按 thread_id 作用域隔离，落盘到 ``data/memory/<thread_id>.json``；
    - 写入即原子落盘（tmp + rename），进程重启 / 崩溃恢复后首次访问自动从磁盘重建，
      修复"长任务中断恢复后风格记忆清空 → 跨镜头一致性静默退化"；
    - 模块级 dict 仅作**进程内缓存层**（评审所述"单例只做进程内缓存"）；
    - ``reset_memory(thread_id)`` 在新 run 开始时清空缓存 + 删除落盘文件；
    - ``reference_embeddings`` 审计结论：该数据本就存放于 MultimediaState 通道
      （随 SqliteSaver 快照持久化，graph.end_frame_director 写入 / qc 消费），
      embedding.py 无模块级值缓存，无需额外处理。
    - 并行模式（P0-2）下 film_studio 在扇出前仅运行一次，记忆库因此只记录一次
      规划快照——这是并行模式既有的文档化降级，与持久化正交。

设计原则：
    - 轻量 JSON 结构，无向量数据库、无第三方依赖
    - 按镜头索引存储风格快照
    - 提供 get_current_style_state() 聚合最近风格趋势
    - 新 trailer 开始时按线程 reset
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# 项目根目录：cinematic_memory.py -> multimedia -> agent -> src -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MEM_DIR = _PROJECT_ROOT / "data" / "memory"

# ============================================================
# 1. 线程作用域记忆库（模块级缓存 + 文件持久化）
# ============================================================

_banks: Dict[str, Dict[int, Dict[str, Any]]] = {}
_loaded: set = set()  # 本进程内已尝试从磁盘加载过的线程（避免重复 IO）


def _safe_tag(thread_id: str) -> str:
    tag = "".join(c for c in (thread_id or "") if c.isalnum() or c in "-_")
    return tag or "_global"


def _mem_file(thread_tag: str) -> Path:
    return _MEM_DIR / f"{thread_tag}.json"


def _bank(thread_id: str) -> Dict[int, Dict[str, Any]]:
    """取线程记忆库：进程内缓存；首次访问时从磁盘重建（P0-3 跨进程恢复）。"""
    key = _safe_tag(thread_id)
    if key not in _loaded:
        _loaded.add(key)
        f = _mem_file(key)
        if f.exists():
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                _banks[key] = {int(k): v for k, v in (raw or {}).items()}
                print(f"    [记忆库] 已从落盘重建 {len(_banks[key])} 个镜头的风格快照（thread={key}）")
            except Exception as e:  # noqa: BLE001 - 记忆恢复失败不阻断主流程
                print(f"    [WARN] 记忆库读取失败（忽略，视为空）: {e}")
                _banks[key] = {}
        else:
            _banks[key] = {}
    return _banks.setdefault(key, {})


def _persist(thread_id: str) -> None:
    key = _safe_tag(thread_id)
    try:
        _MEM_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _mem_file(key).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_banks.get(key, {}), ensure_ascii=False), encoding="utf-8")
        tmp.replace(_mem_file(key))  # 原子替换
    except Exception as e:  # noqa: BLE001 - 落盘失败不阻断主流程（缓存仍在）
        print(f"    [WARN] 记忆库落盘失败（不影响主流程）: {e}")


# ============================================================
# 2. 记忆操作函数（thread_id 作用域）
# ============================================================

def record_scene_style(
    thread_id: str,
    scene_index: int,
    cinematography: Dict[str, Any],
    lighting: Dict[str, Any],
    editing: Dict[str, Any],
) -> None:
    """
    记录一个镜头的电影风格快照（写入即原子落盘）。

    Args:
        thread_id: 线程标识（记忆作用域）
        scene_index: 镜头索引
        cinematography: cinematography_agent 的输出
        lighting: lighting_agent 的输出
        editing: editor_agent 的输出
    """
    bank = _bank(thread_id)
    bank[scene_index] = {
        "scene_index": scene_index,
        "cinematography": cinematography.get("style_summary", {}),
        "lighting": lighting.get("style_summary", {}),
        "editing": editing.get("style_summary", {}),
        "coherence_scores": {
            "camera": cinematography.get("consistency_score", 1.0),
            "lighting": lighting.get("consistency_score", 1.0),
            "pacing": editing.get("pacing_score", 1.0),
        },
    }
    _persist(thread_id)


def get_previous_scene_style(
    thread_id: str,
    scene_index: int,
) -> Optional[Dict[str, Any]]:
    """获取前一个镜头的风格记录（无记录返回 None）。"""
    return _bank(thread_id).get(scene_index - 1)


def get_all_recorded_styles(thread_id: str) -> List[Dict[str, Any]]:
    """获取所有已记录镜头的风格快照列表。"""
    bank = _bank(thread_id)
    return [bank[i] for i in sorted(bank.keys())]


def get_current_style_state(thread_id: str) -> Dict[str, Any]:
    """获取当前全局风格状态（所有已记录镜头的聚合），用于全局风格报告。"""
    bank = _bank(thread_id)
    if not bank:
        return {
            "scenes_recorded": 0,
            "established_style": "none",
            "camera_consistency": 1.0,
            "lighting_consistency": 1.0,
            "pacing_consistency": 1.0,
        }

    records = get_all_recorded_styles(thread_id)
    n = len(records)

    cam_scores = [r["coherence_scores"]["camera"] for r in records]
    light_scores = [r["coherence_scores"]["lighting"] for r in records]
    pace_scores = [r["coherence_scores"]["pacing"] for r in records]

    all_shot_types = set()
    all_transitions = set()
    for r in records:
        all_shot_types.update(r.get("cinematography", {}).get("shot_types", []))
        all_transitions.update(r.get("editing", {}).get("transitions", []))

    return {
        "scenes_recorded": n,
        "established_style": "established" if n >= 2 else "forming",
        "camera_consistency": round(sum(cam_scores) / n, 2),
        "lighting_consistency": round(sum(light_scores) / n, 2),
        "pacing_consistency": round(sum(pace_scores) / n, 2),
        "all_shot_types": sorted(all_shot_types),
        "all_transitions": sorted(all_transitions),
    }


def reset_memory(thread_id: str = "") -> None:
    """清空指定线程的记忆库（缓存 + 落盘文件）。新 trailer 开始时调用。"""
    key = _safe_tag(thread_id)
    _banks.pop(key, None)
    _loaded.discard(key)
    try:
        f = _mem_file(key)
        if f.exists():
            f.unlink()
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] 记忆库落盘文件删除失败（忽略）: {e}")


# ============================================================
# 3. 全局电影意识报告（Global Film Brain）
# ============================================================

def generate_film_brain_report(
    global_setting: str,
    total_scenes: int,
    thread_id: str = "",
) -> str:
    """生成全局电影意识报告（导演意图摘要），注入后续镜头的 prompt。"""
    state = get_current_style_state(thread_id)

    lines = [
        f"[Global Film Brain — {state['scenes_recorded']}/{total_scenes} scenes processed]",
        f"Style Status: {state['established_style']}",
        f"Camera Consistency: {state['camera_consistency']:.0%}",
        f"Lighting Consistency: {state['lighting_consistency']:.0%}",
        f"Pacing Consistency: {state['pacing_consistency']:.0%}",
    ]

    if state.get("all_shot_types"):
        lines.append(f"Established Shot Vocabulary: {', '.join(state['all_shot_types'])}")
    if state.get("all_transitions"):
        lines.append(f"Established Transition Palette: {', '.join(state['all_transitions'])}")

    if state["scenes_recorded"] >= 2:
        lines.append("")
        lines.append("[Directorial Note]: Maintain established visual language.")
        lines.append("New scenes should feel like they belong in the same film —")
        lines.append("consistent camera grammar, lighting mood, and editing rhythm.")
    elif state["scenes_recorded"] == 1:
        lines.append("")
        lines.append("[Directorial Note]: First scene established the visual tone.")
        lines.append("Next scene should build on this foundation.")

    return "\n".join(lines)
