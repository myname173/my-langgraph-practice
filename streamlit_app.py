# streamlit_app.py
import json
import os
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st
from langgraph.types import Command

from src.agent.multimedia.graph import (
    multimedia_agent,
    get_thread_state,
    get_thread_history,
    get_latest_interrupt_payload,
)
from src.agent.multimedia.task_store import (
    init_task_store,
    save_task_state,
    list_task_states,
    load_task_state,
    get_latest_task_state,
    get_latest_active_task_state,
)

DEFAULT_TASK = (
    "制作一段约30秒的电影级科幻动作短片预告。"
    "整体风格为冷色调未来都市，类似《银翼杀手2049》与《攻壳机动队》结合，强调霓虹灯、雨夜、反射光与孤独氛围。"
    "主角是一位女性赛博特工，短银发，穿黑色战术风衣，左眼为发光义眼，手持能量手枪。"
    "核心视觉：高楼林立的未来城市、霓虹灯反射在湿漉街道、漂浮广告屏、蓝紫色灯光、细雨与蒸汽。"
    "必须包含以下镜头："
    "1. 全景：夜晚城市天际线，霓虹灯闪烁，雨水覆盖街道；"
    "2. 中景：女特工站在街角，风衣随风摆动，义眼发光扫描环境；"
    "3. 动作：敌方无人机出现，主角迅速拔枪射击，能量弹划过雨夜；"
    "4. 结尾：主角转身走入黑暗巷道，霓虹灯在她身后逐渐熄灭。"
    "整体节奏：冷静 → 紧张 → 爆发 → 留白收尾，电影级CG质感。"
)

TASK_TEMPLATES = {
    "电影级科幻动作短片": {
        "project_type": "电影级科幻动作短片",
        "style": "冷色调、霓虹、雨夜、反射光、孤独感",
        "mood": "冷静 → 紧张 → 爆发 → 留白",
        "protagonist": "女性赛博特工，短银发，黑色战术风衣，左眼发光义眼",
        "scene": "未来都市，高楼林立，雨夜街道，漂浮广告屏，蒸汽与蓝紫色灯光",
        "duration": 30,
        "shot_count": 4,
        "must_have": "全景城市天际线；中景角色登场；动作冲突；结尾留白",
        "avoid": "避免过度血腥、过于写实的伤害细节、杂乱镜头",
    },
    "游戏预告片": {
        "project_type": "游戏预告片",
        "style": "史诗、电影感、强对比光影、商业级CG",
        "mood": "恢弘 → 紧张 → 高潮 → 收束",
        "protagonist": "年轻英雄 / 女战士 / 赛博猎人",
        "scene": "宏大世界观场景，标志性地标，强烈空间层次",
        "duration": 30,
        "shot_count": 4,
        "must_have": "主角亮相；世界观展示；冲突升级；高潮收尾",
        "avoid": "避免人物漂移、镜头抖动、过度复杂的场景元素",
    },
    "奇幻冒险短片": {
        "project_type": "奇幻冒险短片",
        "style": "梦幻、明亮、史诗、神秘",
        "mood": "探索 → 发现 → 对峙 → 远景收尾",
        "protagonist": "年轻冒险者，带有独特标志性服装",
        "scene": "古代遗迹、漂浮岛屿、森林、神殿、光粒子",
        "duration": 30,
        "shot_count": 4,
        "must_have": "世界观展示；角色探索；危险出现；高潮镜头",
        "avoid": "避免过暗、过于血腥、过于复杂的动作分解",
    },
    "产品宣传短片": {
        "project_type": "产品宣传短片",
        "style": "简洁、高级、明亮、商业感",
        "mood": "吸引 → 展示 → 强调 → 收尾",
        "protagonist": "产品本身是主角",
        "scene": "干净背景、核心卖点场景、应用场景",
        "duration": 20,
        "shot_count": 3,
        "must_have": "产品特写；功能演示；卖点总结",
        "avoid": "避免画面拥挤、信息过载、无关元素",
    },
}

STATUS_LABELS = {
    "RUNNING": "运行中",
    "PAUSED": "待审",
    "DONE": "已完成",
    "ABORTED": "已中止",
    "UNKNOWN": "未知",
}


def init_session() -> None:
    init_task_store()

    defaults = {
        "thread_id": "",
        "thread_id_input": "",
        "pending_interrupt": None,
        "last_result": None,
        "final_state": None,
        "latest_snapshot": None,
        "started": False,
        "pending_action": None,
        "pending_task_defaults": None,
        "auto_bootstrapped": False,
        "task_mode": "quick",
        "task_title": "未命名任务",
        "task_tags_text": "",
        "task_template_name": "电影级科幻动作短片",
        "task_project_type": "电影级科幻动作短片",
        "task_duration": 30,
        "task_shot_count": 4,
        "task_style": TASK_TEMPLATES["电影级科幻动作短片"]["style"],
        "task_mood": TASK_TEMPLATES["电影级科幻动作短片"]["mood"],
        "task_protagonist": TASK_TEMPLATES["电影级科幻动作短片"]["protagonist"],
        "task_scene": TASK_TEMPLATES["电影级科幻动作短片"]["scene"],
        "task_must_have": TASK_TEMPLATES["电影级科幻动作短片"]["must_have"],
        "task_avoid": TASK_TEMPLATES["电影级科幻动作短片"]["avoid"],
        "task_notes": "",
        "task_text": DEFAULT_TASK,
        "task_use_first_last_frame": False, # 新增首尾帧开关
        "fork_source_thread_id": "",
        "fork_source_checkpoint_id": "",
        "fork_target_thread_id": "",
        "fork_editor_ready": False,
        "fork_title": "",
        "fork_tags_text": "",
        "fork_task_text": "",
        "fork_global_setting": "",
        "parent_thread_id": "",
        "fork_from_checkpoint_id": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sync_shared_to_pro_widgets() -> None:
    for field in["task_title", "task_tags_text", "task_text"]:
        pro_key = f"pro_{field}"
        if field in st.session_state:
            st.session_state[pro_key] = st.session_state[field]

def sync_pro_to_shared() -> None:
    for field in["task_title", "task_tags_text", "task_text"]:
        pro_key = f"pro_{field}"
        if pro_key in st.session_state:
            st.session_state[field] = st.session_state[pro_key]

def sync_thread_from_input() -> None:
    value = st.session_state.thread_id_input.strip()
    if value:
        st.session_state.thread_id = value

def _normalize_tags_text(text: str) -> List[str]:
    return[t.strip() for t in text.split(",") if t.strip()]

def _tags_to_text(tags: Any) -> str:
    if tags is None:
        return ""
    if isinstance(tags, list):
        return ", ".join(str(t).strip() for t in tags if str(t).strip())
    if isinstance(tags, str):
        return tags
    return str(tags)

def build_plain_task_prompt(
    project_name: str, duration: int, shot_count: int, style: str, mood: str,
    protagonist: str, scene: str, must_have: str, avoid: str, notes: str,
) -> str:
    parts =[
        f"制作一段约{duration}秒的{project_name}。",
        f"整体风格：{style}。",
        f"整体情绪与节奏：{mood}。",
        f"主角设定：{protagonist}。",
        f"主要场景与环境：{scene}。",
        f"必须包含以下{shot_count}个连续镜头要点：{must_have}。",
        f"需要避免的内容：{avoid}。",
    ]
    if notes.strip():
        parts.append(f"额外要求：{notes.strip()}。")
    parts.append("请输出适合拆分为多个连贯镜头的完整创意任务描述，方便后续生成分镜、关键帧和视频。")
    return "".join(parts)

def on_template_change() -> None:
    template_name = st.session_state.task_template_name
    data = TASK_TEMPLATES.get(template_name, TASK_TEMPLATES["电影级科幻动作短片"])
    st.session_state.task_project_type = data["project_type"]
    st.session_state.task_duration = data["duration"]
    st.session_state.task_shot_count = data["shot_count"]
    st.session_state.task_style = data["style"]
    st.session_state.task_mood = data["mood"]
    st.session_state.task_protagonist = data["protagonist"]
    st.session_state.task_scene = data["scene"]
    st.session_state.task_must_have = data["must_have"]
    st.session_state.task_avoid = data["avoid"]
    if not st.session_state.task_title or st.session_state.task_title == "未命名任务":
        st.session_state.task_title = template_name

def apply_pre_widget_bootstrap() -> None:
    pending = st.session_state.pending_task_defaults
    if isinstance(pending, dict):
        for key, value in pending.items():
            st.session_state[key] = value
        st.session_state.pending_task_defaults = None
    sync_shared_to_pro_widgets()

def collect_task_metadata() -> Dict[str, Any]:
    title = st.session_state.task_title.strip() if st.session_state.task_title else ""
    tags = _normalize_tags_text(st.session_state.task_tags_text or "")
    template_name = st.session_state.task_template_name or ""
    project_type = st.session_state.task_project_type or template_name or "自定义任务"
    mode = st.session_state.task_mode or "unknown"
    duration = st.session_state.task_duration
    shot_count = st.session_state.task_shot_count
    parent_thread_id = st.session_state.get("parent_thread_id") or None
    fork_from_checkpoint_id = st.session_state.get("fork_from_checkpoint_id") or None

    return {
        "title": title or project_type or "未命名任务",
        "tags": tags,
        "template_name": template_name,
        "project_type": project_type,
        "mode": mode,
        "duration": int(duration) if duration is not None else None,
        "shot_count": int(shot_count) if shot_count is not None else None,
        "parent_thread_id": parent_thread_id,
        "fork_from_checkpoint_id": fork_from_checkpoint_id,
    }

def derive_task_text_if_needed() -> str:
    task_text = (st.session_state.task_text or "").strip()
    if task_text:
        return task_text
    return build_plain_task_prompt(
        project_name=st.session_state.task_project_type or "短片",
        duration=int(st.session_state.task_duration or 30),
        shot_count=int(st.session_state.task_shot_count or 4),
        style=st.session_state.task_style or "",
        mood=st.session_state.task_mood or "",
        protagonist=st.session_state.task_protagonist or "",
        scene=st.session_state.task_scene or "",
        must_have=st.session_state.task_must_have or "",
        avoid=st.session_state.task_avoid or "",
        notes=st.session_state.task_notes or "",
    )

def row_to_session_defaults(row: Dict[str, Any], *, thread_id: Optional[str] = None) -> Dict[str, Any]:
    tags_text = _tags_to_text(row.get("tags",[]))
    return {
        "thread_id": thread_id or row["thread_id"],
        "thread_id_input": thread_id or row["thread_id"],
        "task_text": row.get("task_text", "") or DEFAULT_TASK,
        "task_title": row.get("title", "") or "未命名任务",
        "task_tags_text": tags_text,
        "task_template_name": row.get("template_name", "") or "电影级科幻动作短片",
        "task_project_type": row.get("project_type", "") or row.get("template_name", "") or "自定义任务",
        "task_mode": row.get("mode", "unknown") or "unknown",
        "task_duration": row.get("duration", 30) or 30,
        "task_shot_count": row.get("shot_count", 4) or 4,
        "parent_thread_id": row.get("parent_thread_id"),
        "fork_from_checkpoint_id": row.get("fork_from_checkpoint_id"),
    }

def load_checkpoint_snapshot(thread_id: str, checkpoint_id: str):
    config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    return multimedia_agent.get_state(config)

def clear_checkpoint_fork_editor() -> None:
    st.session_state.fork_source_thread_id = ""
    st.session_state.fork_source_checkpoint_id = ""
    st.session_state.fork_target_thread_id = ""
    st.session_state.fork_editor_ready = False

def prepare_checkpoint_fork_editor(thread_id: str, checkpoint_id: str) -> None:
    snapshot = load_checkpoint_snapshot(thread_id, checkpoint_id)
    values = getattr(snapshot, "values", None) or {}
    task_row = load_task_state(thread_id) or {}

    st.session_state.fork_source_thread_id = thread_id
    st.session_state.fork_source_checkpoint_id = checkpoint_id
    st.session_state.fork_target_thread_id = str(uuid.uuid4())
    st.session_state.fork_editor_ready = True

    st.session_state.fork_title = f"{task_row.get('title') or task_row.get('project_type') or '分叉任务'}（分叉）"
    st.session_state.fork_tags_text = _tags_to_text(task_row.get("tags",[]))
    st.session_state.fork_task_text = values.get("task") or task_row.get("task_text") or DEFAULT_TASK
    st.session_state.fork_global_setting = values.get("global_setting") or task_row.get("summary", {}).get("global_setting", "")

    st.session_state.task_mode = "fork"
    st.session_state.parent_thread_id = thread_id
    st.session_state.fork_from_checkpoint_id = checkpoint_id

def perform_checkpoint_fork() -> None:
    source_thread_id = (st.session_state.fork_source_thread_id or "").strip()
    checkpoint_id = (st.session_state.fork_source_checkpoint_id or "").strip()
    target_thread_id = (st.session_state.fork_target_thread_id or "").strip() or str(uuid.uuid4())

    if not source_thread_id or not checkpoint_id:
        st.error("没有可用的分叉源 checkpoint。")
        return

    snapshot = load_checkpoint_snapshot(source_thread_id, checkpoint_id)
    values = getattr(snapshot, "values", None) or {}

    branch_task_text = (st.session_state.fork_task_text or "").strip() or values.get("task", "")
    branch_global_setting = (st.session_state.fork_global_setting or "").strip() or values.get("global_setting", "")

    meta = collect_task_metadata()
    save_task_state(
        thread_id=target_thread_id,
        task_text=branch_task_text,
        status="RUNNING",
        stage="forking",
        summary=summarize_current_state(values),
        pending_payload=None,
        title=st.session_state.fork_title or meta["title"],
        tags=_normalize_tags_text(st.session_state.fork_tags_text or ""),
        template_name=st.session_state.task_template_name or meta["template_name"],
        project_type=st.session_state.task_project_type or meta["project_type"],
        mode="fork",
        duration=meta["duration"],
        shot_count=meta["shot_count"],
        parent_thread_id=source_thread_id,
        fork_from_checkpoint_id=checkpoint_id,
        final_movie_path=None,
        abort_reason=None,
        last_error=None,
    )

    updates = {}
    if branch_task_text:
        updates["task"] = branch_task_text
    if branch_global_setting:
        updates["global_setting"] = branch_global_setting

    fork_config = {"configurable": {"thread_id": target_thread_id, "checkpoint_id": checkpoint_id}}
    if updates:
        fork_config = multimedia_agent.update_state(fork_config, updates)

    result = multimedia_agent.invoke(None, fork_config)
    payload = extract_interrupt(result)

    st.session_state.thread_id = target_thread_id
    st.session_state.thread_id_input = target_thread_id
    st.session_state.started = True
    st.session_state.last_result = result
    st.session_state.pending_interrupt = payload
    st.session_state.final_state = None

    st.session_state.task_mode = "fork"
    st.session_state.task_title = (st.session_state.fork_title or "分叉任务").strip()
    st.session_state.task_tags_text = st.session_state.fork_tags_text or ""
    st.session_state.task_text = branch_task_text
    st.session_state.parent_thread_id = source_thread_id
    st.session_state.fork_from_checkpoint_id = checkpoint_id

    persist_task_state_from_thread(target_thread_id, pending_payload=payload)

def render_checkpoint_fork_editor() -> None:
    st.subheader("Checkpoint 分叉")
    st.caption("先从下面的时间轴选择一个 checkpoint，再在这里编辑后创建分叉。")

    if not st.session_state.fork_source_checkpoint_id:
        st.info("还没有选择分叉源。请先在时间轴里点击某个 checkpoint 的“从此分叉”按钮。")
        return

    thread_id = st.session_state.fork_source_thread_id
    checkpoint_id = st.session_state.fork_source_checkpoint_id

    try:
        snapshot = load_checkpoint_snapshot(thread_id, checkpoint_id)
        values = getattr(snapshot, "values", None) or {}
        next_nodes = list(getattr(snapshot, "next", ()) or ())
        metadata = getattr(snapshot, "metadata", {}) or {}
    except Exception as e:
        st.error(f"读取分叉源 checkpoint 失败：{e}")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("source thread", thread_id[:8] + "..." if len(thread_id) > 8 else thread_id)
    col2.metric("checkpoint", checkpoint_id[:8] + "..." if len(checkpoint_id) > 8 else checkpoint_id)
    col3.metric("step", metadata.get("step", "N/A"))

    with st.expander("分叉源 checkpoint 预览", expanded=True):
        st.json({
            "next": next_nodes,
            "task": values.get("task", ""),
            "global_setting": values.get("global_setting", ""),
            "current_scene_index": values.get("current_scene_index"),
            "final_movie_path": values.get("final_movie_path"),
            "aborted": values.get("aborted"),
            "abort_reason": values.get("abort_reason") or values.get("error_log"),
            "target_thread_id": st.session_state.fork_target_thread_id,
        })

    if st.button("清空分叉目标", use_container_width=True):
        clear_checkpoint_fork_editor()
        st.rerun()

    with st.form("checkpoint_fork_form", clear_on_submit=False):
        st.text_input("分叉标题", key="fork_title")
        st.text_input("标签（用逗号分隔）", key="fork_tags_text")
        st.text_area("分叉后的任务描述", key="fork_task_text", height=220)
        st.text_area("分叉后的全局设定", key="fork_global_setting", height=160)
        submitted = st.form_submit_button("创建分叉并继续")

    if submitted:
        perform_checkpoint_fork()
        st.rerun()

def queue_restore_task(thread_id: str) -> None:
    row = load_task_state(thread_id)
    if row is None:
        st.error(f"找不到 thread_id={thread_id} 对应的任务")
        return
    st.session_state.pending_task_defaults = row_to_session_defaults(row, thread_id=thread_id)
    st.session_state.pending_action = "restore"

def queue_fork_task(row: Dict[str, Any]) -> None:
    new_thread_id = str(uuid.uuid4())
    defaults = row_to_session_defaults(row, thread_id=new_thread_id)
    defaults["task_title"] = f"{defaults['task_title']}（副本）"
    defaults["parent_thread_id"] = row["thread_id"]
    defaults["fork_from_checkpoint_id"] = None
    st.session_state.pending_task_defaults = defaults
    st.session_state.pending_action = "start_new"

def queue_new_task_from_current_inputs(mode: str) -> None:
    new_thread_id = str(uuid.uuid4())
    title = st.session_state.task_title.strip() if st.session_state.task_title else ""
    task_text = derive_task_text_if_needed()

    st.session_state.pending_task_defaults = {
        "thread_id": new_thread_id,
        "thread_id_input": new_thread_id,
        "task_mode": mode,
        "task_title": title or st.session_state.task_project_type or "未命名任务",
        "task_tags_text": st.session_state.task_tags_text or "",
        "task_template_name": st.session_state.task_template_name or "",
        "task_project_type": st.session_state.task_project_type or "",
        "task_duration": st.session_state.task_duration,
        "task_shot_count": st.session_state.task_shot_count,
        "task_style": st.session_state.task_style,
        "task_mood": st.session_state.task_mood,
        "task_protagonist": st.session_state.task_protagonist,
        "task_scene": st.session_state.task_scene,
        "task_must_have": st.session_state.task_must_have,
        "task_avoid": st.session_state.task_avoid,
        "task_notes": st.session_state.task_notes,
        "task_text": task_text,
    }
    st.session_state.pending_action = "start_new"

def get_config() -> Dict[str, Any]:
    return {"configurable": {"thread_id": st.session_state.thread_id}}

def extract_interrupt(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    if hasattr(first, "value"):
        return first.value
    return first

def _safe_get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def summarize_current_state(values: Dict[str, Any]) -> Dict[str, Any]:
    scenes = values.get("scenes", []) or[]
    idx = values.get("current_scene_index")
    current_scene: Dict[str, Any] = {}
    if isinstance(idx, int) and 0 <= idx < len(scenes):
        current_scene = scenes[idx] or {}

    return {
        "task": values.get("task", ""),
        "global_setting": values.get("global_setting", ""),
        "current_scene_index": idx,
        "scene_count": len(scenes),
        "current_scene_script": current_scene.get("script", ""),
        "image_prompt": current_scene.get("image_prompt", ""),
        "video_prompt": current_scene.get("video_prompt", ""),
        "embedding_similarity": current_scene.get("embedding_similarity"),
        "critique": current_scene.get("critique", ""),
        "video_critique": current_scene.get("video_critique", ""),
        "image_url": current_scene.get("image_url", ""),
        "last_image_url": current_scene.get("last_image_url", ""), # 记录尾帧
        "video_url": current_scene.get("raw_video_url") or current_scene.get("final_video_url", ""),
        "final_movie_path": values.get("final_movie_path"),
        "aborted": bool(values.get("aborted")),
        "abort_reason": values.get("abort_reason") or values.get("error_log"),
    }

def normalize_status(values: Dict[str, Any], pending_payload: Optional[Dict[str, Any]]) -> str:
    if values.get("aborted"):
        return "ABORTED"
    if values.get("final_movie_path"):
        return "DONE"
    if pending_payload is not None:
        return "PAUSED"
    return "RUNNING"

def persist_task_state_from_thread(thread_id: str, pending_payload: Optional[Dict[str, Any]] = None) -> None:
    try:
        snapshot = get_thread_state(thread_id)
        values = getattr(snapshot, "values", None) or {}
    except Exception:
        row = load_task_state(thread_id)
        if row is None:
            return
        values = row.get("summary", {}) or {}

    summary = summarize_current_state(values)
    status = normalize_status(values, pending_payload)

    stage = None
    if pending_payload and isinstance(pending_payload, dict):
        stage = pending_payload.get("stage")
    if stage is None:
        if values.get("final_movie_path"):
            stage = "stitcher"
        elif values.get("aborted"):
            stage = "abort"
        else:
            idx = values.get("current_scene_index")
            if isinstance(idx, int):
                stage = f"scene_{idx + 1}"

    meta = collect_task_metadata()
    save_task_state(
        thread_id=thread_id,
        task_text=values.get("task", st.session_state.task_text or DEFAULT_TASK),
        status=status,
        stage=stage,
        summary=summary,
        pending_payload=pending_payload,
        title=meta["title"],
        tags=meta["tags"],
        template_name=meta["template_name"],
        project_type=meta["project_type"],
        mode=meta["mode"],
        duration=meta["duration"],
        shot_count=meta["shot_count"],
        parent_thread_id=meta["parent_thread_id"],
        fork_from_checkpoint_id=meta["fork_from_checkpoint_id"],
        final_movie_path=values.get("final_movie_path"),
        abort_reason=values.get("abort_reason") or values.get("error_log"),
        last_error=values.get("error_log"),
    )

def restore_thread_state_only(thread_id: str) -> None:
    thread_id = thread_id.strip()
    if not thread_id:
        return

    st.session_state.thread_id = thread_id
    st.session_state.started = True
    st.session_state.last_result = None
    st.session_state.pending_interrupt = None
    st.session_state.final_state = None
    st.session_state.latest_snapshot = None

    try:
        snapshot = get_thread_state(thread_id)
        values = getattr(snapshot, "values", None) or {}
        pending_payload = get_latest_interrupt_payload(thread_id)

        if values.get("final_movie_path") or values.get("aborted"):
            st.session_state.final_state = values
        elif pending_payload is not None:
            st.session_state.pending_interrupt = pending_payload
        else:
            st.session_state.latest_snapshot = snapshot

        persist_task_state_from_thread(thread_id, pending_payload=pending_payload)

    except Exception as e:
        row = load_task_state(thread_id)
        if row is not None:
            summary = row.get("summary", {}) or {}
            if row.get("status") == "DONE":
                st.session_state.final_state = summary
            elif row.get("status") in {"PAUSED", "INTERRUPTED"}:
                st.session_state.pending_interrupt = row.get("pending_payload")
            else:
                st.session_state.latest_snapshot = {"values": summary}
            st.session_state.started = True
            return
        st.error(f"恢复线程失败：{e}")

def run_graph(input_or_command: Any) -> Dict[str, Any]:
    return multimedia_agent.invoke(input_or_command, config=get_config())

def start_new_run() -> None:
    st.session_state.started = True
    st.session_state.pending_interrupt = None
    st.session_state.last_result = None
    st.session_state.final_state = None
    st.session_state.latest_snapshot = None

    task_text = derive_task_text_if_needed()
    if not task_text.strip():
        task_text = DEFAULT_TASK

    initial_state = {
        "task": task_text,
        "scenes":[],
        "current_scene_index": 0,
        "global_setting": "",
        "reference_images":[],
        "reference_embeddings":[],
        "final_movie_path": None,
        "error_log": None,
        "aborted": False,
        "abort_reason": None,
        "use_first_last_frame": st.session_state.task_use_first_last_frame, # 传入首尾帧开关
    }

    meta = collect_task_metadata()
    save_task_state(
        thread_id=st.session_state.thread_id,
        task_text=task_text,
        status="RUNNING",
        stage="starting",
        summary=summarize_current_state(initial_state),
        pending_payload=None,
        title=meta["title"],
        tags=meta["tags"],
        template_name=meta["template_name"],
        project_type=meta["project_type"],
        mode=meta["mode"],
        duration=meta["duration"],
        shot_count=meta["shot_count"],
        parent_thread_id=meta["parent_thread_id"],
        fork_from_checkpoint_id=meta["fork_from_checkpoint_id"],
        final_movie_path=None,
        abort_reason=None,
        last_error=None,
    )

    with st.spinner("启动图文视频流水线..."):
        result = run_graph(initial_state)

    st.session_state.last_result = result
    payload = extract_interrupt(result)

    if payload is not None:
        st.session_state.pending_interrupt = payload
        persist_task_state_from_thread(st.session_state.thread_id, pending_payload=payload)
    else:
        try:
            snapshot = get_thread_state(st.session_state.thread_id)
            values = getattr(snapshot, "values", None) or {}
            if values.get("final_movie_path") or values.get("aborted"):
                st.session_state.final_state = values
            persist_task_state_from_thread(st.session_state.thread_id)
        except Exception:
            pass

def resume_with_decision(decision: Dict[str, Any]) -> None:
    with st.spinner("继续执行图文视频流水线..."):
        result = run_graph(Command(resume=decision))

    st.session_state.last_result = result
    payload = extract_interrupt(result)

    if payload is not None:
        st.session_state.pending_interrupt = payload
        persist_task_state_from_thread(st.session_state.thread_id, pending_payload=payload)
    else:
        st.session_state.pending_interrupt = None
        try:
            snapshot = get_thread_state(st.session_state.thread_id)
            values = getattr(snapshot, "values", None) or {}
            if values.get("final_movie_path") or values.get("aborted"):
                st.session_state.final_state = values
            persist_task_state_from_thread(st.session_state.thread_id)
        except Exception:
            pass

def apply_pending_actions() -> None:
    if st.session_state.pending_action == "start_new":
        st.session_state.pending_action = None
        start_new_run()
    elif st.session_state.pending_action == "restore":
        st.session_state.pending_action = None
        restore_thread_state_only(st.session_state.thread_id)

def auto_bootstrap_latest_task() -> None:
    if st.session_state.auto_bootstrapped:
        return
    if st.session_state.thread_id_input.strip():
        st.session_state.auto_bootstrapped = True
        return
    latest = get_latest_active_task_state() or get_latest_task_state()
    if latest is not None:
        st.session_state.pending_task_defaults = row_to_session_defaults(latest, thread_id=latest["thread_id"])
        st.session_state.pending_action = "restore"
    st.session_state.auto_bootstrapped = True

def render_quick_start_card() -> None:
    st.subheader("新手模式：只填自然语言，也能直接开始")
    st.caption("先选模板，再写几句话。下面的任务描述可以直接编辑。")

    st.selectbox("选择一个模板", list(TASK_TEMPLATES.keys()), key="task_template_name", on_change=on_template_change)

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("任务标题", key="task_title", placeholder="例如：未来都市短片预告")
        st.text_input("标签（用逗号分隔）", key="task_tags_text", placeholder="科幻, 霓虹, 女性主角")
        st.slider("时长（秒）", 10, 60, key="task_duration")
        st.slider("镜头数量", 3, 6, key="task_shot_count")
    with c2:
        st.text_input("项目类型", key="task_project_type")
        st.text_area("画面风格", key="task_style", height=90)
        st.text_area("情绪与节奏", key="task_mood", height=90)

    st.text_area("主角 / 关键对象", key="task_protagonist", height=90)
    st.text_area("主要场景 / 环境", key="task_scene", height=90)
    st.text_area("必须出现的镜头要点", key="task_must_have", height=90)
    st.text_area("要避免什么", key="task_avoid", height=90)
    st.text_area("额外要求（可选）", key="task_notes", height=90)

    suggested = build_plain_task_prompt(
        project_name=st.session_state.task_project_type or "短片",
        duration=int(st.session_state.task_duration or 30),
        shot_count=int(st.session_state.task_shot_count or 4),
        style=st.session_state.task_style or "",
        mood=st.session_state.task_mood or "",
        protagonist=st.session_state.task_protagonist or "",
        scene=st.session_state.task_scene or "",
        must_have=st.session_state.task_must_have or "",
        avoid=st.session_state.task_avoid or "",
        notes=st.session_state.task_notes or "",
    )

    st.caption("任务描述（可直接编辑）")
    st.text_area("任务描述", key="task_text", height=260)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("应用当前配置生成任务描述", use_container_width=True):
            st.session_state.pending_task_defaults = {"task_text": suggested}
            st.rerun()
    with col_b:
        if st.button("开始生成", use_container_width=True):
            queue_new_task_from_current_inputs(mode="quick")
            st.rerun()

    with st.expander("系统根据当前配置的建议版本", expanded=False):
        st.write(suggested)

def render_professional_mode() -> None:
    st.subheader("专业模式：直接编辑完整任务")
    st.caption("适合已经知道要做什么的时候，直接写完整描述。")

    st.text_input("任务标题", key="pro_task_title", placeholder="例如：赛博特工短片", on_change=sync_pro_to_shared)
    st.text_input("标签（用逗号分隔）", key="pro_task_tags_text", placeholder="科幻, 未来都市, 电影感", on_change=sync_pro_to_shared)
    st.text_area("完整任务描述", key="pro_task_text", height=360, on_change=sync_pro_to_shared)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("开始生成（专业模式）", use_container_width=True):
            st.session_state.task_mode = "professional"
            queue_new_task_from_current_inputs(mode="professional")
            st.rerun()
    with col_b:
        st.caption("你也可以先把描述改好，再开始。")

def render_task_card(item: Dict[str, Any]) -> None:
    summary = item.get("summary", {}) or {}
    title = item.get("title", "") or "未命名任务"
    tags = item.get("tags", []) or[]
    tag_text = ", ".join(tags) if tags else "无标签"
    task_text = item.get("task_text", "")

    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"**{title}**")
            st.caption(f"thread_id: {item['thread_id']}")
            st.write(task_text[:180] + ("..." if len(task_text) > 180 else ""))
            st.caption(f"模板：{item.get('template_name') or '未设置'} | 类型：{item.get('project_type') or '未设置'} | 标签：{tag_text}")
        with right:
            status = STATUS_LABELS.get(item.get("status", "UNKNOWN"), item.get("status", "UNKNOWN"))
            st.metric("状态", status)
            if item.get("updated_at"):
                st.caption(f"更新: {item['updated_at']}")
            if item.get("stage"):
                st.caption(f"阶段: {item['stage']}")

        cols = st.columns(3)
        with cols[0]:
            if st.button("继续这个任务", key=f"resume_{item['thread_id']}"):
                queue_restore_task(item["thread_id"])
                st.rerun()
        with cols[1]:
            if st.button("复制为新任务", key=f"fork_{item['thread_id']}"):
                queue_fork_task(load_task_state(item["thread_id"]) or item)
                st.rerun()
        with cols[2]:
            if st.button("打开时间轴", key=f"timeline_{item['thread_id']}"):
                st.session_state.pending_task_defaults = row_to_session_defaults(
                    load_task_state(item["thread_id"]) or {"thread_id": item["thread_id"], "task_text": DEFAULT_TASK},
                    thread_id=item["thread_id"],
                )
                st.rerun()

        with st.expander("摘要", expanded=False):
            st.json({
                "scene_index": item.get("scene_index"),
                "scene_count": item.get("scene_count"),
                "final_movie_path": item.get("final_movie_path"),
                "abort_reason": item.get("abort_reason"),
                "current_scene_script": summary.get("current_scene_script", ""),
            })

def render_recovery_center() -> None:
    st.subheader("恢复中心")
    st.caption("这里可以恢复暂停任务、复制旧任务、或者继续最近一次任务。")

    cols = st.columns(3)
    with cols[0]:
        if st.button("继续最近一个任务", use_container_width=True):
            latest = get_latest_active_task_state() or get_latest_task_state()
            if latest is None:
                st.info("当前没有可恢复的任务。")
            else:
                queue_restore_task(latest["thread_id"])
                st.rerun()
    with cols[1]:
        if st.button("新建一个空白任务", use_container_width=True):
            st.session_state.pending_task_defaults = {
                "thread_id": str(uuid.uuid4()),
                "thread_id_input": "",
                "task_mode": "quick",
                "task_title": "未命名任务",
                "task_tags_text": "",
                "task_template_name": "电影级科幻动作短片",
                "task_project_type": "电影级科幻动作短片",
                "task_duration": 30,
                "task_shot_count": 4,
                "task_style": TASK_TEMPLATES["电影级科幻动作短片"]["style"],
                "task_mood": TASK_TEMPLATES["电影级科幻动作短片"]["mood"],
                "task_protagonist": TASK_TEMPLATES["电影级科幻动作短片"]["protagonist"],
                "task_scene": TASK_TEMPLATES["电影级科幻动作短片"]["scene"],
                "task_must_have": TASK_TEMPLATES["电影级科幻动作短片"]["must_have"],
                "task_avoid": TASK_TEMPLATES["电影级科幻动作短片"]["avoid"],
                "task_notes": "",
                "task_text": DEFAULT_TASK,
            }
            st.session_state.pending_action = "start_new"
            st.rerun()
    with cols[2]:
        if st.button("刷新任务列表", use_container_width=True):
            st.rerun()

    tasks = list_task_states(limit=50)
    if not tasks:
        st.info("还没有历史任务。")
        return
    for item in tasks:
        render_task_card(item)

def render_interrupt_panel(payload: Dict[str, Any]) -> None:
    stage = payload.get("stage", "unknown")
    st.info(f"当前暂停点：{stage}")

    with st.expander("原始暂停数据", expanded=False):
        st.json(payload)

    if stage == "showrunner_review":
        st.subheader("总导演审片")
        st.caption("你可以直接通过，也可以改全局设定和分镜。")
        with st.form("showrunner_review_form"):
            action = st.radio("操作", ["approve", "rewrite", "edit_prompt"], horizontal=True, index=0)
            task_edit = st.text_area("任务描述", value=payload.get("task", ""), height=140)
            global_setting_edit = st.text_area("全局设定", value=payload.get("global_setting", ""), height=160)
            scenes_text = st.text_area(
                "分镜列表（JSON 数组）",
                value=json.dumps(payload.get("scenes",[]), ensure_ascii=False, indent=2),
                height=280,
            )
            reason = st.text_input("备注（可选）", value="")
            submitted = st.form_submit_button("提交并继续")

        if submitted:
            try:
                scenes = json.loads(scenes_text)
                if not isinstance(scenes, list):
                    raise ValueError("分镜列表必须是 JSON 数组")
            except Exception as e:
                st.error(f"分镜解析失败：{e}")
                return
            resume_with_decision({
                "action": action, "task": task_edit, "global_setting": global_setting_edit,
                "scenes": scenes, "reason": reason,
            })
            st.rerun()

    elif stage == "image_review":
        st.subheader("关键帧审片")
        c1, c2 = st.columns([1, 1])
        with c1:
            img_url = payload.get("image_url", "")
            if img_url:
                st.image(img_url, caption="当前关键帧", use_container_width=True)
            prev_url = payload.get("reference_image_url")
            if prev_url:
                st.image(prev_url, caption="上一镜头关键帧", use_container_width=True)
        with c2:
            sim = payload.get("embedding_similarity")
            st.metric("Embedding similarity", "N/A" if sim is None else f"{sim:.4f}")
            st.text_area("当前 image_prompt", value=payload.get("image_prompt", ""), height=200, disabled=True)
            st.text_area("自动反馈 / 失败原因", value=payload.get("auto_feedback", ""), height=160, disabled=True)

        with st.form("image_review_form"):
            action = st.radio("操作", ["approve", "rewrite", "edit_prompt"], horizontal=True, index=0)
            edited_prompt = st.text_area("修改后的 image_prompt", value=payload.get("image_prompt", ""), height=200)
            reason = st.text_input("备注（可选）", value="")
            submitted = st.form_submit_button("提交并继续")

        if submitted:
            resume_with_decision({"action": action, "image_prompt": edited_prompt, "reason": reason})
            st.rerun()

    # 新增：尾帧审片界面
    elif stage == "end_frame_review":
        st.subheader("尾帧审片")
        c1, c2 = st.columns([1, 1])
        with c1:
            img_url = payload.get("image_url", "")
            if img_url:
                st.image(img_url, caption="当前生成的【尾帧】", use_container_width=True)
            prev_url = payload.get("reference_image_url")
            if prev_url:
                st.image(prev_url, caption="参考图：当前镜头的【首帧】", use_container_width=True)
        with c2:
            st.text_area("当前尾帧 Prompt", value=payload.get("image_prompt", ""), height=200, disabled=True)

        with st.form("end_frame_review_form"):
            action = st.radio("操作",["approve", "rewrite", "edit_prompt"], horizontal=True, index=0)
            edited_prompt = st.text_area("修改后的尾帧 Prompt", value=payload.get("image_prompt", ""), height=200)
            reason = st.text_input("备注（可选）", value="")
            submitted = st.form_submit_button("提交并继续")

        if submitted:
            resume_with_decision({"action": action, "image_prompt": edited_prompt, "reason": reason})
            st.rerun()

    elif stage == "video_review":
        st.subheader("视频审片")
        c1, c2 = st.columns([1, 1])
        with c1:
            video_url = payload.get("video_url", "")
            if video_url:
                st.video(video_url)
            prev_url = payload.get("reference_image_url")
            if prev_url:
                st.image(prev_url, caption="上一镜头关键帧", use_container_width=True)
        with c2:
            sim = payload.get("embedding_similarity")
            st.metric("Embedding similarity", "N/A" if sim is None else f"{sim:.4f}")
            st.text_area("当前 video_prompt", value=payload.get("video_prompt", ""), height=200, disabled=True)
            st.text_area("自动反馈 / 失败原因", value=payload.get("auto_feedback", ""), height=160, disabled=True)

        with st.form("video_review_form"):
            action = st.radio("操作",["approve", "rewrite", "edit_prompt"], horizontal=True, index=0)
            edited_prompt = st.text_area("修改后的 video_prompt", value=payload.get("video_prompt", ""), height=200)
            reason = st.text_input("备注（可选）", value="")
            submitted = st.form_submit_button("提交并继续")

        if submitted:
            resume_with_decision({"action": action, "video_prompt": edited_prompt, "reason": reason})
            st.rerun()

    else:
        st.warning("未知暂停点。你可以查看原始数据后再继续。")
        with st.form("unknown_review_form"):
            raw = st.text_area("resume payload", value=json.dumps({"action": "approve"}, ensure_ascii=False, indent=2))
            submitted = st.form_submit_button("提交")
        if submitted:
            try:
                decision = json.loads(raw)
                resume_with_decision(decision)
                st.rerun()
            except Exception as e:
                st.error(f"JSON 解析失败：{e}")

def render_final_state(final_state: Dict[str, Any]) -> None:
    if final_state.get("aborted"):
        st.error(f"任务中止：{final_state.get('abort_reason') or final_state.get('error_log')}")
        return
    movie_path = final_state.get("final_movie_path")
    if movie_path:
        st.success("任务完成")
        st.write(f"最终成片路径：`{movie_path}`")
        if os.path.exists(movie_path):
            st.video(movie_path)
    scenes = final_state.get("scenes",[])
    if scenes:
        st.subheader("镜头结果")
        st.json(scenes)

def get_latest_checkpoint_id(thread_id: str) -> Optional[str]:
    try:
        snapshot = get_thread_state(thread_id)
        config = _safe_get(snapshot, "config", {}) or {}
        configurable = _safe_get(config, "configurable", {}) or {}
        if isinstance(configurable, dict):
            checkpoint_id = configurable.get("checkpoint_id")
            if checkpoint_id:
                return str(checkpoint_id)
    except Exception:
        pass
    try:
        history = list(get_thread_history(thread_id))
        for snap in history:
            config = _safe_get(snap, "config", {}) or {}
            configurable = _safe_get(config, "configurable", {}) or {}
            if isinstance(configurable, dict):
                checkpoint_id = configurable.get("checkpoint_id")
                if checkpoint_id:
                    return str(checkpoint_id)
    except Exception:
        pass
    return None

def build_fork_tree(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    node_map: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        thread_id = task["thread_id"]
        node_map[thread_id] = {"task": task, "children": []}
    roots: List[Dict[str, Any]] =[]
    for task in tasks:
        thread_id = task["thread_id"]
        parent_thread_id = task.get("parent_thread_id")
        if parent_thread_id and parent_thread_id in node_map and parent_thread_id != thread_id:
            node_map[parent_thread_id]["children"].append(node_map[thread_id])
        else:
            roots.append(node_map[thread_id])
    def sort_node(node: Dict[str, Any]) -> None:
        node["children"].sort(key=lambda n: n["task"].get("updated_at", "") or "", reverse=True)
        for child in node["children"]:
            sort_node(child)
    roots.sort(key=lambda n: n["task"].get("updated_at", "") or "", reverse=True)
    for root in roots:
        sort_node(root)
    return roots

def render_fork_tree_node(node: Dict[str, Any], depth: int = 0, current_thread_id: Optional[str] = None, path_visited: Optional[set] = None) -> None:
    task = node["task"]
    children = node["children"]
    thread_id = task["thread_id"]

    if path_visited is None:
        path_visited = set()
    if thread_id in path_visited:
        with st.container(border=True):
            st.warning(f"检测到循环引用：{thread_id}")
        return
    next_visited = set(path_visited)
    next_visited.add(thread_id)

    title = task.get("title") or task.get("project_type") or "未命名任务"
    status = STATUS_LABELS.get(task.get("status", "UNKNOWN"), task.get("status", "UNKNOWN"))
    is_current = thread_id == current_thread_id
    prefix = "🟢" if is_current else "⚪"
    indent = "　" * depth

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
        with c1:
            st.markdown(f"{indent}{prefix} **{title}**")
            st.caption(f"thread_id: {thread_id}")
            st.caption(f"状态: {status} | 阶段: {task.get('stage') or 'unknown'} | 模板: {task.get('template_name') or '未设置'} | 类型: {task.get('project_type') or '未设置'}")
            if task.get("parent_thread_id"):
                st.caption(f"parent_thread_id: {task.get('parent_thread_id')}")
        with c2:
            if st.button("进入", key=f"tree_enter_{thread_id}"):
                queue_restore_task(thread_id)
                st.rerun()
        with c3:
            latest_checkpoint_id = get_latest_checkpoint_id(thread_id)
            if latest_checkpoint_id:
                if st.button("从此分叉", key=f"tree_fork_{thread_id}"):
                    prepare_checkpoint_fork_editor(thread_id, latest_checkpoint_id)
                    st.rerun()
            else:
                st.caption("无 checkpoint")
        with c4:
            if st.button("时间轴", key=f"tree_timeline_{thread_id}"):
                st.session_state.thread_id = thread_id
                st.session_state.thread_id_input = thread_id
                st.rerun()

    for child in children:
        render_fork_tree_node(child, depth=depth + 1, current_thread_id=current_thread_id, path_visited=next_visited)

def render_checkpoint_timeline(thread_id: str) -> None:
    st.subheader("🕒 执行时间轴")
    st.caption(f"thread_id: **{thread_id}**")

    try:
        history = list(get_thread_history(thread_id))
        if not history:
            st.info("该线程还没有任何 checkpoint 记录。")
            return
    except Exception as e:
        st.error(f"读取时间轴失败：{e}")
        return

    st.success(f"共找到 {len(history)} 个 checkpoint（从新到旧排序）")

    for idx, snapshot in enumerate(reversed(history)):
        config = getattr(snapshot, "config", {}) or {}
        configurable = config.get("configurable", {}) or {}
        checkpoint_id = configurable.get("checkpoint_id", f"unknown_{idx}")
        metadata = getattr(snapshot, "metadata", {}) or {}
        values = getattr(snapshot, "values", {}) or {}
        next_nodes = getattr(snapshot, "next",[])
        step = metadata.get("step", idx + 1)

        with st.expander(f"📍 Checkpoint {len(history) - idx} • Step {step} • ID: `{str(checkpoint_id)[:8]}...`", expanded=(idx == 0)):
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.write(f"**Checkpoint ID**：`{checkpoint_id}`")
                st.caption(f"下一步节点：{next_nodes}")
            with col2:
                if values.get("final_movie_path"):
                    st.success("✅ 已完成最终成片")
                elif values.get("aborted"):
                    st.error(f"❌ 已中止 - {values.get('abort_reason', '未知原因')}")
                else:
                    st.info("⏳ 运行中 / 已暂停")
            with col3:
                if st.button("🔀 从此分叉", key=f"timeline_fork_{checkpoint_id}", use_container_width=True, type="primary"):
                    prepare_checkpoint_fork_editor(thread_id, str(checkpoint_id))
                    st.rerun()

            summary = summarize_current_state(values)
            with st.expander("📊 当前状态摘要", expanded=False):
                st.json({
                    "current_scene_index": summary.get("current_scene_index"),
                    "scene_count": summary.get("scene_count"),
                    "final_movie_path": bool(summary.get("final_movie_path")),
                    "aborted": summary.get("aborted", False),
                })
                if summary.get("current_scene_script"):
                    st.caption("当前镜头脚本预览")
                    st.write(summary["current_scene_script"][:280] + "..." if len(summary["current_scene_script"]) > 280 else summary["current_scene_script"])

def render_fork_tree_board() -> None:
    st.subheader("任务分叉树")
    st.caption("按 parent_thread_id 组织任务树，绿色节点是当前线程。")
    tasks = list_task_states(limit=200)
    if not tasks:
        st.info("当前没有任何任务记录。")
        return
    current_thread_id = st.session_state.get("thread_id") or None
    roots = build_fork_tree(tasks)
    st.write(f"共 {len(tasks)} 个任务，{len(roots)} 个根节点。")
    for root in roots:
        render_fork_tree_node(root, depth=0, current_thread_id=current_thread_id)

def render_task_table() -> None:
    st.subheader("任务列表")
    st.caption("按更新时间排序的任务总览。")
    tasks = list_task_states(limit=200)
    if not tasks:
        st.info("当前没有历史任务。")
        return

    table_rows =[]
    for item in tasks:
        summary = item.get("summary", {}) or {}
        table_rows.append({
            "thread_id": item["thread_id"],
            "title": item.get("title", ""),
            "status": item.get("status", ""),
            "stage": item.get("stage", ""),
            "template_name": item.get("template_name", ""),
            "project_type": item.get("project_type", ""),
            "tags": ", ".join(item.get("tags", []) or[]),
            "updated_at": item.get("updated_at", ""),
            "scene_index": item.get("scene_index"),
            "scene_count": item.get("scene_count"),
            "final_movie_path": item.get("final_movie_path"),
            "abort_reason": item.get("abort_reason"),
            "parent_thread_id": item.get("parent_thread_id"),
            "fork_from_checkpoint_id": item.get("fork_from_checkpoint_id"),
            "task_preview": (item.get("task_text", "")[:90] + "..." if len(item.get("task_text", "")) > 90 else item.get("task_text", "")),
            "current_scene_script": summary.get("current_scene_script", ""),
        })

    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    selected_thread_id = st.selectbox(
        "选择一个 thread_id",
        options=[row["thread_id"] for row in table_rows],
        format_func=lambda tid: next(
            (f"{tid} | {STATUS_LABELS.get(next((r['status'] for r in table_rows if r['thread_id'] == tid), 'UNKNOWN'), '未知')} | {next((r['stage'] or 'unknown' for r in table_rows if r['thread_id'] == tid), 'unknown')}"),
            tid,
        ),
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("恢复选中任务", use_container_width=True):
            queue_restore_task(selected_thread_id)
            st.rerun()
    with c2:
        if st.button("复制为新任务", use_container_width=True):
            row = next((r for r in table_rows if r["thread_id"] == selected_thread_id), None)
            if row is not None:
                queue_fork_task(load_task_state(selected_thread_id) or row)
                st.rerun()
    with c3:
        if st.button("打开时间轴", use_container_width=True):
            st.session_state.thread_id = selected_thread_id
            st.session_state.thread_id_input = selected_thread_id
            st.rerun()

def render_task_registry() -> None:
    st.subheader("任务看板")
    view_mode = st.radio("展示模式", ["列表", "分叉树"], horizontal=True, key="task_board_view_mode")
    if view_mode == "列表":
        render_task_table()
    else:
        render_fork_tree_board()

def render_current_state() -> None:
    if not st.session_state.thread_id:
        st.info("当前没有 thread_id。")
        return
    try:
        snapshot = get_thread_state(st.session_state.thread_id)
        values = getattr(snapshot, "values", None) or {}
    except Exception as e:
        st.error(f"读取当前状态失败：{e}")
        return

    summary = summarize_current_state(values)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("current_scene_index", summary.get("current_scene_index", "N/A"))
    c2.metric("scenes", summary.get("scene_count", 0))
    c3.metric("aborted", "是" if summary.get("aborted") else "否")
    c4.metric("final_movie_path", "有" if summary.get("final_movie_path") else "无")

    with st.expander("当前 state 原文", expanded=False):
        st.json(values)

    scenes = values.get("scenes", []) or[]
    if scenes:
        st.markdown("### 镜头摘要")
        for idx, scene in enumerate(scenes):
            with st.container(border=True):
                s1, s2 = st.columns([2, 1])
                with s1:
                    st.markdown(f"**镜头 {idx + 1}**")
                    st.write(scene.get("script", ""))
                with s2:
                    st.write({
                        "image_ok": scene.get("is_perfect"),
                        "last_image_ok": scene.get("last_image_is_perfect"),
                        "video_ok": scene.get("video_is_perfect"),
                        "image_iterations": scene.get("iterations", 0),
                        "last_image_iterations": scene.get("last_image_iterations", 0),
                        "video_iterations": scene.get("video_iterations", 0),
                        "similarity": scene.get("embedding_similarity"),
                    })

def main() -> None:
    st.set_page_config(page_title="Multimedia HITL Console", layout="wide")
    init_session()
    
    apply_pre_widget_bootstrap()
    sync_shared_to_pro_widgets()   
    auto_bootstrap_latest_task()
    st.title("多模态视频生成工作台")
    st.caption("这版加入了更完整的任务元数据和更灵活的新手编辑。")

    with st.sidebar:
        st.subheader("任务设置")
        # 新增：首尾帧双控模式开关
        st.session_state.task_use_first_last_frame = st.toggle(
            "启用首尾帧双控生成 (更精准的动作控制，但耗时增加)", 
            value=st.session_state.get("task_use_first_last_frame", False)
        )
        
        st.divider()
        st.subheader("线程控制")
        st.text_input("thread_id", key="thread_id_input", on_change=sync_thread_from_input)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("新建线程", use_container_width=True):
                st.session_state.pending_task_defaults = {
                    "thread_id": str(uuid.uuid4()),
                    "thread_id_input": "",
                }
                st.session_state.pending_action = "start_new"
                st.rerun()
        with c2:
            if st.button("恢复输入线程", use_container_width=True):
                queue_restore_task(st.session_state.thread_id_input.strip())
                st.rerun()

        st.write("当前 thread_id")
        st.code(st.session_state.thread_id or "未设置")

        if st.button("清空当前会话", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        with st.expander("帮助说明", expanded=False):
            st.markdown(
                """
                **怎么用：**
                1. 在“新手模式”里选模板并直接编辑任务描述；
                2. 或在“专业模式”里直接写完整描述；
                3. 任务暂停时，会自动停在审片界面；
                4. 刷新页面后，系统会自动恢复最近任务；
                5. 在“恢复中心”里可以继续旧任务，或者复制旧任务重新开始。
                """
            )

        if st.session_state.thread_id:
            try:
                snapshot = get_thread_state(st.session_state.thread_id)
                values = getattr(snapshot, "values", None) or {}
                with st.expander("当前线程摘要", expanded=False):
                    st.json({
                        "current_scene_index": values.get("current_scene_index"),
                        "aborted": values.get("aborted"),
                        "abort_reason": values.get("abort_reason") or values.get("error_log"),
                        "final_movie_path": values.get("final_movie_path"),
                        "scene_count": len(values.get("scenes", []) or[]),
                        "reference_images_count": len(values.get("reference_images", []) or[]),
                    })
            except Exception as e:
                st.caption(f"无法读取当前 checkpoint：{e}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["新手模式", "专业模式", "恢复中心", "分叉恢复", "时间轴", "当前状态"])

    with tab1:
        render_quick_start_card()
    with tab2:
        render_professional_mode()
    with tab3:
        render_recovery_center()
    with tab4:
        render_checkpoint_fork_editor()
    with tab5:
        if st.session_state.thread_id:
            render_checkpoint_timeline(st.session_state.thread_id)
        else:
            st.info("没有可查看的 thread_id。")
    with tab6:
        render_current_state()

    apply_pending_actions()

    if st.session_state.pending_interrupt:
        with st.expander("当前暂停点", expanded=True):
            render_interrupt_panel(st.session_state.pending_interrupt)

    if st.session_state.final_state:
        with st.expander("最终结果", expanded=True):
            render_final_state(st.session_state.final_state)

if __name__ == "__main__":
    main()
