# src/agent/multimedia/graph.py
import json
import re
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt

from .state import MultimediaState
from .prompts import (
    SHOWRUNNER_PROMPT,
    DIRECTOR_SYSTEM_PROMPT,
    END_FRAME_DIRECTOR_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    VIDEOGRAPHER_PROMPT,
    VIDEOGRAPHER_DUAL_FRAME_PROMPT,
    VIDEO_REVIEWER_SYSTEM_PROMPT,
    SAFETY_PROMPT,
)
from .checkpointer import checkpointer, make_thread_config
from .tools.text_llm import call_llm
from .tools.image_gen import generate_keyframe
from .tools.video_gen import generate_video_from_image, FreeTierQuotaExhaustedError
from .tools.vision_eval import evaluate_image, evaluate_video, design_camera_movement
from .tools.video_stitcher import stitch_videos

try:
    from .tools.embedding import compute_similarity, get_image_embedding
except Exception:
    compute_similarity = None
    get_image_embedding = None


APPROVE_ACTIONS = {"approve", "pass", "ok", "accept", "yes", "通过", "批准"}
REWRITE_ACTIONS = {"rewrite", "edit", "edit_prompt", "revise", "modify", "重写", "修改"}


def safe_parse_json(response: str) -> dict:
    cleaned = response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise Exception(f"总导演未按要求输出有效 JSON 对象！原始输出: {response[:500]}...")


def _normalize_action(raw_action: Any) -> str:
    if raw_action is None:
        return "approve"
    if isinstance(raw_action, bool):
        return "approve" if raw_action else "rewrite"
    action = str(raw_action).strip().lower()
    if action in {"通过", "批准"}:
        return "approve"
    if action in {"重写", "修改"}:
        return "rewrite"
    return action


def _normalize_decision(decision: Any) -> Dict[str, Any]:
    if isinstance(decision, dict):
        out = dict(decision)
    elif isinstance(decision, str):
        out = {"action": decision}
    elif isinstance(decision, bool):
        out = {"action": "approve" if decision else "rewrite"}
    else:
        out = {"action": "approve"}
    out["action"] = _normalize_action(out.get("action", "approve"))
    return out


def _normalize_scenes(scenes_value: Any) -> List[Dict[str, Any]]:
    if scenes_value is None:
        raise ValueError("scenes 不能为空")
    if isinstance(scenes_value, str):
        scenes_value = json.loads(scenes_value)
    if not isinstance(scenes_value, list):
        raise ValueError("scenes 必须是 list 或 JSON 数组字符串")
    normalized: List[Dict[str, Any]] =[]
    for item in scenes_value:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append({"script": item})
        else:
            normalized.append({"script": str(item)})
    return normalized


def _scene_base(script: str) -> Dict[str, Any]:
    return {
        "script": script,
        "iterations": 0,
        "critique": "无",
        "is_perfect": False,
        "image_prompt": "",
        "image_url": "",
        "embedding_similarity": None,
        "image_auto_feedback": "",
        
        # 尾帧相关字段
        "last_image_prompt": "",
        "last_image_url": "",
        "last_image_iterations": 0,
        "last_image_critique": "无",
        "last_image_is_perfect": False,
        "last_image_auto_feedback": "",

        "video_iterations": 0,
        "video_critique": "无",
        "video_is_perfect": False,
        "video_prompt": "",
        "raw_video_url": "",
        "final_video_url": "",
        "video_auto_feedback": "",
    }


# ================= 节点定义 =================
def showrunner_node(state: MultimediaState):
    print("\n--- 👑[节点1: 总导演] 正在拆解长视频分镜剧本 ---")
    prompt = SHOWRUNNER_PROMPT.format(task=state["task"])
    response = call_llm(prompt, "总导演")

    parsed_data = safe_parse_json(response)

    global_setting = parsed_data.get("global_setting", "无全局设定")
    raw_scenes = parsed_data.get("scenes", [])

    print(f"[*] 成功生成全局设定: {global_setting[:60]}...")
    print(f"[*] 成功拆解出 {len(raw_scenes)} 个连续镜头！")

    safety_check_prompt = (
        SAFETY_PROMPT
        + f"\n\n待检查内容：\n全局视觉设定：{global_setting}\n\n分镜列表：\n"
        + json.dumps([s.get("script", "") for s in raw_scenes], ensure_ascii=False, indent=2)
    )
    safety_result = call_llm(safety_check_prompt, "安全审核员")
    if "FAIL" in safety_result.upper():
        print(f"    ❌ 初始内容安全审核失败: {safety_result}")
        raise Exception(f"内容安全违规: {safety_result}")
    print("    ✅ 初始内容安全审核通过")

    processed_scenes =[]
    for i, s in enumerate(raw_scenes):
        scene_item = _scene_base(s.get("script", s.get("description", "")))
        processed_scenes.append(scene_item)
        print(f"    镜头 {i+1} 剧本已就绪: {scene_item['script'][:60]}...")

    return {
        "global_setting": global_setting,
        "scenes": processed_scenes,
        "current_scene_index": 0,
        "reference_images": [],
        "reference_embeddings":[],
        "aborted": False,
        "abort_reason": None,
        "error_log": None,
    }


def showrunner_review_gate_node(state: MultimediaState):
    payload = {
        "stage": "showrunner_review",
        "title": "总导演人工审片",
        "task": state["task"],
        "global_setting": state["global_setting"],
        "scenes": state["scenes"],
        "message": "请审查总导演输出的全局设定与分镜列表。可直接通过，也可修改 global_setting / scenes 后再继续。",
        "actions":["approve", "rewrite", "edit_prompt"],
    }
    decision = _normalize_decision(interrupt(payload))

    scenes = state["scenes"]
    global_setting = state["global_setting"]
    task = state["task"]

    if decision.get("task") is not None:
        task = str(decision["task"]).strip() or task
    if decision.get("global_setting") is not None:
        global_setting = str(decision["global_setting"]).strip() or global_setting
    if decision.get("scenes") is not None:
        scenes = _normalize_scenes(decision["scenes"])

    return {
        "task": task,
        "global_setting": global_setting,
        "scenes": scenes,
    }


def director_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    global_setting = state.get("global_setting", "")
    print(f"\n--- 🎬 [镜头 {idx+1}] 导演设计关键帧 ---")

    critique = scene.get("critique", "无")
    prompt = DIRECTOR_SYSTEM_PROMPT.format(
        global_setting=global_setting,
        script=scene["script"],
        critique=critique,
        scene_index=idx + 1
    )
    image_prompt = call_llm(prompt, "导演")

    scenes = state["scenes"].copy()
    scenes[idx]["image_prompt"] = image_prompt
    return {"scenes": scenes}


def image_gen_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🎨 [镜头 {idx+1}] 画师渲染关键帧 ---")

    image_url = generate_keyframe(scene["image_prompt"])

    scenes = state["scenes"].copy()
    scenes[idx]["image_url"] = image_url
    scenes[idx]["iterations"] += 1
    return {"scenes": scenes}


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
            print(f"    [*] embedding 相似度: {similarity_score:.4f}")
        except Exception as e:
            print(f"    ⚠️ embedding 计算失败，继续人工审核: {str(e)}")

    feedback = evaluate_image(
        scene["image_url"],
        scene["script"],
        REVIEWER_SYSTEM_PROMPT,
        previous_image_url=previous_image_url
    )

    scenes = state["scenes"].copy()
    scenes[idx]["image_auto_feedback"] = feedback
    scenes[idx]["embedding_similarity"] = similarity_score
    scenes[idx]["critique"] = feedback
    scenes[idx]["is_perfect"] = feedback.upper().startswith("PASS")

    return {"scenes": scenes}


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
        "auto_feedback": scene.get("image_auto_feedback", scene.get("critique", "")),
        "message": "请审核关键帧：可通过、重写，或修改 image_prompt 后继续。",
        "actions":["approve", "rewrite", "edit_prompt"],
    }

    decision = _normalize_decision(interrupt(payload))
    action = decision.get("action", "approve")

    scenes = state["scenes"].copy()

    if decision.get("image_prompt") is not None:
        scenes[idx]["image_prompt"] = str(decision["image_prompt"]).strip()

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


# ================= 新增：尾帧相关节点 =================
def end_frame_director_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"\n--- 🎬 [镜头 {idx+1}] 导演设计【尾帧】 ---")

    prompt = END_FRAME_DIRECTOR_PROMPT.format(
        global_setting=state.get("global_setting", ""),
        script=scene["script"],
        first_frame_prompt=scene["image_prompt"],
        critique=scene.get("last_image_critique", "无"),
        scene_index=idx + 1
    )
    last_image_prompt = call_llm(prompt, "导演")

    scenes = state["scenes"].copy()
    scenes[idx]["last_image_prompt"] = last_image_prompt
    return {"scenes": scenes}


def end_frame_gen_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🎨[镜头 {idx+1}] 画师渲染【尾帧】 ---")

    # 传入首帧作为参考图，保证高度一致性
    last_image_url = generate_keyframe(
        scene["last_image_prompt"], 
        reference_image_url=scene["image_url"]
    )
    
    scenes = state["scenes"].copy()
    scenes[idx]["last_image_url"] = last_image_url
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

    decision = _normalize_decision(interrupt(payload))
    action = decision.get("action", "approve")
    scenes = state["scenes"].copy()

    if decision.get("image_prompt") is not None:
        scenes[idx]["last_image_prompt"] = str(decision["image_prompt"]).strip()

    if action in APPROVE_ACTIONS:
        scenes[idx]["last_image_is_perfect"] = True
        scenes[idx]["last_image_critique"] = "无"

        # 【修复】将尾帧也追加到 reference_images 中，保持历史完整
        ref_images = state.get("reference_images",[]).copy()
        ref_images.append(scenes[idx]["last_image_url"])

        updates = {
            "scenes": scenes,
            "reference_images": ref_images,
        }

        if get_image_embedding is not None:
            try:
                ref_embs = state.get("reference_embeddings", []).copy()
                ref_embs.append(get_image_embedding(scenes[idx]["last_image_url"]))
                updates["reference_embeddings"] = ref_embs
            except Exception as e:
                print(f"    ⚠️ 记录 embedding 失败，但不影响主流程: {str(e)}")

        return updates

    scenes[idx]["last_image_is_perfect"] = False
    scenes[idx]["last_image_critique"] = decision.get("reason") or "Human requested rewrite"
    return {"scenes": scenes}
# ===================================================


def videographer_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🎥 [镜头 {idx+1}] 摄影师设计运镜 ---")

    # 根据是否开启双帧模式，选择不同的 Prompt
    prompt_template = VIDEOGRAPHER_DUAL_FRAME_PROMPT if state.get("use_first_last_frame") else VIDEOGRAPHER_PROMPT

    video_prompt = design_camera_movement(
        scene["image_url"],
        scene["script"],
        prompt_template,
        critique=scene.get("video_critique", "无")
    )
    print(f"    [*] 运镜提示词: {video_prompt}")

    scenes = state["scenes"].copy()
    scenes[idx]["video_prompt"] = video_prompt
    return {"scenes": scenes}


def video_gen_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🎞️ [镜头 {idx+1}] 动画师生成原片 ---")

    scenes = state["scenes"].copy()
    scenes[idx]["video_iterations"] = scenes[idx].get("video_iterations", 0) + 1

    try:
        last_url = scene.get("last_image_url", "") if state.get("use_first_last_frame") else ""
        raw_video_url = generate_video_from_image(scene["image_url"], scene["video_prompt"], last_url)

        scenes[idx]["raw_video_url"] = raw_video_url
        scenes[idx]["final_video_url"] = raw_video_url
        scenes[idx]["video_is_perfect"] = False

        print(f"    🌟 镜头 {idx+1} 制作完成！URL: {raw_video_url}")
        return {"scenes": scenes, "aborted": False, "abort_reason": None}

    except FreeTierQuotaExhaustedError as e:
        msg = str(e)
        print(f"    ⛔ [配额中止] {msg}")
        scenes[idx]["raw_video_url"] = ""
        scenes[idx]["final_video_url"] = ""
        scenes[idx]["video_is_perfect"] = False
        scenes[idx]["video_critique"] = "FAIL: " + msg
        return {
            "scenes": scenes,
            "aborted": True,
            "abort_reason": msg,
            "error_log": msg,
        }

    except Exception as e:
        msg = f"视频生成失败: {str(e)}"
        print(f"    ⛔ [视频中止] {msg}")
        scenes[idx]["raw_video_url"] = ""
        scenes[idx]["final_video_url"] = ""
        scenes[idx]["video_is_perfect"] = False
        scenes[idx]["video_critique"] = "FAIL: " + msg
        return {
            "scenes": scenes,
            "aborted": True,
            "abort_reason": msg,
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
    scenes[idx]["video_is_perfect"] = feedback.upper().startswith("PASS")

    return {"scenes": scenes}


def video_review_gate_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    # 【修复】显式获取上一镜头的图片
    previous_image_url = None
    if idx > 0:
        prev_scene = state["scenes"][idx - 1]
        previous_image_url = prev_scene.get("last_image_url") or prev_scene.get("image_url")

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
        "message": "请审核视频成片：可通过、重写，或修改 video_prompt 后再生成。",
        "actions": ["approve", "rewrite", "edit_prompt"],
    }

    decision = _normalize_decision(interrupt(payload))
    action = decision.get("action", "approve")

    scenes = state["scenes"].copy()

    if decision.get("video_prompt") is not None:
        scenes[idx]["video_prompt"] = str(decision["video_prompt"]).strip()

    if action in APPROVE_ACTIONS:
        scenes[idx]["video_is_perfect"] = True
        scenes[idx]["video_critique"] = "无"
        scenes[idx]["final_video_url"] = scene["raw_video_url"]
        return {"scenes": scenes}

    scenes[idx]["video_is_perfect"] = False
    scenes[idx]["video_critique"] = decision.get("reason") or scene.get("video_auto_feedback") or "Human requested rewrite"
    return {"scenes": scenes}


def advance_scene_node(state: MultimediaState):
    idx = state["current_scene_index"]
    next_idx = idx + 1
    print(f"--- ✅[镜头 {idx+1}] 当前镜头完成，切换到下一镜头 ---")
    return {"current_scene_index": next_idx}


def abort_node(state: MultimediaState):
    reason = state.get("abort_reason") or state.get("error_log") or "未知原因"
    print(f"\n--- ⛔ [任务中止] {reason}")
    return {"error_log": reason, "aborted": True, "abort_reason": reason}


def stitcher_node(state: MultimediaState):
    print("\n--- 🎬 [剪辑师] 正在将所有镜头拼接成最终宣传片 ---")
    video_urls = [scene["final_video_url"] for scene in state["scenes"]]
    final_movie_path = stitch_videos(video_urls, "epic_game_trailer.mp4")
    return {"final_movie_path": final_movie_path}


# ================= 路由逻辑 =================
def decide_image_quality(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    if scene["is_perfect"] or scene["iterations"] >= 3:
        if state.get("use_first_last_frame"):
            return "end_frame_director"
        return "videographer"
    return "director"

def decide_end_frame_quality(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    if scene["last_image_is_perfect"] or scene["last_image_iterations"] >= 3:
        return "videographer"
    return "end_frame_director"

def decide_after_video_generation(state: MultimediaState):
    if state.get("aborted"):
        return "abort"
    return "video_reviewer"

def decide_video_quality(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    if scene["video_is_perfect"] or scene["video_iterations"] >= 3:
        return "advance_scene"
    return "videographer"

def decide_next_scene(state: MultimediaState):
    if state["current_scene_index"] < len(state["scenes"]):
        print(f"\n🔄 [路由] 准备开始制作 镜头 {state['current_scene_index'] + 1}...")
        return "director"
    print("\n✅ [路由] 所有镜头制作完毕，移交剪辑部门！")
    return "stitcher"


# ================= 组装 Graph =================
workflow = StateGraph(MultimediaState)

workflow.add_node("showrunner", showrunner_node)
workflow.add_node("showrunner_review", showrunner_review_gate_node)
workflow.add_node("director", director_node)
workflow.add_node("image_generator", image_gen_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("image_review", image_review_gate_node)

# 新增尾帧节点
workflow.add_node("end_frame_director", end_frame_director_node)
workflow.add_node("end_frame_generator", end_frame_gen_node)
workflow.add_node("end_frame_review", end_frame_review_gate_node)

workflow.add_node("videographer", videographer_node)
workflow.add_node("video_generator", video_gen_node)
workflow.add_node("video_reviewer", video_reviewer_node)
workflow.add_node("video_review", video_review_gate_node)
workflow.add_node("advance_scene", advance_scene_node)
workflow.add_node("abort", abort_node)
workflow.add_node("stitcher", stitcher_node)

workflow.set_entry_point("showrunner")
workflow.add_edge("showrunner", "showrunner_review")
workflow.add_edge("showrunner_review", "director")

workflow.add_edge("director", "image_generator")
workflow.add_edge("image_generator", "reviewer")
workflow.add_edge("reviewer", "image_review")

workflow.add_conditional_edges(
    "image_review",
    decide_image_quality,
    {
        "end_frame_director": "end_frame_director",
        "videographer": "videographer",
        "director": "director"
    }
)

# 尾帧路由
workflow.add_edge("end_frame_director", "end_frame_generator")
workflow.add_edge("end_frame_generator", "end_frame_review")
workflow.add_conditional_edges(
    "end_frame_review",
    decide_end_frame_quality,
    {
        "videographer": "videographer",
        "end_frame_director": "end_frame_director"
    }
)

workflow.add_edge("videographer", "video_generator")
workflow.add_conditional_edges(
    "video_generator",
    decide_after_video_generation,
    {
        "video_reviewer": "video_reviewer",
        "abort": "abort"
    }
)

workflow.add_edge("video_reviewer", "video_review")
workflow.add_conditional_edges(
    "video_review",
    decide_video_quality,
    {
        "videographer": "videographer",
        "advance_scene": "advance_scene"
    }
)

workflow.add_conditional_edges(
    "advance_scene",
    decide_next_scene,
    {
        "director": "director",
        "stitcher": "stitcher"
    }
)

workflow.add_edge("abort", END)
workflow.add_edge("stitcher", END)

multimedia_agent = workflow.compile(checkpointer=checkpointer)


def get_thread_state(thread_id: str):
    return multimedia_agent.get_state(make_thread_config(thread_id))

def get_thread_history(thread_id: str):
    return list(multimedia_agent.get_state_history(make_thread_config(thread_id)))

def get_latest_interrupt_payload(thread_id: str):
    history = get_thread_history(thread_id)
    if not history:
        return None
    latest = history[0]
    tasks = getattr(latest, "tasks", ()) or ()
    for task in tasks:
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            first = interrupts[0]
            return getattr(first, "value", first)
    return None
