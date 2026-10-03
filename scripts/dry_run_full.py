# -*- coding: utf-8 -*-
r"""全链路零额度模拟验证（P0-2 并行 + 设置面板 + 熔断台账 联合冒烟）。

原理：在 graph 模块边界把所有外部模型调用替换为「带时延的确定性假实现」，
但图结构、路由、LangGraph Send 并行扇出、熔断器、成本台账、真实 moviepy
剪辑拼接 / 字幕生成 / 配音混音全部走真实代码——不消耗任何模型额度。

用法（项目根目录）：
    python scripts/dry_run_full.py [--shots 5] [--sequential]

产物：output/clips/dryrun_<ts>/FINAL_成片.mp4（真实可播放的无声→有声成片）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RUN_METRICS_ENABLED", "0")           # 指标落盘静默
os.environ.setdefault("MULTIMEDIA_PARALLEL_SHOTS", "1")     # P0-2 并行扇出开启

TMPD = Path(tempfile.mkdtemp(prefix="dryrun_"))

# ────────────────────────── mock：模型边界假实现 ──────────────────────────
LLM_SLEEP = 0.2
KF_SLEEP = 0.25
CAM_SLEEP = 0.3
EVAL_SLEEP = 0.15

_BEATS = ["setup", "develop", "turn", "climax", "resolution"]
_EMOTIONS = ["紧张", "希望", "惊惧", "狂怒", "释然"]
_ACTIONS = [
    "少年剑客按剑立于屋脊，雨幕中霓虹明灭",
    "剑客掠过全息广告间隙，斗篷猎猎",
    "黑袍法师自天台升起，符文环绕",
    "剑光与法术对撞，气浪掀翻水洼",
    "雨停，剑客收剑，霓虹在水面碎成星子",
]


def _scenes_json(n: int) -> str:
    scenes = []
    for i in range(n):
        scenes.append({
            "script": f"镜头{i + 1}：赛博朋克霓虹屋顶，{_ACTIONS[i % len(_ACTIONS)]}",
            "beat": _BEATS[i % len(_BEATS)],
            "emotion": _EMOTIONS[i % len(_EMOTIONS)],
            "action_beat": _ACTIONS[i % len(_ACTIONS)],
            "visual_elements": ["少年剑客", "霓虹屋顶", "全息广告", "雨幕"],
            "transition_in": "" if i == 0 else "剑光渐隐转场",
            "dialogue": f"第{i + 1}幕旁白：风起于青萍之末，浪成于微澜之间。",
            "duration_seconds": 1.0,
            "camera_note": "低角度跟拍",
            "scale": "中景",
        })
    return json.dumps({"global_setting": "近未来赛博朋克都市，永夜霓虹，酸雨不停",
                       "scenes": scenes}, ensure_ascii=False)


def fake_call_llm(prompt, role="", **kw):
    time.sleep(LLM_SLEEP)
    if role == "总导演":
        n = _WANT_SHOTS
        data = json.loads(_scenes_json(n))
        return json.dumps(data, ensure_ascii=False)
    if role == "安全审核员":
        return "PASS: 内容安全审核通过，无违规描写。"
    if role == "风格抽取":
        return json.dumps({"style_key": "custom:mock", "style_description": "mock cyber style",
                           "style_prompt": "cyberpunk neon, cinematic"}, ensure_ascii=False)
    if role == "提示词翻译":
        return "mock english prompt, neon rooftop, hero with sword"
    return f"Mock {role} prompt: hero on neon rooftop, dramatic rim light"


def fake_generate_keyframe(prompt, reference_image_urls=None, ref_strength=None, thread_id="", **kw):
    time.sleep(KF_SLEEP)
    name = "kf_" + str(abs(hash(prompt)) % 10_000_000).zfill(7) + ".png"
    f = TMPD / name
    f.write_bytes(b"\x89PNG\r\n\x1a\nMOCK")
    return str(f)


def _real_mp4(path: Path, duration: float = 1.0, size=(320, 180), color=(20, 18, 34)) -> str:
    """真实可解码的 mp4（moviepy + 自带 ffmpeg），供拼接/混音消费。"""
    from moviepy import ColorClip
    path.parent.mkdir(parents=True, exist_ok=True)
    ColorClip(size=size, color=color, duration=duration).write_videofile(
        str(path), fps=12, codec="libx264", audio=False, logger=None)
    return str(path)


def fake_generate_video_from_image(*args, **kwargs):
    """用真实 moviepy 生成占位 mp4（随机色块 + 真实 H.264），零模型额度。"""
    src = str(args[0]) if args else ""
    tag = str(abs(hash(src)) % 10_000_000).zfill(7)
    duration = float(kwargs.get("duration") or 1.0) or 1.0
    time.sleep(0.2)
    return _real_mp4(TMPD / f"vid_{tag}.mp4", duration=min(duration, 2.0))


def fake_design_camera_movement(*a, **k):
    time.sleep(CAM_SLEEP)
    return "MOCK camera: slow push in, hero leaping through neon rain, hold on sword"


def fake_evaluate_image(*a, **k):
    time.sleep(EVAL_SLEEP)
    return "PASS: composition and identity consistent"


def fake_evaluate_video(*a, **k):
    time.sleep(EVAL_SLEEP)
    return "PASS: visible motion, no static frame"


def fake_critic_final_cut(*a, **k):
    time.sleep(EVAL_SLEEP)
    return {"available": True, "score": 0.8, "issues": [], "method": "vlm_final_cut"}


def fake_assess_character_consistency(*a, **k):
    time.sleep(EVAL_SLEEP)
    return {
        "available": True,
        "same_character": True,
        "score": 0.95,
        "drifting": False,
        "reason": "mock: same character",
        "method": "vlm_pair",
        "threshold": 0.6,
    }


def fake_build_visual_context(script, global_setting="", scene_index=0, total_scenes=1,
                              prev_anchors=None, **kw):
    return {
        "intent": "action",
        "pacing_phase": "opening" if scene_index == 0 else "climax" if scene_index >= total_scenes - 2 else "buildup",
        "camera_rules": ["mock: low angle tracking shot"],
        "lighting_rules": ["mock: neon rim light with haze"],
        "continuity_directives": [],
        "formatted_prompt_block": "MOCK visual context",
        "energy": 0.6,
    }


def fake_retrieve_rag_context(**kw):
    return None  # RAG 静默降级（不连 chroma / 嵌入 API）


def fake_generate_reference_sheets(**kw):
    return {"characters": {}, "props": {}, "environments": {}}


_CAMS = [("wide", "pan", "low", "24mm"), ("medium", "dolly_in", "eye", "35mm"),
         ("close_up", "static", "high", "85mm"), ("aerial", "crane_up", "top", "18mm"),
         ("medium", "tracking", "low", "50mm")]
_ORIENTS = ["facing_camera", "profile_left", "back_to_camera", "three_quarter", "profile_right"]
_LIGHTS = ["neon rim light", "soft key + haze", "hard top light", "moonlit ambient", "practical neon wash"]
_PACES = [0.6, 0.75, 0.5, 0.95, 0.4]


def fake_generate_shot_strategy(script, visual_context=None, **kw):
    shots = []
    for i in range(_WANT_SHOTS):
        c = _CAMS[i % len(_CAMS)]
        shots.append({
            "shot_name": f"S{i + 1}",
            "camera": {"type": c[0], "movement": c[1], "angle": c[2], "lens": c[3],
                       "orientation": _ORIENTS[i % len(_ORIENTS)]},
            "orientation": _ORIENTS[i % len(_ORIENTS)],
            "lighting": {"setup": _LIGHTS[i % len(_LIGHTS)]},
            "focus_object": "hero",
            "focus_hint": "hero face and sword",
            "motion_line": "hero leaps through neon rain",
            "next_shot_hint": "carry motion into next shot",
            "transition_preference": "match_cut",
            "lens_feel": "35mm standard",
            "emotion": _EMOTIONS[i % len(_EMOTIONS)],
            "pacing_weight": _PACES[i % len(_PACES)],
            "duration": 1.0,
        })
    return {
        "shots": shots,
        "hero_shot_index": 0,
        "end_shot_index": _WANT_SHOTS - 1,
        "formatted_plan": "MOCK PLAN: " + script[:60],
        "focus_object": "hero",
        "intent": "action",
        "genres": ["cyberpunk", "action"],
        "pacing_phase": "opening",
        "ambient_context": "neon rain, night city",
        "camera_overrides": {},
    }


def fake_orchestrate_film_studio(shot_plan=None, sequence_graph=None, visual_context=None,
                                 scene_index=0, total_scenes=1, global_setting="", **kw):
    return {
        "formatted_studio_report": "[MOCK] 多 Agent 工作室：共识 90%，无冲突",
        "studio_consensus": 0.9,
        "modifications": [],
        "debate": [],
        "studio_refined_plan": {"shot_plan": shot_plan or {}, "sequence_graph": sequence_graph or {}},
    }


def _silent_wav(path: Path, seconds: float = 2.0, rate: int = 16000) -> str:
    import wave
    w = wave.open(str(path), "w")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(b"\x00\x00" * int(rate * seconds))
    w.close()
    return str(path)


def fake_build_segmented_voiceover(seg_specs, voice="xiaoxiao", **kw):
    timings = {int(s["index"]): (float(s["start"]), float(s["start"]) + 1.2) for s in seg_specs}
    return _silent_wav(TMPD / "vo_segmented.wav", 2.0), timings


def fake_generate_voiceover(text, voice="xiaoxiao", out_path=None, **kw):
    return _silent_wav(TMPD / "vo_full.wav", 2.0)


def fake_generate_bgm(mood=None, duration=30.0, **kw):
    return None  # 无曲库，优雅跳过 BGM


# ────────────────────────── 主流程 ──────────────────────────
def main() -> int:
    global _WANT_SHOTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--sequential", action="store_true", help="强制关闭并行（对照）")
    ap.add_argument("--target-duration", type=float, default=None,
                    help="P2-5 目标时长（秒），驱动 showrunner 反推镜头数")
    ap.add_argument("--episode", type=int, default=None,
                    help="P2-5 连续剧集数（>1 时注入承接提示）")
    args = ap.parse_args()
    _WANT_SHOTS = max(2, min(args.shots, 5))
    if args.sequential:
        os.environ["MULTIMEDIA_PARALLEL_SHOTS"] = "0"

    t0 = time.perf_counter()
    import agent.multimedia.graph as G
    import agent.multimedia.tools.backend_health as bh
    print(f"[dry-run] 模块导入 {time.perf_counter() - t0:.1f}s · 临时目录 {TMPD}")

    # 台账重定向到临时目录，不污染真实成本账本
    bh._LEDGER_DIR = TMPD / "cost_ledger"

    # ── 打补丁：模型边界全部 mock ──
    G.call_llm = fake_call_llm
    G.generate_keyframe = fake_generate_keyframe
    G.generate_video_from_image = fake_generate_video_from_image
    G.design_camera_movement = fake_design_camera_movement
    G.evaluate_image = fake_evaluate_image
    G.evaluate_video = fake_evaluate_video
    G.assess_character_consistency = fake_assess_character_consistency
    G.critic_final_cut = fake_critic_final_cut
    G.build_visual_context = fake_build_visual_context
    G.retrieve_rag_context = fake_retrieve_rag_context
    G.generate_reference_sheets = fake_generate_reference_sheets
    G.generate_shot_strategy = fake_generate_shot_strategy
    G.orchestrate_film_studio = fake_orchestrate_film_studio
    G.build_segmented_voiceover = fake_build_segmented_voiceover
    G.generate_voiceover = fake_generate_voiceover
    G.generate_bgm = fake_generate_bgm
    print("[dry-run] 模型边界已全部 mock（LLM/生图/视频/VLM/RAG/TTS/BGM）")

    parallel = os.environ.get("MULTIMEDIA_PARALLEL_SHOTS", "1") != "0"
    thread_id = f"dryrun_{int(time.time())}"
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 300}
    init_state = {
        "task": "赛博朋克短片：少年剑客与黑袍法师的霓虹屋顶对决",
        "auto_mode": True,
        "enable_audio": True,
        "prefer_native_audio": False,
        "quality_tier": "fast",
        "voice_role": "",
        "visual_style": "custom:mock",
        "style_description": "mock cyber style",
        "style_suffix": "cyberpunk neon, cinematic",
        "thread_id": thread_id,
        "assets_imported": False,
    }
    if args.target_duration is not None:
        init_state["target_duration"] = args.target_duration
    if args.episode is not None:
        init_state["episode"] = args.episode

    print(f"[dry-run] 开始全图运行 · shots={_WANT_SHOTS} · parallel={parallel} · thread={thread_id}")
    t1 = time.perf_counter()
    final = G.multimedia_agent.invoke(init_state, config=cfg)
    dt = time.perf_counter() - t1

    # ── 校验 ──
    scenes = final.get("scenes") or []
    urls = [str(s.get("final_video_url") or "") for s in scenes]
    ok_slots = all(f"scene_{i:02d}.mp4" in urls[i] for i in range(len(urls)))
    final_movie = final.get("final_movie_with_audio") or ""
    entries = final.get("subtitle_entries") or []
    flags = final.get("branch_results") or {}

    print("\n===== dry-run 结果 =====")
    print(f"耗时        : {dt:.1f}s（parallel={parallel}）")
    print(f"镜头槽位    : {sum(1 for u in urls if u)}/{len(urls)} · 槽位对齐={ok_slots}")
    print(f"熔断分支    : branch_results={sorted(flags)}")
    print(f"无声成片    : {final.get('final_movie_path')}")
    print(f"有声成片    : {final_movie}")
    print(f"字幕条数    : {len(entries)} · 音色: {final.get('voice_role')}")
    print(f"台账        : {json.dumps(bh.summarize_ledger(''), ensure_ascii=False)[:200]}")

    errs = []
    if final.get("aborted"):
        errs.append(f"aborted: {final.get('abort_reason')}")
    if not ok_slots:
        errs.append(f"槽位错位: {urls}")
    if not final_movie or not os.path.exists(final_movie):
        errs.append(f"有声成片缺失: {final.get('error_log')}")
    if not entries:
        errs.append("字幕为空")
    if parallel and len(flags) != len(scenes):
        errs.append(f"分支数 {len(flags)} != 镜头数 {len(scenes)}")

    if errs:
        print("\nDRY-RUN FAIL ❌")
        for e in errs:
            print("  -", e)
        return 1
    print("\nDRY-RUN OK ✅ 全链路（分镜→规划→并行生成→拼接→字幕→配音混音）零额度通过")
    print(f"成片: {final_movie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
