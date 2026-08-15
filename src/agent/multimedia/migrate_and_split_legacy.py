# src/agent/multimedia/migrate_and_split_legacy.py
"""
一次性脚本：迁移历史即梦产物 + 拆成两个独立历史 thread。

背景
----
之前两天用即梦免费积分跑过生图（赛博朋克女主 / 海鸥 两个故事），产物落在
src/agent/multimedia/output/{keyframes,clips}/，文件名只是时间戳、无角色/功能元数据，
且全部误拼进同一个 ``jimeng_history_20260813`` thread，前端点开空白或视频串味。

本脚本做三件事：
1. 幂等迁移：把 src/agent/multimedia/output/{keyframes,clips}/* 移到
   <root>/output/{keyframes,clips}/_legacy/（按扩展名过滤，已存在则跳过，不删源目录）。
2. 写 manifest：output/keyframes/_legacy/manifest.json，记录 8 张图的
   filename / story / category(角色|武器|场景) / label / role_name / description
   （由 AI 识图结果直接填充）。
3. 重拼装：删除旧 jimeng_history_20260813，改用 AsyncSqliteSaver.aput 写两个新 thread：
   - jimeng_cyber_20260813（赛博朋克：女主×2、手枪、无人机）
   - jimeng_gull_20260813（海鸥：海鸥×2、海边日落、海鸥飞翔）
   每个 thread 的 channel_values 含 scenes（首帧/尾帧指向对应图 /media/...）、
   reference_sheets（按识图结果预填 characters/props/environments）、
   assets_imported=True、character_portrait_url、task（侧栏标题）。

运行：python src/agent/multimedia/migrate_and_split_legacy.py
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 让脚本可独立运行（python src/agent/multimedia/migrate_and_split_legacy.py），
# 直接以 multimedia 目录为导入根，复用 checkpointer 的 DB 路径解析。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpointer  # noqa: E402

_resolve_db_path = checkpointer._resolve_db_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_OUTPUT = PROJECT_ROOT / "src" / "agent" / "multimedia" / "output"
DST_ROOT = PROJECT_ROOT / "output"

OLD_THREAD = "jimeng_history_20260813"
CYBER_THREAD = "jimeng_cyber_20260813"
GULL_THREAD = "jimeng_gull_20260813"

# ── 识图结果（AI 已逐张识别）───────────────────────────────────────────────
MANIFEST = [
    {"filename": "jimeng_kf_1786534486882.png", "story": "cyber", "category": "character",
     "label": "女主（赛博朋克）", "role_name": "女主",
     "description": "赛博朋克女角色，白短发、绿光眼、黑长风衣、持枪，夜晚科幻都市"},
    {"filename": "jimeng_kf_1786534594239.png", "story": "cyber", "category": "character",
     "label": "女主·三视图", "role_name": "女主",
     "description": "同一角色正/侧/背三视图设定图"},
    {"filename": "jimeng_kf_1786534629056.png", "story": "cyber", "category": "prop",
     "label": "能量手枪", "role_name": "手枪",
     "description": "科幻能量手枪，黑色枪身、橙色能量枪管"},
    {"filename": "jimeng_kf_1786534674987.png", "story": "cyber", "category": "prop",
     "label": "无人机", "role_name": "无人机",
     "description": "八角形科幻无人机/飞行器，带橙色光环与雷电爆炸特效"},
    {"filename": "jimeng_kf_1786551725263.png", "story": "gull", "category": "character",
     "label": "海鸥（角色）", "role_name": "海鸥",
     "description": "白羽毛、黄喙、展翅飞翔的海鸥角色"},
    {"filename": "jimeng_kf_1786551814612.png", "story": "gull", "category": "character",
     "label": "海鸥·三视图", "role_name": "海鸥",
     "description": "海鸥卡通风格三视图设定图"},
    {"filename": "jimeng_kf_1786551863550.png", "story": "gull", "category": "environment",
     "label": "海边日落", "role_name": "海边",
     "description": "海边日落礁石场景，苔藓礁石与海面"},
    {"filename": "jimeng_kf_1786552167137.png", "story": "gull", "category": "environment",
     "label": "海鸥飞翔·场景", "role_name": "海边",
     "description": "海鸥飞翔于夕阳海面的关键帧/场景图"},
]

MEDIA_PREFIX = "/media/keyframes/_legacy/"


def _url(name: str) -> str:
    return f"{MEDIA_PREFIX}{name}"


def _migrate_once() -> None:
    """幂等迁移历史产物到 _legacy（按扩展名过滤）。"""
    for kind in ("keyframes", "clips"):
        src = SRC_OUTPUT / kind
        if not src.exists():
            continue
        dst = DST_ROOT / kind / "_legacy"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".mp4", ".webm", ".gif"):
                continue
            target = dst / f.name
            if target.exists():
                continue  # 幂等：已迁移则跳过
            try:
                shutil.move(str(f), str(target))
                print(f"    [MV] {f.name} -> {kind}/_legacy/")
            except Exception as e:  # noqa: BLE001 - 失败跳过不中断
                print(f"    [WARN] 迁移失败 {f.name}: {e}")


def _write_manifest() -> None:
    dst = DST_ROOT / "keyframes" / "_legacy" / "manifest.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [OK] manifest.json 写入 {dst.name}（{len(MANIFEST)} 张图）")


def _build_reference_sheets(story: str) -> dict:
    """按识图结果预填 reference_sheets（与 reference_gen._persist_reference_image 同 schema）。

    同一 role_name 的多张图（如"女主·三视图"）累积进 urls 列表，实现"资产下多参考图"，
    供续跑时多图融合注入（对齐 Vidu）。
    """
    chars, props, envs = {}, {}, {}

    def _acc(bucket: dict, name: str, url: str, description: str):
        if name in bucket:
            bucket[name]["urls"].append(url)
        else:
            bucket[name] = {"urls": [url], "description": description, "name": name}

    for m in MANIFEST:
        if m["story"] != story:
            continue
        url = _url(m["filename"])
        name = m["role_name"]
        if m["category"] == "character":
            _acc(chars, name, url, m["description"])
        elif m["category"] == "prop":
            _acc(props, name, url, m["description"])
        else:
            _acc(envs, name, url, m["description"])
    return {"characters": chars, "props": props, "environments": envs}


def _build_scenes(story: str) -> list:
    """构造展示用 scenes：首帧/尾帧指向对应图，覆该故事全部图。"""
    items = [m for m in MANIFEST if m["story"] == story]
    # 角色图优先作首帧，非角色图作尾帧
    chars = [m for m in items if m["category"] == "character"]
    others = [m for m in items if m["category"] != "character"]
    scenes = []
    for i, ch in enumerate(chars):
        tail = others[i] if i < len(others) else (others[-1] if others else ch)
        scenes.append({
            "scene_index": i,
            "image_url": _url(ch["filename"]),
            "last_image_url": _url(tail["filename"]),
            "title": ch["label"],
            "video_url": None,
            "final_clip_url": None,
            "video_prompt": "",
            "critique": None,
            "status": "done",
        })
    # 剩余非角色图若未配对，单独成 scene
    if len(others) > len(chars):
        for extra in others[len(chars):]:
            scenes.append({
                "scene_index": len(scenes),
                "image_url": _url(extra["filename"]),
                "last_image_url": _url(extra["filename"]),
                "title": extra["label"],
                "video_url": None,
                "final_clip_url": None,
                "video_prompt": "",
                "critique": None,
                "status": "done",
            })
    return scenes


def _make_checkpoint(channel_values: dict) -> dict:
    """构造一个最小可用的 LangGraph checkpoint dict（aput 会负责序列化）。"""
    cv = dict(channel_values)
    ts = uuid.uuid1().hex  # UUIDv1 时间序，便于列表按时间排序
    return {
        "id": uuid.uuid4().hex,
        "ts": ts,
        "parent_id": None,
        "channel_values": cv,
        "channel_versions": {k: "v1" for k in cv},
        "versions_seen": {k: {"v1"} for k in cv},
    }


def _thread_values(thread_id: str, task: str, story: str) -> dict:
    sheets = _build_reference_sheets(story)
    first_char = next(iter(sheets.get("characters", {}).values()), {})
    portrait = (first_char.get("urls") or [None])[0]
    return {
        "thread_id": thread_id,
        "task": task,
        "scenes": _build_scenes(story),
        "reference_sheets": sheets,
        "character_portrait_url": portrait,
        "assets_imported": True,
        "current_scene_index": 0,
        "status": "done",
    }


def _delete_thread(saver, thread_id: str) -> None:
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    found = False
    for _t in saver.list(cfg, limit=1000):
        found = True
        break
    if found:
        saver.delete_thread(thread_id)
        print(f"    [DEL] 旧 thread {thread_id} 已清理")


def _put_thread(saver, thread_id: str, task: str, story: str) -> None:
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint = _make_checkpoint(_thread_values(thread_id, task, story))
    metadata = {"thread_id": thread_id, "story": story, "source": "migrate_and_split_legacy"}
    saver.put(cfg, checkpoint, metadata, {})
    print(f"    [OK] 拼装 thread {thread_id}（{task}）")


def main() -> None:
    print("== 1/3 迁移历史产物到 _legacy ==")
    _migrate_once()
    print("== 2/3 写 manifest ==")
    _write_manifest()
    print("== 3/3 重拼装两个独立 thread ==")
    saver = checkpointer.build_checkpointer()  # 同步 SqliteSaver
    _delete_thread(saver, OLD_THREAD)
    _put_thread(saver, CYBER_THREAD, "即梦历史·赛博朋克女主（参考图已生成）", "cyber")
    _put_thread(saver, GULL_THREAD, "即梦历史·海鸥故事（参考图已生成）", "gull")
    print("完成。可在前端侧栏看到两个新历史会话，点开即见素材，可继续用参考图续跑。")


if __name__ == "__main__":
    main()
