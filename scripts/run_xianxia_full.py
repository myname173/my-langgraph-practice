#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后端复刻前端 TaskLauncher 的完整流程（仙侠三视图角色完整流程测试）。

精确复刻前端行为：
  - frontend/src/components/TaskLauncher.tsx
      uploadViaFormData()  -> POST /assets/upload
      buildSheets()        -> reference_sheets (key 须与剧本称呼一致)
      submit()             -> 构造 RunInput {reference_sheets, assets_imported}
  - frontend/src/hooks/useAgentStream.ts
      startRun()           -> client.threads.create() + consume({input})
      consume()            -> client.runs.stream(tid, ASSISTANT_ID, {...streamMode})

运行：python scripts/run_xianxia_full.py
会在遇到审核中断（interrupt）时停机并打印 thread_id，用户可：
  A) 在前端打开该 thread 点 approve（复刻路径），或
  B) 再跑一次 python scripts/resume_xianxia.py <thread_id> 由后端续跑。
"""
import os
import sys
import re
import json
import time
import uuid
import requests
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "workspace" / "assets"
MEDIA_BASE = "http://localhost:8900"          # /assets/upload 所在服务
LG_BASE = "http://localhost:2024"              # LangGraph 图服务
ASSISTANT_ID = "fe096781-5601-53d2-b2f6-0d3403f7e9ca"

# 剧本里对角色/道具/场景的称呼 —— 必须与 reference_sheets 的 key 完全一致
TASK = """制作一段约30秒的电影级仙侠奇幻冒险短片。整体风格为国风仙侠 CG，类似《诛仙》与《仙剑奇侠传》结合的恢弘写实质感，强调青金色调、灵气光粒、浮空巨构与云海。主角「主角」是一位白衣剑修，束高马尾，气质清冷，手持青光飞剑「主角飞剑」；反派一号「反派一号」是黑袍魔修，手持缠绕锁链的魔剑「反派一号魔剑」；反派二号「反派二号」是红衣刀客，手持血痕长刀「反派二号血刃」。核心视觉：悬浮于云海的「青云浮空山门」巨构——玉金材质的层叠山门、飞瀑、发光符文拱桥。必须包含以下镜头：1. 全景：云海中「青云浮空山门」巍峨浮现，符文流转；2. 中景：「主角」踏剑立于山门之巅，衣袂翻飞；3. 动作：「反派一号」自暗云中降临，魔剑出鞘，锁链横扫；4. 对峙：「反派二号」血刃突进，「主角」拔剑格挡，剑光炸裂；5. 高潮：三人在浮空拱桥上混战，灵气与魔气对撞；6. 结尾：「主角」一剑破阵，反派败退，山门在晨光中归于宁静。整体节奏：恢弘 → 蓄势 → 冲突 → 爆发 → 高潮 → 留白收尾，电影级仙侠质感。"""

# 本地文件 -> (上传后在 sheets 里的 key, 分类, 描述)
# 分类: character / prop / environment
ASSET_MAP = [
    ("仙侠主角三视图.jpg", "主角", "character", "仙侠主角，白衣剑修，束高马尾，气质清冷，手持青光飞剑"),
    ("仙侠反派一号三视图.jpg", "反派一号", "character", "仙侠反派一号，黑袍魔修，手持缠绕锁链的魔剑"),
    ("仙侠反派二号三视图.jpg", "反派二号", "character", "仙侠反派二号，红衣刀客，手持血痕长刀"),
    ("主角飞剑道具.png", "主角飞剑", "prop", "主角所用飞剑，玉绿剑穗，青锋流转"),
    ("反派一号魔剑道具.png", "反派一号魔剑", "prop", "反派一号魔剑，黑铁锁链缠绕，血色符文"),
    ("反派二号血刃道具.png", "反派二号血刃", "prop", "反派二号血刃，刀身血痕，粗麻缠柄"),
    ("仙侠巨构浮空山门全景.png", "青云浮空山门", "environment", "悬浮云海的仙侠巨构山门，玉金材质，符文拱桥，飞瀑"),
]

CAT_TO_SHEET = {"character": "characters", "prop": "props", "environment": "environments"}


def upload_assets() -> dict:
    """复刻 uploadViaFormData(): POST /assets/upload（无 thread_id 自动建临时目录）。"""
    files = []
    paths = []
    for fname, *_ in ASSET_MAP:
        p = ASSET_DIR / fname
        if not p.is_file():
            raise FileNotFoundError(f"资产缺失: {p}")
        paths.append(p)
    # 用同一个临时 thread 目录，便于在 8900 上集中管理本次上传
    tmp_tid = f"upload_{uuid.uuid4().hex}"
    for p in paths:
        files.append(("files", (p.name, p.read_bytes(), "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png")))
    r = requests.post(f"{MEDIA_BASE}/assets/upload", params={"thread_id": tmp_tid}, files=files, timeout=60)
    r.raise_for_status()
    data = r.json()
    print(f"[upload] thread={data['thread_id']} 上传 {len(data['files'])} 张")
    # 文件名(去uuid前缀) -> url
    url_by_name = {}
    for f in data["files"]:
        base = re.sub(r"\.[^.]+$", "", f["filename"])
        url_by_name[base] = f["url"]
    return tmp_tid, url_by_name


def build_sheets(url_by_name: dict) -> dict:
    """复刻 buildSheets(): key 用 ASSET_MAP 里的语义名，description 写清语义。"""
    sheets = {"characters": {}, "props": {}, "environments": {}}
    for fname, key, cat, desc in ASSET_MAP:
        base = re.sub(r"\.[^.]+$", "", fname)
        url = url_by_name.get(base)
        if not url:
            raise RuntimeError(f"上传未返回 {fname} 的 url")
        sheet_key = CAT_TO_SHEET[cat]
        sheets[sheet_key][key] = {"urls": [url], "description": desc, "name": key}
    return sheets


async def start_run(sheets: dict) -> str:
    """复刻 startRun(): client.threads.create + consume({input})。"""
    from langgraph_sdk import get_client
    client = get_client(url=LG_BASE, api_key="")

    # 复刻 RunInput（与 frontend types.ts RunInput 对齐；scenes=[] 让后端 showrunner 自生成）
    run_input = {
        "task": TASK,
        "global_setting": "",
        "scenes": [],
        "current_scene_index": 0,
        "use_first_last_frame": True,
        "enable_audio": True,
        "prefer_native_audio": False,
        "voice_role": "yunxi",
        "bgm_mood": "tense",
        "reference_images": [],
        "reference_embeddings": [],
        "reference_sheets": sheets,
        "assets_imported": True,
        "aborted": False,
        "final_movie_path": None,
        "error_log": None,
        "rewrite_count": 0,
    }

    thread = await client.threads.create()
    tid = thread["thread_id"]
    print(f"[start] thread_id={tid}")

    # 复刻 consume(): runs.stream(tid, ASSISTANT_ID, {input, streamMode})
    print("[stream] 开始消费（values/updates），遇到中断即停机...")
    interrupt_payload = None
    async for chunk in client.runs.stream(
        tid, ASSISTANT_ID,
        input=run_input,
        stream_mode=["values", "updates"],
    ):
        if chunk.event == "updates":
            for node, out in (chunk.data or {}).items():
                if isinstance(out, dict) and "__interrupt__" in out:
                    interrupt_payload = out["__interrupt__"]
                    print(f"[interrupt] 节点 {node} 触发审核中断")
        elif chunk.event == "values":
            # 打印关键进度
            vals = chunk.data or {}
            node = vals.get("__node__") if isinstance(vals, dict) else None
            if node:
                print(f"  -> 节点: {node}")
        elif chunk.event == "error":
            print(f"[error] {chunk.data}")
            break
    if interrupt_payload is not None:
        print("\n=== 已触发审核中断，等待处理 ===")
        print(f"THREAD_ID={tid}")
        print("可在前端 http://localhost:5173 打开该会话点 approve，")
        print("或执行: python scripts/resume_xianxia.py " + tid)
        # 持久化，便于 resume 脚本读取
        (ROOT / "workspace" / "_xianxia_thread.json").write_text(json.dumps({
            "thread_id": tid, "task": TASK, "run_input": run_input
        }, ensure_ascii=False), encoding="utf-8")
        return tid
    print(f"[done] 流程跑完，未中断。thread_id={tid}")
    return tid


def main():
    tmp_tid, url_by_name = upload_assets()
    sheets = build_sheets(url_by_name)
    print(f"[sheets] characters={list(sheets['characters'])} props={list(sheets['props'])} envs={list(sheets['environments'])}")
    import asyncio
    asyncio.run(start_run(sheets))


if __name__ == "__main__":
    main()
