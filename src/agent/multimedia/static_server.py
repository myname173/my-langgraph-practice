# src/agent/multimedia/static_server.py
"""轻量产物静态服务 + 历史会话恢复接口。

LangGraph Server 只负责图执行，不托管生成产物（图片 / 视频 / 音频）。
本模块提供一个独立的 FastAPI 应用，承担两项职责：

1. **产物托管**：把项目根目录下的 `output/` 暴露为 HTTP 静态资源，
   供前端 `<img>` / `<video>` 直接加载（`/media/*`）。
2. **历史会话恢复**：挂载 `history_api.router`，直接从 sqlite 读取历史
   checkpoint，绕开平台 `getState` 对旧版图 checkpoint 的兼容性问题
   （`/history/*`，详见 history_api.py 模块文档）。

前端通过 Vite dev proxy 把 `/media/*` 与 `/history/*` 转发到本服务
（见 frontend/vite.config.ts）。

启动方式::

    python -m src.agent.multimedia.static_server
    # 或指定端口
    MEDIA_SERVER_PORT=8900 python -m src.agent.multimedia.static_server

安全说明：仅只读暴露 `output/` 目录，且对请求路径做了逃逸校验，
禁止通过 `..` 访问目录之外的文件。
"""

from __future__ import annotations

import os
import json
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .checkpointer import _resolve_db_path, build_checkpointer
from .history_api import router as history_router
from .tools import image_gen, video_gen

# 项目根目录：static_server.py -> multimedia -> agent -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "output"

# 按扩展名兜底 MIME：本地产物经此服务返回时显式声明 Content-Type，
# 防止浏览器 <video>/<img> 因推断失败而报“没有找到支持的视频格式和 mime 类型”。
_MEDIA_TYPE_BY_EXT = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".mkv": "video/x-matroska",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".json": "application/json",
}

app = FastAPI(title="Multimedia Agent Media Server", version="0.2.0")

# 前端 dev server 与本服务不同源时需要放行
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "OPTIONS", "POST"],
    allow_headers=["*"],
)


# 历史会话恢复接口（/history/*）
app.include_router(history_router)


# ── 素材库写接口：把前端标注好的即梦图写入 reference_sheets ─────────────────
class AssetsApplyPayload(BaseModel):
    thread_id: str
    reference_sheets: dict


assets_router = APIRouter(prefix="/assets", tags=["assets"])


@assets_router.post("/apply")
def apply_assets(payload: AssetsApplyPayload) -> dict:
    """把前端 AssetLibraryPanel 标注/编辑好的参考图资产写回该 thread 的
    reference_sheets，并置 assets_imported=True。续跑时 reference_gen_node
    会跳过重新生成、直接复用这些图（省额度、角色一致）。

    写回方式：读取该 thread 最新 checkpoint 的 channel_values，合并
    reference_sheets / assets_imported 后整 checkpoint 回写（LangGraph
    aget_tuple 不会自动合并 writes 表 pending writes，故必须写 channel_values）。
    """
    thread_id = payload.thread_id
    try:
        cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        saver = build_checkpointer()
        tup = saver.get_tuple(cfg)
        if tup is None or not tup.checkpoint:
            raise HTTPException(status_code=404, detail=f"thread not found: {thread_id}")
        existing_cv = dict(tup.checkpoint.get("channel_values") or {})
        existing_cv["reference_sheets"] = payload.reference_sheets
        existing_cv["assets_imported"] = True
        if not existing_cv.get("character_portrait_url"):
            chars = (payload.reference_sheets or {}).get("characters") or {}
            if chars:
                first_char = next(iter(chars.values()))
                # 兼容新结构 urls(list) 与旧结构 url(str)
                urls = first_char.get("urls") or (
                    [first_char["url"]] if first_char.get("url") else []
                )
                existing_cv["character_portrait_url"] = urls[0] if urls else None
        checkpoint = {
            "v": 2,
            "id": uuid.uuid4().hex,
            "ts": uuid.uuid1().hex,
            "parent_id": tup.checkpoint.get("id"),
            "channel_values": existing_cv,
            "channel_versions": {k: "0" for k in existing_cv},
            "versions_seen": {k: {} for k in existing_cv},
            "pending_sends": [],
        }
        metadata = dict(tup.metadata or {})
        metadata["step"] = metadata.get("step", 0)
        metadata["source"] = "asset_library_panel"
        saver.put(cfg, checkpoint, metadata, {})
        return {"ok": True, "thread_id": thread_id, "assets_imported": True}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - 把异常细节回传给前端便于排障
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"apply_assets failed: {e}") from e


# ── 素材上传接口：用户从本地上传图片作为一致性参考图 ───────────────────────
# 上传图仅作为"一致性参考"（角色/道具/场景锚点）注入后续生图/视频流程，
# 不会直接当成品视频；存盘后由 /media 静态托管返回可访问 URL。
_ALLOWED_UPLOAD_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 单文件 ≤ 10MB


@assets_router.post("/upload")
async def upload_assets(
    thread_id: str | None = None,
    story: str | None = None,
    files: list[UploadFile] = File(...),
) -> dict:
    """接收用户本地上传的素材参考图，存到 output/assets_uploaded/<thread_id>/
    （uuid 重命名防路径穿越与同名覆盖），返回可经 /media 访问的 URL 列表。

    thread_id 可选：新建任务时前端尚无 thread（由后端 startRun 创建），
    故缺省自动生成 upload_<uuid> 临时目录；上传图仅作参考 URL 引用，
    与最终 thread 解耦，提交时随 reference_sheets 注入即可。
    """
    if not files:
        raise HTTPException(status_code=400, detail="no files provided")
    if not thread_id:
        thread_id = f"upload_{uuid.uuid4().hex}"

    upload_dir = OUTPUT_DIR / "assets_uploaded" / thread_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict] = []
    for f in files:
        raw_name = (f.filename or "upload").lower()
        ext = os.path.splitext(raw_name)[1]
        if ext not in _ALLOWED_UPLOAD_EXT:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的扩展名: {ext or '(无)'}（仅允许 {sorted(_ALLOWED_UPLOAD_EXT)}）",
            )
        data = await f.read()
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大: {f.filename}（{len(data)} 字节，上限 {_MAX_UPLOAD_BYTES} 字节）",
            )
        # uuid 重命名，保留原扩展名，杜绝 `..` 穿越与同名覆盖
        stored = f"{uuid.uuid4().hex}{ext}"
        (upload_dir / stored).write_bytes(data)
        saved.append(
            {
                "filename": f.filename or stored,
                "url": f"/media/assets_uploaded/{thread_id}/{stored}",
                # 透传故事标签（如 xianxia），便于前端把素材归为对应故事，供素材库隔离与展示
                "story": story or None,
            }
        )
        print(f"[*] 素材上传: thread={thread_id} file={f.filename} -> {stored} ({len(data)}B)")
    return {"ok": True, "thread_id": thread_id, "files": saved}


# 共享参考图素材库目录：只存放用户上传/登记的参考图（角色/道具/场景）。
# 由 jimeng 生成的关键帧/尾帧必须放到 output/keyframes/<thread>/ 或 _generated/，
# 不得进入本目录，避免被前端误当作参考图素材。
_LIBRARY_DIR = OUTPUT_DIR / "reference_sheets" / "library"
_MANIFEST_PATH = _LIBRARY_DIR / "manifest.json"


class RegisterAssetPayload(BaseModel):
    """把一张已上传（/assets/upload 返回）或已有的图登记进共享参考图素材库。

    登记后该图会写入 reference_sheets/library/<category>/ 目录并记录到 manifest.json，
    之后在「创建任务 -> 已有素材库」与独立「素材库」界面中均可勾选，
    并随任务提交真正参与视频生成（作一致性参考锚点）。
    注意：jimeng 生成的关键帧/尾帧不得通过本接口登记，避免混入参考图库。
    """

    url: str  # /media/assets_uploaded/<tid>/<file> 或 /media/reference_sheets/library/<category>/<file>
    name: str  # 资产名（角色/道具/场景名），用于剧本匹配
    category: str = "character"  # character | prop | environment
    description: str = ""
    story: str = ""  # 故事标签（如 xianxia），仅归类展示，不影响锚定


@assets_router.post("/register")
async def register_asset(payload: RegisterAssetPayload) -> dict:
    """把素材图登记进共享素材库（复制文件 + 写 manifest）。"""
    if not payload.url or not payload.name.strip():
        raise HTTPException(status_code=400, detail="url 与 name 必填")

    norm = payload.url.replace("\\", "/").lstrip("/")
    if not norm.startswith("media/"):
        raise HTTPException(status_code=400, detail="仅支持 /media/ 开头的素材 url")

    # 解析源文件真实路径（复用 /media 解析逻辑：优先 OUTPUT_DIR，否则 PROJECT_ROOT）
    rel_path = norm[len("media/"):]
    root_marker = f"{PROJECT_ROOT.name}/"
    if root_marker in rel_path:
        src = (PROJECT_ROOT / rel_path.split(root_marker, 1)[1]).resolve()
        anchor = PROJECT_ROOT.resolve()
    else:
        src = (OUTPUT_DIR / rel_path).resolve()
        anchor = OUTPUT_DIR.resolve()
    try:
        src.relative_to(anchor)
    except ValueError:
        raise HTTPException(status_code=403, detail="禁止访问该路径") from None
    if not src.is_file():
        raise HTTPException(status_code=404, detail=f"源文件不存在: {payload.url}")

    # 登记到 reference_sheets/library/<category>/ 下，按资产名+分类组织，
    # 不再平铺进 keyframes/_legacy，避免与 jimeng 关键帧混放
    _LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    cat_dir = _LIBRARY_DIR / payload.category
    cat_dir.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(src.name)[1].lower() or ".png"
    # 以资产名+分类命名，重名追加后缀，杜绝覆盖
    base = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{payload.category}_{payload.name.strip()}")
    dest_name = f"{base}{ext}"
    dest = cat_dir / dest_name
    n = 1
    while dest.exists():
        dest_name = f"{base}_{n}{ext}"
        dest = cat_dir / dest_name
        n += 1
    dest.write_bytes(src.read_bytes())

    # 读写 manifest（共享素材库元数据），互斥用简单串行即可（低频写）
    manifest: list[dict] = []
    if _MANIFEST_PATH.is_file():
        try:
            manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
            if not isinstance(manifest, list):
                manifest = []
        except Exception:
            manifest = []
    manifest.append(
        {
            "filename": dest_name,
            "story": payload.story or None,
            "category": payload.category,
            "label": payload.name.strip(),
            "role_name": payload.name.strip(),
            "description": payload.description or "",
            "source": "user",
        }
    )
    _MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[*] 素材登记入库: {payload.name} -> {dest_name} ({payload.category})")
    return {
        "ok": True,
        "filename": dest_name,
        "url": f"/media/reference_sheets/library/{payload.category}/{dest_name}",
    }


class RerunFromPayload(BaseModel):
    thread_id: str
    scene_index: int
    # F-4：重跑锁定镜头时顺带把它移出 locked_scenes（否则链式扇出会跳过它）
    unlock: bool = False


@assets_router.post("/rerun_from")
def rerun_from(payload: RerunFromPayload) -> dict:
    """让用户从某个镜头重新跑：把该 thread 最新 checkpoint 的
    current_scene_index 设为目标镜头，并清空该镜头及之后所有镜头的
    生成产物，再把续跑入口挂到 advance_scene（经由 decide_next_scene
    路由自然走到 visual_context_builder，从该镜头重头制作）。

    注意：只清空“生成产物”字段，保留 scene.script（用户原始脚本），
    避免误删用户素材。回写方式与 apply_assets 一致——整 checkpoint 重写。
    """
    thread_id = payload.thread_id
    idx = int(payload.scene_index)
    # 重跑时会被节点重新生成的产物字段（保留 script / 用户输入字段）
    PRODUCT_FIELDS = (
        "image_url", "last_image_url", "raw_video_url", "final_video_url",
        "image_prompt", "last_image_prompt", "video_prompt", "action_beat",
        "critique", "video_critique", "auto_feedback", "editor_note",
        "editor_pass", "reviewer_note", "is_perfect", "last_image_is_perfect",
        "video_is_perfect", "embedding_similarity", "portrait_similarity",
        "continuity_score", "flicker_score", "audio_track", "subtitle_path",
        "bgm_path", "final_movie_with_audio",
        # F-1/F-4：本轮新增的可再生字段（候选/一致性/外科手术痕迹）随重跑一并清理
        "image_candidates", "needs_realign", "vlm_consistency",
        "fix_type", "extend_applied", "render_params",
    )
    try:
        cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        saver = build_checkpointer()
        tup = saver.get_tuple(cfg)
        if tup is None or not tup.checkpoint:
            raise HTTPException(status_code=404, detail=f"thread not found: {thread_id}")
        existing_cv = dict(tup.checkpoint.get("channel_values") or {})
        scenes = existing_cv.get("scenes") or []
        if not isinstance(scenes, list) or idx < 0 or idx >= len(scenes):
            raise HTTPException(
                status_code=400,
                detail=f"invalid scene_index={idx}, total scenes={len(scenes)}",
            )
        # 清空目标镜头及之后所有镜头的产物
        for i in range(idx, len(scenes)):
            sc = scenes[i]
            if not isinstance(sc, dict):
                continue
            for f in PRODUCT_FIELDS:
                sc.pop(f, None)
        # F-4：解锁目标镜头及其后（保持 locked_scenes 与待重跑范围一致）
        if payload.unlock:
            locked = existing_cv.get("locked_scenes")
            if isinstance(locked, list):
                existing_cv["locked_scenes"] = [i for i in locked if isinstance(i, int) and i < idx]
        existing_cv["scenes"] = scenes
        existing_cv["current_scene_index"] = idx
        # 重置成片/音频产物（因为后续镜头会被重跑、旧成片不可用）
        existing_cv["final_movie_path"] = None
        existing_cv["final_movie_with_audio"] = None
        existing_cv["error_log"] = None
        existing_cv["aborted"] = False
        existing_cv["abort_reason"] = None
        checkpoint = {
            "v": 2,
            "id": uuid.uuid4().hex,
            "ts": uuid.uuid1().hex,
            "parent_id": tup.checkpoint.get("id"),
            "channel_values": existing_cv,
            "channel_versions": {k: "0" for k in existing_cv},
            "versions_seen": {k: {} for k in existing_cv},
            "pending_sends": [],
            "next": ["advance_scene"],  # 续跑入口：路由到该镜头的 visual_context_builder
        }
        # metadata 必须继承原始并补 step：LangGraph 续跑 AsyncPregelLoop 会访问 metadata["step"]
        metadata = dict(tup.metadata or {})
        metadata["step"] = 0
        metadata["source"] = "rerun_from"
        saver.put(cfg, checkpoint, metadata, {})
        return {
            "ok": True,
            "thread_id": thread_id,
            "scene_index": idx,
            "total_scenes": len(scenes),
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - 把异常细节回传给前端便于排障
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"rerun_from failed: {e}") from e


class AbortPayload(BaseModel):
    thread_id: str
    reason: Optional[str] = None


@assets_router.post("/abort")
def abort_thread(payload: AbortPayload) -> dict:
    """中止会话（真正的服务端中止）。

    背景：此前前端 stop() 只调用 AbortController.abort()，仅断开 SSE 流，
    后端 run 仍在跑；用户随后再次启动会对同一 thread 并发跑两条 run，
    互相覆盖 state，导致产物错乱/进度回跳。

    本端点把 aborted=True 与续跑入口 next=["abort"] 一起写入 checkpoint：
      - 历史列表/详情立即呈现 aborted 状态；
      - 若之后对该 thread 发起 continue（input=null），LangGraph 会直接
        走 abort_node 终止，而不是与残留 run 并行跑两条。

    说明：无法强杀"已在运行中的进程内 run"（LangGraph 无抢占式取消），
    但通过把终态写进 checkpoint，保证不会被意外续跑出两条并发 run。
    """
    thread_id = payload.thread_id
    reason = payload.reason or "用户手动停止"
    try:
        cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        saver = build_checkpointer()
        tup = saver.get_tuple(cfg)
        if tup is None or not tup.checkpoint:
            raise HTTPException(status_code=404, detail=f"thread not found: {thread_id}")

        existing_cv = dict(tup.checkpoint.get("channel_values") or {})
        existing_cv["aborted"] = True
        existing_cv["abort_reason"] = reason
        existing_cv["error_log"] = reason
        checkpoint = {
            "v": 2,
            "id": uuid.uuid4().hex,
            "ts": uuid.uuid1().hex,
            "parent_id": tup.checkpoint.get("id"),
            "channel_values": existing_cv,
            "channel_versions": {k: "0" for k in existing_cv},
            "versions_seen": {k: {} for k in existing_cv},
            "pending_sends": [],
            "next": ["abort"],  # 续跑时直接进 abort_node，杜绝并行双跑
        }
        metadata = dict(tup.metadata or {})
        metadata["step"] = 0
        metadata["source"] = "abort"
        saver.put(cfg, checkpoint, metadata, {})
        return {"ok": True, "thread_id": thread_id, "aborted": True, "reason": reason}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - 回传细节便于排障
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"abort failed: {e}") from e


app.include_router(assets_router)


# ── P2-4 镜头配方卡接口（/recipe/*）───────────────────────────────
# 把一个 run 的“镜头配方”（首帧/尾帧/视频的 prompt、ref_strength、时长、参考图、后处理等）
# 开放给前端：查看、复制、另存、以及「微调并重跑」。纯读/纯本地，不消耗生图额度。
from .tools import recipe as _recipe  # noqa: E402


class RecipeApplyPayload(BaseModel):
    thread_id: str
    scene_index: int
    # 微调补丁，例：{"image": {"prompt": "...", "ref_strength": 0.5}, "video": {"prompt": "..."}}
    patch: dict = {}


recipe_router = APIRouter(prefix="/recipe", tags=["recipe"])


def _load_checkpoint_cv(thread_id: str):
    """读取某 thread 最新 checkpoint 的 channel_values（不存盘）。"""
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    saver = build_checkpointer()
    tup = saver.get_tuple(cfg)
    if tup is None or not tup.checkpoint:
        raise HTTPException(status_code=404, detail=f"thread not found: {thread_id}")
    return tup, dict(tup.checkpoint.get("channel_values") or {})


@recipe_router.get("/{thread_id}")
def get_recipe(thread_id: str) -> dict:
    """返回该 thread 的完整镜头配方（即时从 checkpoint 构建）+ 可读 Markdown。"""
    _tup, cv = _load_checkpoint_cv(thread_id)
    cv.setdefault("thread_id", thread_id)
    r = _recipe.build_task_recipe(cv)
    return {"ok": True, "thread_id": thread_id, "recipe": r, "markdown": _recipe.recipe_to_markdown(r)}


@recipe_router.post("/{thread_id}/save")
def save_recipe(thread_id: str) -> dict:
    """把当前配方落盘到 output/recipes/<thread>/（可分享的 JSON 文件）。"""
    _tup, cv = _load_checkpoint_cv(thread_id)
    cv.setdefault("thread_id", thread_id)
    r = _recipe.build_task_recipe(cv)
    paths = _recipe.save_task_recipe(r)
    return {"ok": True, "thread_id": thread_id, **paths}


@recipe_router.post("/{thread_id}/apply")
def apply_recipe(thread_id: str, payload: RecipeApplyPayload) -> dict:
    """微调：把配方补丁写回指定镜头并标记从该镜重跑。

    做法与 /assets/rerun_from 一致（整 checkpoint 重写 + next=["advance_scene"]），
    但：仅清空“生成产物”字段，**保留 render_params**（它正是被微调的配方），
    并把补丁里的 prompt/时长/ref_strength 写回 scene 字段，使重跑直接采用新参数。
    """
    thread_id = payload.thread_id
    idx = int(payload.scene_index)
    PRODUCT_FIELDS = (
        "image_url", "last_image_url", "raw_video_url", "final_video_url",
        "critique", "video_critique", "auto_feedback", "editor_note",
        "editor_pass", "reviewer_note", "is_perfect", "last_image_is_perfect",
        "video_is_perfect", "embedding_similarity", "portrait_similarity",
        "continuity_score", "flicker_score",
        "video_duration", "segment_start",
        "image_candidates", "needs_realign", "vlm_consistency",
        "extend_applied",
    )
    try:
        tup, cv = _load_checkpoint_cv(thread_id)
        cv.setdefault("thread_id", thread_id)
        scenes = cv.get("scenes") or []
        if not isinstance(scenes, list) or idx < 0 or idx >= len(scenes):
            raise HTTPException(status_code=400, detail=f"invalid scene_index={idx}, total={len(scenes)}")
        # 1) 基于当前状态构建配方并应用补丁
        base = _recipe.build_task_recipe(cv)
        patched = _recipe.apply_shot_patch(base, idx, payload.patch or {})
        # 2) 补丁 → scene 字段
        fields = _recipe.shot_patch_to_scene_fields(payload.patch or {})
        sc = scenes[idx]
        if not isinstance(sc, dict):
            raise HTTPException(status_code=400, detail=f"scene {idx} 非 dict")
        for i in range(idx, len(scenes)):
            s = scenes[i]
            if not isinstance(s, dict):
                continue
            for f in PRODUCT_FIELDS:
                s.pop(f, None)
        # 写入微调后的字段（保留 render_params 并合并补丁）
        merged_rp = dict(sc.get("render_params") or {})
        incoming_rp = fields.pop("render_params", {}) or {}
        for sect, val in incoming_rp.items():
            if isinstance(val, dict):
                merged_rp[sect] = {**(merged_rp.get(sect) or {}), **val}
            else:
                merged_rp[sect] = val
        sc.update(fields)
        sc["render_params"] = merged_rp
        cv["scenes"] = scenes
        cv["current_scene_index"] = idx
        cv["final_movie_path"] = None
        cv["final_movie_with_audio"] = None
        cv["error_log"] = None
        cv["aborted"] = False
        cv["abort_reason"] = None
        checkpoint = {
            "v": 2,
            "id": uuid.uuid4().hex,
            "ts": uuid.uuid1().hex,
            "parent_id": tup.checkpoint.get("id"),
            "channel_values": cv,
            "channel_versions": {k: "0" for k in cv},
            "versions_seen": {k: {} for k in cv},
            "pending_sends": [],
            "next": ["advance_scene"],
        }
        metadata = dict(tup.metadata or {})
        metadata["step"] = 0
        metadata["source"] = "recipe_apply"
        saver = build_checkpointer()
        cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        saver.put(cfg, checkpoint, metadata, {})
        # 3) 同步落盘微调后的配方，便于追溯
        try:
            _recipe.save_task_recipe(patched)
        except Exception as _e:  # noqa: BLE001
            print(f"[WARN] recipe apply 落盘失败(不阻断): {_e}")
        return {
            "ok": True,
            "thread_id": thread_id,
            "scene_index": idx,
            "applied_fields": sorted(fields.keys()),
            "recipe": patched,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"recipe apply failed: {e}") from e


app.include_router(recipe_router)

# ── 用户设置接口（S1）：前端「设置」面板读写 API Key / 模型 / 后端 / 档位 ────────
#
# 跨进程生效：设置持久化到 data/user_settings.json（已在 .gitignore 的 data/ 下）；
# 任何进程的 config_loader 在导入与保存时都会 apply_user_settings() 把文件合并进
# 自己的 os.environ，因此 LangGraph server / Streamlit / 本服务无需重启即可生效。

from .tools import backend_health as _backend_health
from .config_loader import (
    apply_user_settings as _apply_user_settings,
    load_user_settings as _load_user_settings,
    mask_secret as _mask_secret,
    save_user_settings as _save_user_settings,
)
from .tools.video_gen import _probe_dashscope_video as _probe_dashscope_video
from .tools.image_gen import _probe_dashscope_image as _probe_dashscope_image
from .tools.image_gen import _probe_siliconflow_image as _probe_siliconflow_image

# 允许经设置面板写入的键白名单（防止任意环境变量注入）
SETTINGS_ALLOWED_KEYS = {
    # 凭证
    "DASHSCOPE_API_KEY", "AGNES_API_KEY", "ZHIPU_API_KEY", "JIMENG_SESSION_ID",
    "GEMINI_API_KEY", "SILICONFLOW_API_KEY",
    "OPENAI_API_KEY",          # 文本 LLM（OpenAI 兼容模式，默认即 DashScope）
    # 后端选择 / 档位
    "VIDEO_PROVIDER", "MULTIMODIA_BACKEND", "QUALITY_TIER",
    # 模型覆盖（视频 / 生图）
    "AGNES_VIDEO_MODEL", "ZHIPU_VIDEO_MODEL", "DASHSCOPE_VIDEO_MODEL",
    "GEMINI_VIDEO_MODEL", "GEMINI_IMAGE_MODEL", "SILICONFLOW_IMAGE_MODEL",
    "DASHSCOPE_IMAGE_MODEL",
    # 模型覆盖（文本 LLM / 接口地址）
    "DASHSCOPE_TEXT_MODEL", "SILICONFLOW_TEXT_MODEL", "OPENAI_BASE_URL",
    # 语音（默认音色覆盖；字幕为规则生成，无需模型）
    "TTS_VOICE",
    # 生成参数
    "VIDEO_DURATION", "JIMENG_IMAGE_RESOLUTION", "AGNES_MIN_SUBMIT_GAP",
}

_SETTINGS_SECRET_KEYS = {
    "DASHSCOPE_API_KEY", "AGNES_API_KEY", "ZHIPU_API_KEY", "JIMENG_SESSION_ID",
    "GEMINI_API_KEY", "SILICONFLOW_API_KEY", "OPENAI_API_KEY",
}

settings_router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsPayload(BaseModel):
    values: dict


@settings_router.get("")
async def get_settings() -> dict:
    """当前设置（敏感键打码，标注来源 file/env）+ 各后端熔断健康快照。"""
    file_vals = _load_user_settings()
    merged: dict = {}
    for k in sorted(SETTINGS_ALLOWED_KEYS):
        current = os.environ.get(k, "")
        from_file = file_vals.get(k, "")
        shown = from_file if from_file else current
        merged[k] = _mask_secret(str(shown)) if k in _SETTINGS_SECRET_KEYS else str(shown)
        merged[k + "__source"] = "file" if from_file else ("env" if current else "")
    return {"settings": merged, "backend_health": _backend_health.snapshot()}


@settings_router.post("")
async def post_settings(req: SettingsPayload) -> dict:
    values = {
        k: str(v).strip()
        for k, v in (req.values or {}).items()
        if k in SETTINGS_ALLOWED_KEYS
    }
    if not values:
        raise HTTPException(status_code=400, detail="没有可写入的设置键（白名单外的键被拒绝）")
    _save_user_settings(values)
    return {
        "ok": True,
        "applied_keys": sorted(values.keys()),
        "backend_health": _backend_health.snapshot(),
        "note": "设置已写入 data/user_settings.json 并在本进程生效；其他进程"
                "（LangGraph server / Streamlit）在下一次生成调用时自动加载。",
    }


class ProbePayload(BaseModel):
    kind: str                  # "dashscope-video"（后续可扩展 openai-models 等）
    model: str = ""
    api_key: str = ""          # 可选；留空则用当前已保存/环境中的 Key


@settings_router.post("/probe")
async def settings_probe(req: ProbePayload) -> dict:
    """零额度探测：验证「模型名 + API Key」是否可用（不创建任务、不消耗额度）。

    探测失败时返回 ok=false + 具体原因（Key 无效 / 模型名无效 / 额度受限 / 网络错误），
    由设置面板直接展示给用户。
    """
    _apply_user_settings()
    model = (req.model or "").strip()
    if req.kind == "dashscope-image":
        # P2 后续：生图后端自助接入 —— 用户填 Key/模型后先连通测试，通了才参与 fallback
        api_key = (req.api_key or "").strip() or \
            (os.getenv("DASHSCOPE_API_KEY") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="未提供 DASHSCOPE_API_KEY")
        if not model:
            model = "qwen-image-3.0"
        ok, detail = _probe_dashscope_image(api_key, model)
        return {"ok": ok, "kind": req.kind, "model": model, "detail": detail}
    if req.kind == "siliconflow-image":
        api_key = (req.api_key or "").strip() or \
            (os.getenv("SILICONFLOW_API_KEY") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="未提供 SILICONFLOW_API_KEY")
        ok, detail = _probe_siliconflow_image(api_key, model)
        return {"ok": ok, "kind": req.kind, "model": model, "detail": detail}
    if req.kind == "dashscope-video":
        if not model:
            raise HTTPException(status_code=400, detail="未提供模型名")
        api_key = (req.api_key or "").strip() or \
            (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="未提供 DASHSCOPE_API_KEY（请在上方密钥区填写）")
        ok, detail = _probe_dashscope_video(api_key, model)
        return {"ok": ok, "kind": req.kind, "model": model, "detail": detail}
    raise HTTPException(status_code=400, detail=f"暂不支持探测类型: {req.kind}")


@settings_router.get("/health")
async def settings_health() -> dict:
    return {"backend_health": _backend_health.snapshot()}


@settings_router.get("/cost/{thread_id}")
async def settings_cost(thread_id: str) -> dict:
    return _backend_health.summarize_ledger(thread_id)


app.include_router(settings_router)



@app.get("/health")
def health() -> dict:
    """健康检查，回报产物目录与历史存档库状态，便于排障。"""
    try:
        db_path = _resolve_db_path()
        db_info: dict = {
            "checkpoint_db": str(db_path),
            "checkpoint_db_exists": db_path.is_file(),
            "checkpoint_db_size_mb": (
                round(db_path.stat().st_size / 1024 / 1024, 1)
                if db_path.is_file()
                else 0
            ),
        }
    except Exception as exc:  # noqa: BLE001 - 健康检查不允许抛错
        db_info = {"checkpoint_db_error": f"{type(exc).__name__}: {exc}"}

    return {
        "status": "ok",
        "output_dir": str(OUTPUT_DIR),
        "exists": OUTPUT_DIR.is_dir(),
        **db_info,
    }


# 目录列举仅允许返回的产物扩展名（避免把 sqlite / 中间文件暴露给前端）。
# 同时包含图片扩展名：前端 AssetLibraryPanel 需要列举 keyframes/<threadId>
# 下的即梦图（png/jpg）供用户标注参考图素材，故一并放开。
_LISTABLE_EXT = {
    ".mp4", ".webm", ".mov", ".m4v", ".mkv",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
}


@app.get("/media/list")
def list_media(dir: str = "clips") -> dict:
    """列举 `output/<dir>/` 下的视频文件，供前端本地片段画廊直接消费。

    仅只读扫描 `output/` 之内、限定视频扩展名，不返回文件内容。
    返回的 `path` 为相对 `output/` 的路径，前端直接拼成 `/media/<path>` 即可访问。

    Args:
        dir: 相对 `output/` 的子目录名，默认 `clips`（如 `clips`、`reference_sheets`）。

    Returns:
        { ok: true, dir: str, count: int, items: [{ name, path, size }] }
    """
    # 允许多级子目录（如 clips/<thread_id>），但仍禁止越界与特殊段
    if not dir or dir in (".", "..") or dir.startswith("/") or dir.startswith("\\"):
        raise HTTPException(status_code=400, detail="非法的目录参数")
    if ".." in dir.replace("\\", "/").split("/"):
        raise HTTPException(status_code=400, detail="非法的目录参数")

    target_dir = (OUTPUT_DIR / dir).resolve()
    anchor = OUTPUT_DIR.resolve()
    try:
        target_dir.relative_to(anchor)
    except ValueError:
        raise HTTPException(status_code=403, detail="禁止访问该路径") from None

    items: list[dict] = []
    if target_dir.is_dir():
        for p in sorted(target_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in _LISTABLE_EXT:
                items.append(
                    {
                        "name": p.name,
                        "path": f"{dir}/{p.name}",
                        "size": p.stat().st_size,
                    }
                )

    return {"ok": True, "dir": dir, "count": len(items), "items": items}


@app.get("/media/{file_path:path}")
def get_media(file_path: str) -> FileResponse:
    """按相对路径返回 `output/` 下的产物文件。

    Args:
        file_path: 相对于 `output/` 的路径，支持多级子目录（如 `clips/xxx.mp4`）。

    Raises:
        HTTPException: 路径越界（403）或文件不存在（404）。
    """
    if not file_path:
        raise HTTPException(status_code=404, detail="未指定文件")

    # 支持“项目内绝对路径”：前端会把 C:\proj\video.mp4 归一化为
    # /media/Users/.../my-langgraph-practice-main/video.mp4。
    # 这里识别项目根目录名，截取其后作为相对于 PROJECT_ROOT 的路径，
    # 以便访问 output/ 之外的产物（如仓库根的成片 video_*.mp4）。
    norm = file_path.replace("\\", "/")
    root_marker = f"{PROJECT_ROOT.name}/"
    if root_marker in norm:
        rel = norm.split(root_marker, 1)[1]
        target = (PROJECT_ROOT / rel).resolve()
        anchor = PROJECT_ROOT.resolve()
    else:
        target = (OUTPUT_DIR / file_path).resolve()
        anchor = OUTPUT_DIR.resolve()

    # 目录逃逸防护：解析后的真实路径必须仍在对应根目录之内
    try:
        target.relative_to(anchor)
    except ValueError:
        raise HTTPException(status_code=403, detail="禁止访问该路径") from None

    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    # 【修复】显式指定 media_type，避免 FastAPI 在个别环境下推断失败，
    # 导致前端 <video> 报“没有找到支持的视频格式和 mime 类型”。
    media_type = _MEDIA_TYPE_BY_EXT.get(target.suffix.lower())
    # 视频需要 Range 支持才能拖动进度条，FileResponse 已内建处理
    return FileResponse(target, media_type=media_type)


# ────────────────────────────────────────────────────────────
# 快速生成接口：前端文字直接生成视频 / 图片
# ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    type: str = "video"  # "video" | "image"


def _download_remote(url: str, dest: Path) -> bool:
    """下载远程 URL 到本地 dest；失败返回 False（调用方可回退到原始 URL）。"""
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        return dest.is_file() and dest.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] 下载远程产物失败，回退原始 URL: {exc}")
        return False


def _ext_from_url(url: str, default: str) -> str:
    """从 URL 推断扩展名（去掉查询串）。"""
    path = url.split("?")[0]
    ext = os.path.splitext(path)[1].lower()
    return ext or default


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    """文字直接生成单条视频或图片。

    - type=video: 调用 video_gen.generate_video_from_image（默认走 t2v 有额度链）
    - type=image : 调用 image_gen.generate_image_to_url（qwen-image 有额度）
    产物下载落盘 output/quickgen/，返回 {url: "/media/quickgen/<file>"}。
    额度耗尽时返回 402 + 友好消息，前端据实提示、不重试。
    """
    prompt = (req.prompt or "").strip()
    gtype = (req.type or "video").strip().lower()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt 不能为空")
    if gtype not in ("video", "image"):
        raise HTTPException(status_code=400, detail="type 必须为 video 或 image")

    try:
        if gtype == "video":
            print(f"[quickgen] 文字生成视频: {prompt[:40]}...")
            remote_url = video_gen.generate_video_from_image("", prompt)
        else:
            print(f"[quickgen] 文字生成图片: {prompt[:40]}...")
            remote_url = image_gen.generate_image_to_url(prompt)
    except video_gen.FreeTierQuotaExhaustedError as e:
        raise HTTPException(status_code=402, detail=f"免费额度已耗尽：{e}")
    except Exception as e:  # noqa: BLE001
        err_text = str(e)
        # 兼容 image_gen 的额度/模型错误判断
        if image_gen._is_quota_or_model_error(err_text):
            raise HTTPException(status_code=402, detail=f"免费额度已耗尽：{err_text}")
        raise HTTPException(status_code=500, detail=f"生成失败：{err_text[:300]}")

    # 落盘 output/quickgen/
    quickgen_dir = OUTPUT_DIR / "quickgen"
    quickgen_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    default_ext = ".mp4" if gtype == "video" else ".png"
    ext = _ext_from_url(remote_url, default_ext)
    filename = f"{gtype}_{ts}_{suffix}{ext}"
    dest = quickgen_dir / filename

    if _download_remote(remote_url, dest):
        rel = f"quickgen/{filename}"
        return {"url": f"/media/{rel}", "type": gtype}

    # 下载失败：直接返回远程 URL（带签名时效，预览可能过期）
    return {"url": remote_url, "type": gtype, "remote": True}


def main() -> None:
    """以独立进程启动静态服务。"""
    import uvicorn

    port = int(os.getenv("MEDIA_SERVER_PORT", "8900"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[media-server] 托管目录: {OUTPUT_DIR}")
    print(f"[media-server] 产物地址: http://127.0.0.1:{port}/media/<filename>")
    print(f"[media-server] 历史会话: http://127.0.0.1:{port}/history/threads")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
