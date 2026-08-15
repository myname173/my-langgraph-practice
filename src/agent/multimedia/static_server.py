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
import time
import uuid
from pathlib import Path

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
            }
        )
        print(f"[*] 素材上传: thread={thread_id} file={f.filename} -> {stored} ({len(data)}B)")
    return {"ok": True, "thread_id": thread_id, "files": saved}


class RerunFromPayload(BaseModel):
    thread_id: str
    scene_index: int


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


app.include_router(assets_router)


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
