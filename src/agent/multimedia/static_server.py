# src/agent/multimedia/static_server.py
"""轻量产物静态服务。

LangGraph Server 只负责图执行，不托管生成产物（图片 / 视频 / 音频）。
本模块提供一个独立的 FastAPI 应用，把项目根目录下的 `output/` 暴露为
HTTP 静态资源，供前端 `<img>` / `<video>` 直接加载。

前端通过 Vite dev proxy 把 `/media/*` 转发到本服务（见 frontend/vite.config.ts）。

启动方式::

    python -m src.agent.multimedia.static_server
    # 或指定端口
    MEDIA_SERVER_PORT=8900 python -m src.agent.multimedia.static_server

安全说明：仅只读暴露 `output/` 目录，且对请求路径做了逃逸校验，
禁止通过 `..` 访问目录之外的文件。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# 项目根目录：static_server.py -> multimedia -> agent -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "output"

app = FastAPI(title="Multimedia Agent Media Server", version="0.1.0")

# 前端 dev server 与本服务不同源时需要放行
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """健康检查，同时回报产物目录状态。"""
    return {
        "status": "ok",
        "output_dir": str(OUTPUT_DIR),
        "exists": OUTPUT_DIR.is_dir(),
    }


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

    target = (OUTPUT_DIR / file_path).resolve()

    # 目录逃逸防护：解析后的真实路径必须仍在 OUTPUT_DIR 之内
    try:
        target.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="禁止访问该路径") from None

    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    # 视频需要 Range 支持才能拖动进度条，FileResponse 已内建处理
    return FileResponse(target)


def main() -> None:
    """以独立进程启动静态服务。"""
    import uvicorn

    port = int(os.getenv("MEDIA_SERVER_PORT", "8900"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[media-server] 托管目录: {OUTPUT_DIR}")
    print(f"[media-server] 访问地址: http://127.0.0.1:{port}/media/<filename>")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
