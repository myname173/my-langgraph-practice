# -*- coding: utf-8 -*-
"""P1-4 前置：基于 mediapipe 的人脸对齐（摆正 + 裁剪）。

用途
----
角色一致性判断（P1-4）比较「当前帧 vs 参照帧」时，若两帧里角色的**头部朝向/倾斜**
差异较大，VLM 容易被姿态干扰而误判漂移。本模块用 mediapipe FaceLandmarker（478 关键点）
先做几何归一化，再交给视觉模型比较，降低姿态噪声。

设计要点
--------
- **零硬依赖**：mediapipe 缺失 / 无脸 / 任何异常 → ``available=False``，调用方回退原图，
  绝不阻断主流程。
- **模型自动缓存**：``.task`` 模型首次使用时下载并缓存在
  ``MULTIMEDIA_FACE_MODEL_DIR``（默认 ``~/.cache/multimedia_face``）。
- **惰性初始化**：landmarker 只创建一次并复用；线程内缓存。

对外接口
--------
- :func:`align_face` → 返回对齐后的 JPEG data URL 及元信息。
"""
from __future__ import annotations

import os
import io
import math
import base64

from .media_paths import media_url_to_local
import threading
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

__all__ = ["align_face", "is_available"]

# ── 模型 ────────────────────────────────────────────────────────────────
_LANDMARKER_MODEL = "face_landmarker.task"
_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_MODEL_DIR = Path(
    os.getenv("MULTIMEDIA_FACE_MODEL_DIR")
    or (Path.home() / ".cache" / "multimedia_face")
)

# 左/右眼关键点（478 点 face mesh）：优先用虹膜中心，退化到眼角均值
_RIGHT_IRIS, _LEFT_IRIS = 468, 473
_LEFT_EYE_CORNERS = (33, 133)
_RIGHT_EYE_CORNERS = (362, 263)

_landmarker = None
_lock = threading.Lock()
_init_failed = False


def _ensure_model() -> Optional[Path]:
    path = _MODEL_DIR / _LANDMARKER_MODEL
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(_LANDMARKER_URL, tmp)
        tmp.replace(path)
        return path
    except Exception as e:  # noqa: BLE001
        print(f"    [P1-4][FaceAlign] 模型下载失败(不阻断): {str(e)[:120]}")
        return None


def _get_landmarker():
    global _landmarker, _init_failed
    if _landmarker is not None or _init_failed:
        return _landmarker
    with _lock:
        if _landmarker is not None or _init_failed:
            return _landmarker
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_model()
            if model_path is None:
                _init_failed = True
                return None

            opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
            )
            _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        except Exception as e:  # noqa: BLE001
            print(f"    [P1-4][FaceAlign] mediapipe 初始化失败(不阻断): {str(e)[:120]}")
            _init_failed = True
    return _landmarker


def is_available() -> bool:
    """mediapipe 是否可用（不触发模型下载）。"""
    try:
        import mediapipe  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ── 图像 I/O ────────────────────────────────────────────────────────────
def _load_rgb(image_input: str):
    """加载为 RGB numpy 数组；支持 http(s)/data URL/本地路径。"""
    import numpy as np
    from PIL import Image

    s = str(image_input or "")
    if not s:
        raise ValueError("空图片输入")
    if s.startswith("data:"):
        b64 = s.split(",", 1)[1] if "," in s else ""
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
    elif s.startswith(("http://", "https://")):
        with urllib.request.urlopen(s, timeout=30) as resp:
            img = Image.open(io.BytesIO(resp.read()))
    else:
        _lp = media_url_to_local(s) if s.startswith("/media/") else None
        img = Image.open(str(_lp) if _lp is not None else s)
    with img:
        return np.asarray(img.convert("RGB"))


def _to_data_url(image, max_size: int = 512, quality: int = 90) -> str:
    from PIL import Image

    img = image.convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_size:
        scale = max_size / float(long_side)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _eye_center(kps, iris_idx: int, corners: Tuple[int, int]) -> Tuple[float, float]:
    if len(kps) > iris_idx:
        p = kps[iris_idx]
        return float(p.x), float(p.y)
    a, b = kps[corners[0]], kps[corners[1]]
    return (float(a.x) + float(b.x)) / 2.0, (float(a.y) + float(b.y)) / 2.0


def align_face(
    image_input: str,
    size: int = 256,
    margin: float = 0.6,
    max_size: int = 512,
) -> Dict[str, Any]:
    """检测人脸并做几何归一化（摆正 + 裁剪），返回对齐图。

    Args:
        image_input: http(s)/data URL/本地路径。
        size: 输出正方形边长（像素）。
        margin: 人脸框外扩比例（0.6 表示四周各外扩 60%）。
        max_size: 输出 JPEG 的最长边上限（data URL 体积控制）。

    Returns:
        ``{"available", "aligned_data_url", "bbox", "landmarks", "angle",
        "method", "reason"}``。任何失败都返回 ``available=False``，调用方回退原图。
    """
    result: Dict[str, Any] = {
        "available": False,
        "aligned_data_url": None,
        "bbox": None,
        "landmarks": 0,
        "angle": 0.0,
        "method": "mediapipe_align",
        "reason": "",
    }
    landmarker = _get_landmarker()
    if landmarker is None:
        result["reason"] = "mediapipe 不可用"
        return result

    try:
        import numpy as np
        import mediapipe as mp
        from PIL import Image

        arr = _load_rgb(image_input)
        h, w = arr.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        res = landmarker.detect(mp_img)
        if not res.face_landmarks:
            result["reason"] = "未检出人脸"
            return result

        kps = res.face_landmarks[0]
        result["landmarks"] = len(kps)

        e1 = _eye_center(kps, _RIGHT_IRIS, _RIGHT_EYE_CORNERS)  # 归一化坐标
        e2 = _eye_center(kps, _LEFT_IRIS, _LEFT_EYE_CORNERS)
        # 按 x 排序，保证「图左眼→图右眼」，避免左右索引命名与几何方向不一致导致角度反转
        (ax, ay), (bx, by) = (e1, e2) if e1[0] <= e2[0] else (e2, e1)
        # 关键点坐标为归一化 0-1，还原到像素
        ax, ay, bx, by = ax * w, ay * h, bx * w, by * h
        cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
        angle = math.degrees(math.atan2(by - ay, bx - ax))
        result["angle"] = round(angle, 2)

        # 关键点包围盒（原图坐标）
        xs = [float(p.x) * w for p in kps]
        ys = [float(p.y) * h for p in kps]
        bw, bh = (max(xs) - min(xs)), (max(ys) - min(ys))
        side = max(bw, bh) * (1.0 + margin)
        side = max(side, 32.0)

        img = Image.fromarray(arr)
        # 绕人脸中心旋转，使双眼连线水平
        rotated = img.rotate(angle, center=(cx, cy), resample=Image.BICUBIC, expand=False)

        half = side / 2.0
        box = (int(round(cx - half)), int(round(cy - half)),
               int(round(cx + half)), int(round(cy + half)))
        crop = rotated.crop(box)
        crop = crop.resize((size, size), Image.LANCZOS)

        result["available"] = True
        result["aligned_data_url"] = _to_data_url(crop, max_size=max_size)
        result["bbox"] = [int(round(cx - half)), int(round(cy - half)), int(round(side)), int(round(side))]
        return result
    except Exception as e:  # noqa: BLE001
        result["available"] = False
        result["reason"] = f"对齐异常: {str(e)[:120]}"
        return result
