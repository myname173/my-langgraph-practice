# src/agent/multimedia/tools/video_post.py
"""P2-2 后处理板：免费后端输出的清晰度增强与补帧。

定位：免费/低价视频后端的产物常为低分辨率、低帧率。本模块用项目已有的
imageio_ffmpeg 自带 ffmpeg 二进制做两件事（均默认关闭，env 显式开启）：

  - 清晰度增强（target_height）：lanczos 上采样 + 轻量 unsharp 锐化。
    这是经典放大锐化管线，**不是模型超分**——真超分（Real-ESRGAN 等）需要
    额外二进制/模型权重，此处保持零新增依赖。
  - 补帧（target_fps）：interp="dup" 用 fps 滤镜重复帧（快，观感提升有限）；
    interp="mci" 用 minterpolate 运动补偿插值（观感最好，但极慢，分钟级/片段）。

失败一律返回原路径并告警，绝不阻断拼接主流程。
"""
import os
import subprocess
from typing import Optional, Tuple

_PROBE_TIMEOUT = 60
_RUN_TIMEOUT = 1800


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _probe_size(path: str) -> Tuple[Optional[int], Optional[int]]:
    """用 moviepy 读取宽高（比解析 ffprobe 输出更稳）。"""
    try:
        from moviepy import VideoFileClip
        with VideoFileClip(path) as clip:
            return int(clip.w), int(clip.h)
    except Exception:
        return None, None


def enhance_final_cut(
    video_path: str,
    target_height: Optional[int] = None,
    target_fps: Optional[int] = None,
    interp: str = "dup",
    out_path: Optional[str] = None,
) -> str:
    """对无声粗剪做清晰度增强/补帧；无需求或失败时原样返回。

    Args:
        video_path: 输入视频（stitcher 产物，无声）。
        target_height: 目标高度（如 1080）；仅当大于当前高度才放大。
        target_fps: 目标帧率（如 30）；仅当大于当前帧率才补帧。
        interp: "dup" 重复帧 / "mci" 运动补偿插值（很慢）。
        out_path: 输出路径（默认 <name>_post.mp4）。

    Returns:
        处理后的文件路径；失败/无需求返回原路径。
    """
    if not video_path or not os.path.exists(video_path):
        return video_path
    if not target_height and not target_fps:
        return video_path

    try:
        w, h = _probe_size(video_path)
        if not h:
            print("    [WARN] [P2-2] 无法读取视频尺寸，跳过后处理")
            return video_path

        vf_parts = []
        if target_height and int(target_height) > h:
            vf_parts.append(f"scale=-2:{int(target_height)}:flags=lanczos")
            vf_parts.append("unsharp=5:5:0.5:3:3:0.0")
        if target_fps and int(target_fps) >= 24:
            f = int(target_fps)
            if interp == "mci":
                vf_parts.append(
                    f"minterpolate=fps={f}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
                )
            else:
                vf_parts.append(f"fps={f}")

        if not vf_parts:
            print(f"    [P2-2] 无需处理（当前 {w}x{h} 已满足目标），跳过")
            return video_path

        if out_path is None:
            base, ext = os.path.splitext(video_path)
            out_path = f"{base}_post{ext}"

        exe = _ffmpeg_exe()
        cmd = [exe, "-y", "-i", video_path, "-vf", ",".join(vf_parts),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an", out_path]
        print(f"    [P2-2] 后处理：{' + '.join(vf_parts)}")
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              errors="ignore", timeout=_RUN_TIMEOUT)
        if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            tail = (proc.stderr or "")[-200:]
            print(f"    [WARN] [P2-2] 后处理失败，使用原成片: {tail}")
            return video_path
        print(f"    ✓ [P2-2] 后处理完成 → {os.path.basename(out_path)}")
        return out_path
    except subprocess.TimeoutExpired:
        print("    [WARN] [P2-2] 后处理超时，使用原成片（不阻断）")
        return video_path
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] [P2-2] 后处理异常(不阻断): {str(e)[:150]}")
        return video_path
