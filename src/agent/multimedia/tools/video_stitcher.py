# src/agent/multimedia/tools/video_stitcher.py
import os
import time
import requests
from typing import List

# 兼容 MoviePy 1.x / 2.x
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy import VideoFileClip, concatenate_videoclips

from moviepy.video.fx import Resize, Margin


# ==============================
# 下载模块（带重试）
# ==============================
def download_video(url: str, filename: str, retries: int = 3):
    print(f"    [下载] 获取片段: {url[-60:]}...")

    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            if os.path.getsize(filename) < 50_000:
                raise Exception("文件过小，疑似损坏")

            print(f"    ✅ 下载完成: {filename}")
            return

        except Exception as e:
            print(f"    ⚠️ 下载失败 (尝试 {attempt+1}/{retries}): {e}")
            time.sleep(2)

    raise Exception(f"视频下载失败（多次重试）: {url}")


# ==============================
# clip 标准化
# ==============================
def normalize_clip(clip: VideoFileClip) -> VideoFileClip:
    target_width = 1920
    target_height = 1080

    # --- Resize ---
    try:
        if hasattr(clip, "resized"):
            clip = clip.resized(width=target_width)
        else:
            clip = Resize(width=target_width).apply(clip)
    except Exception as e:
        print(f"      ⚠️ Resize 失败: {e}")
        raise

    # --- FPS ---
    try:
        if hasattr(clip, "set_fps"):
            clip = clip.set_fps(24)
        elif hasattr(clip, "with_fps"):
            clip = clip.with_fps(24)
    except Exception:
        pass

    # --- Padding ---
    if abs(clip.h - target_height) > 2:
        top = max(0, (target_height - clip.h) // 2)
        bottom = max(0, target_height - clip.h - top)

        try:
            clip = Margin(
                top=top,
                bottom=bottom,
                left=0,
                right=0,
                color=(0, 0, 0)
            ).apply(clip)
        except Exception as e:
            print(f"      ⚠️ Padding 失败: {e}")
            raise

    return clip


# ==============================
# 主拼接逻辑
# ==============================
def stitch_videos(
    video_urls: List[str],
    output_filename: str = "epic_game_trailer.mp4"
) -> str:
    print("\n--- 🎬 [剪辑师] 开始拼接宣传片 ---")

    temp_files: List[str] = []
    clips: List[VideoFileClip] = []

    try:
        # ==========================
        # 下载 + 构建 clip
        # ==========================
        for i, url in enumerate(video_urls):
            if not url:
                print(f"    ⚠️ 跳过空 URL (镜头 {i+1})")
                continue

            temp_file = f"temp_scene_{i:02d}.mp4"

            try:
                download_video(url, temp_file)
                temp_files.append(temp_file)

                clip = VideoFileClip(temp_file)

                # --- 防止 duration=0 ---
                if not clip.duration or clip.duration < 0.3:
                    print(f"    ⚠️ 无效视频（duration={clip.duration}），跳过")
                    clip.close()
                    continue

                print(f"    [处理] 镜头 {i+1} 标准化")

                clip = normalize_clip(clip)

                clips.append(clip)

            except Exception as e:
                print(f"    ❌ 镜头 {i+1} 处理失败: {e}")
                continue

        # ==========================
        # 兜底检查
        # ==========================
        if len(clips) == 0:
            raise Exception("没有可用的视频片段，无法生成最终视频")

        if len(clips) == 1:
            print("    ⚠️ 仅有一个片段，直接导出")
            final_clip = clips[0]
        else:
            print(f"    [剪辑] 拼接 {len(clips)} 个片段...")
            final_clip = concatenate_videoclips(
                clips,
                method="compose"
            )

        # ==========================
        # 导出
        # ==========================
        print(f"    [渲染] 导出最终视频 → {output_filename}")

        try:
            final_clip.write_videofile(
                output_filename,
                codec="libx264",
                audio_codec="aac",
                fps=24,
                preset="medium",
                threads=4,
                logger=None
            )
        except Exception as e:
            print(f"    ❌ 渲染失败，尝试 fallback: {e}")

            # fallback：降低线程 + 去掉 audio
            final_clip.write_videofile(
                output_filename,
                codec="libx264",
                fps=24,
                preset="ultrafast",
                threads=1,
                logger=None
            )

        abs_path = os.path.abspath(output_filename)
        print(f"    ✅ 成片完成: {abs_path}")
        return abs_path

    except Exception as e:
        print(f"    ❌ [剪辑失败] {str(e)}")
        raise e

    finally:
        print("    [清理] 释放资源...")

        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass

        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"    ⚠️ 删除失败: {f} ({e})")
