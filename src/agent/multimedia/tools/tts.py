# src/agent/multimedia/tools/tts.py
"""
TTS 配音工具 —— 复用 GitHub 现成组件，不自造语音合成算法。

引擎（按优先级 / 可降级）：
  1. edge-tts  (GitHub: rany2/edge-tts) —— 微软神经网络 TTS。
     纯 pip 包、零模型下载、中文多音色、免费。作为默认引擎，直接拼进项目。
  2. dashscope CosyVoice / Sambert (阿里通义) —— 复用项目已有 DASHSCOPE_API_KEY，
     语音更自然。仅当 dashscope 包可用时作为可选降级路径。

接口约定与本模块其它 tool 一致：纯函数，输入文本 → 返回本地音频文件路径。
"""
import asyncio
import os
import tempfile
from typing import Optional

import edge_tts

# 中文神经网络音色（微软免费可用，前沿多情感）
EDGE_TTS_VOICES_ZH = {
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",   # 女声 温柔叙事
    "yunyang": "zh-CN-YunyangNeural",     # 男声 沉稳解说
    "xiaoyi": "zh-CN-XiaoyiNeural",       # 女声 轻快
    "yunxi": "zh-CN-YunxiNeural",         # 男声 活泼
    "xiaochen": "zh-CN-XiaochenNeural",   # 女声 成熟知性
    "xiaomo": "zh-CN-XiaomoNeural",       # 女声 情感丰富
}


async def _edge_tts_save(text: str, voice: str, out_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def generate_voiceover(
    text: str,
    voice: str = "xiaoxiao",
    out_path: Optional[str] = None,
) -> str:
    """生成配音音频，返回本地 .mp3 路径。

    Args:
        text: 配音文本（通常来自分镜 scene["script"] 汇总）。
        voice: 音色 key（见 EDGE_TTS_VOICES_ZH），或直接传 edge-tts voice 名。
        out_path: 可选输出路径，默认用临时文件。
    """
    if not text or not text.strip():
        return ""
    voice_id = EDGE_TTS_VOICES_ZH.get(voice, voice)
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".mp3", prefix="tts_")
        os.close(fd)

    try:
        asyncio.run(_edge_tts_save(text, voice_id, out_path))
        return out_path
    except Exception as e:  # edge-tts 不可用时降级到 dashscope
        print(f"    [TTS] edge-tts 失败: {e}，尝试 dashscope CosyVoice ...")
        try:
            return _generate_voiceover_dashscope(text, voice, out_path)
        except Exception as e2:
            # 两个引擎都失败：返回空串，让上层优雅降级（跳过配音，不中断全流程）
            print(f"    [TTS] dashscope 也失败，跳过该段配音: {e2}")
            return ""


def _generate_voiceover_dashscope(text: str, voice: str, out_path: str) -> str:
    """可选降级：阿里通义 CosyVoice，复用项目已有 DASHSCOPE_API_KEY。

    注意：dashscope 的 SpeechSynthesizer.call 返回 SpeechSynthesizerResult，
    正确用法是 result.get_audio_data() / result.get_timestamp()，**没有 begin_time 字段**
    （那是 edge-tts 的结构）。本函数只取音频字节写盘，不依赖任何不存在的字段。
    """
    try:
        import dashscope
        from dashscope.audio.tts import SpeechSynthesizer
    except ImportError:
        raise RuntimeError("未安装 dashscope，无法使用 CosyVoice 降级路径。")

    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not dashscope.api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法使用 CosyVoice 降级路径。")

    # CosyVoice 默认模型；voice 映射为通义音色（longxiaochun 等）
    cosy_voice = {
        "xiaoxiao": "longxiaochun",
        "yunyang": "longxiang",
        "xiaoyi": "longxiaomiaoen",
    }.get(voice, "longxiaochun")

    result = SpeechSynthesizer.call(
        model="cosyvoice-v1",
        text=text,
        voice=cosy_voice,
        api_key=dashscope.api_key,
    )
    audio = result.get_audio_data()
    if audio is None:
        # SpeechSynthesizerResult 上并没有 .message（那是别的 SDK 的结构）；直接访问该
        # 属性会抛 AttributeError，把真正的合成失败原因（如 API Key 无效）完全掩盖。
        # 这里防御式提取状态信息，保证报错可读。
        _code = getattr(result, "status_code", None) or getattr(result, "code", None)
        _msg = getattr(result, "message", None) or getattr(result, "msg", None)
        if _code is None and _msg is None:
            _msg = f"{type(result).__name__} 未返回音频数据"
        raise RuntimeError(f"CosyVoice 合成失败: code={_code}, message={_msg}")
    with open(out_path, "wb") as f:
        f.write(audio)
    return out_path
