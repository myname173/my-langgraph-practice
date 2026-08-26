import sys
from unittest import mock
sys.path.insert(0, "src")

import agent.multimedia.tools.image_gen as ig

calls = {}

def fake_jimeng_generate_image(prompt, size="2K", reference_image_urls=None,
                                ref_strength=0.35, timeout=120, thread_id=""):
    calls["jimeng"] = True
    calls["prompt"] = prompt
    calls["refs"] = reference_image_urls
    return "data:image/png;base64,AAAA"

# patch 所有可能的 fallback 目标：确保它们不被调用
dashscope_sentinel = mock.MagicMock(side_effect=AssertionError("DashScope 不应被调用"))
siliconflow_sentinel = mock.MagicMock(side_effect=AssertionError("SiliconFlow 不应被调用"))

with mock.patch.object(ig.jimeng, "generate_image", side_effect=fake_jimeng_generate_image):
    # 即便 backend 配成 dashscope，也应只走 jimeng
    with mock.patch.object(ig, "get_visual_backend", return_value="dashscope"):
        out = ig.generate_keyframe(
            "a hero on cliff",
            reference_image_urls=["file:///x/char_main.jpg"],
            ref_strength=0.5,
            negative_prompt="blurry, extra limbs",
            thread_id="t1",
        )

assert out.startswith("data:image"), "jimeng 未返回"
assert calls.get("jimeng"), "jimeng.generate_image 未被调用"
assert "Avoid:" in calls["prompt"], "negative_prompt 未融入"
print(f"[OK] generate_keyframe 只走 jimeng，backend=dashscope 也被强制走 jimeng")
print(f"     prompt: {calls['prompt']!r}")
print(f"     refs: {calls['refs']}")

# 确认文件中不再有 dashscope/siliconflow 的实际调用路径（仅历史注释/辅助函数保留不触发）
src = open("src/agent/multimedia/tools/image_gen.py", encoding="utf-8").read()
# generate_keyframe 体内不应出现 dashscope 提交逻辑
import re
body = src[src.index("def generate_keyframe"):]
# 找到下一个顶层 def 之前的内容
nxt = body.find("\ndef ", 1)
if nxt != -1:
    body = body[:nxt]
assert "dashscope.aliyuncs.com" not in body, "generate_keyframe 仍含 DashScope 调用"
assert "generate_keyframe_siliconflow" not in body, "generate_keyframe 仍调用 siliconflow"
print("[OK] generate_keyframe 函数体已无 DashScope/SiliconFlow 调用路径")

print("\n=== 关键帧只走 jimeng 验证通过 ===")
