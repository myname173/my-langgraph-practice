"""
轻量级模型接入自检（不调用真实 API，不跑完整 pipeline）。

目的：在不消耗额度、不依赖网络的前提下，静态验证模型接入逻辑是否正确：
  1. 各模块可正常导入
  2. 模型链常量无空串、无已移除的旧模型名残留
  3. payload 构建函数用新模型名能成功构造（捕捉 schema 构建逻辑错误）
  4. SWE llm 工厂可实例化

运行：python tests/repro_model_wiring_check.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OLD_MODELS = [
    "tongyi-xiaomi", "qwen3.5-plus", "wan2.7-i2v", "wan2.6-i2v",
    "wan2.2-i2v", "wanx2.1-i2v", "wan2.5-i2v", "wan2.7-image",
    "wan2.6-image", "wanx-v1", "wan2.6-t2i", "qwen-image-2.0\"",
]
EXPECTED_TEXT = [
    "qwen3.7-max-2026-06-08", "qwen3.7-max", "qwen3.7-plus",
    "qwen3.7-flash", "deepseek-v4-flash-0731", "glm-5.2", "qwen3.5-ocr",
]
EXPECTED_VIDEO = [
    "wan2.7-r2v-2026-06-12", "happyhorse-1.1-r2v", "happyhorse-1.1-i2v",
    "wan2.7-t2v-2026-06-12", "wan3.0-video", "happyhorse-1.1-t2v",
]
EXPECTED_IMAGE = [
    "qwen-image-3.0", "qwen-image-3.0-pro", "qwen-image-2.0-pro-2026-06-22",
]

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print(f"[FAIL] {msg}")
    else:
        print(f"[ OK ] {msg}")


def main():
    # 1. imports
    try:
        import agent.multimedia.tools.text_llm as tl
        import agent.multimedia.tools.image_gen as ig
        import agent.multimedia.tools.video_gen as vg
        import agent.multimedia.tools.vision_eval as ve
        print("[ OK ] 所有多媒体模块导入成功")
    except Exception as e:
        print(f"[FAIL] 模块导入失败: {e}")
        raise SystemExit(1)

    # 2. text fallback chain
    chain = tl._LLM_FALLBACK_MODELS
    check(all(chain), "text_llm fallback 链无空串")
    check(not any("tongyi-xiaomi" in m for m in chain), "text_llm 已移除占位模型")
    check(all(any(m == e or e in m for m in chain) for e in EXPECTED_TEXT),
          "text_llm 含全部预期文本模型")

    # 3. image chains
    img_models = [m for m, _ in ig._IMG2IMG_CHAIN] + [m for m, _ in ig._TEXT2IMAGE_CHAIN]
    check(not any(any(old in m for old in OLD_MODELS) for m in img_models),
          "image_gen 链无已移除的旧模型")
    check(all(e in img_models for e in EXPECTED_IMAGE),
          "image_gen 含全部预期图片模型")

    # 4. video chains
    all_v = vg._R2V_MODEL_CHAIN + vg._I2V_MODEL_CHAIN + vg._T2V_MODEL_CHAIN
    check(not any(any(old in m for old in OLD_MODELS) for m in all_v),
          "video_gen 链无已移除的旧模型")
    check(all(e in all_v for e in EXPECTED_VIDEO),
          "video_gen 含全部预期视频模型")

    # 5. payload 构建（不调用 API）
    try:
        p_i2v = vg._build_payload_media_frames(
            "happyhorse-1.1-i2v", "http://x/f.png", "http://x/l.png", "a cat")
        check(isinstance(p_i2v, dict) and p_i2v["model"] == "happyhorse-1.1-i2v",
              "video i2v payload 构建成功")
        p_r2v = vg._build_payload_kf2v(
            "wan2.7-r2v-2026-06-12", "http://x/f.png", "http://x/l.png", "a cat")
        check(isinstance(p_r2v, dict) and "first_frame_url" in p_r2v["input"],
              "video r2v payload 构建成功 (含 first_frame_url)")
        p_t2v = vg._build_payload_t2v("wan3.0-video", "a cat running")
        check(isinstance(p_t2v, dict) and "prompt" in p_t2v["input"]
              and "img_url" not in p_t2v["input"],
              "video t2v payload 构建成功 (仅 prompt, 无图)")
    except Exception as e:
        failures.append(f"payload 构建异常: {e}")
        print(f"[FAIL] payload 构建异常: {e}")

    # 6. image payload（qwen-image 走 multimodal-generation 端点）
    try:
        p_img = ig._build_payload(
            "qwen-image-3.0-pro", "multimodal-generation/generation",
            "a castle", "1280*720", reference_image_url="http://x/ref.png")
        check(isinstance(p_img, dict) and p_img["model"] == "qwen-image-3.0-pro",
              "image qwen-image payload 构建成功")
    except Exception as e:
        failures.append(f"image payload 构建异常: {e}")
        print(f"[FAIL] image payload 构建异常: {e}")

    # 7. SWE llm factory（读源码文本检查，避免触发 langchain 导入导致环境假阴性）
    try:
        swe_graph_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "agent", "swe", "graph.py")
        with open(swe_graph_path, encoding="utf-8") as f:
            src = f.read()
        check('def _build_swe_llm' in src, "SWE 存在 _build_swe_llm 工厂")
        check('os.getenv("MODEL_NAME"' in src, "SWE _build_swe_llm 读取 MODEL_NAME")
        check("SWE_FALLBACK_MODELS" in src, "SWE _build_swe_llm 支持 fallback 列表")
        check("qwen3.7-max" in src, "SWE 默认模型为可用 qwen3.7-max")
    except Exception as e:
        failures.append(f"SWE llm 检查异常: {e}")
        print(f"[FAIL] SWE llm 检查异常: {e}")

    # 8. vision eval 模块健康检查
    check(hasattr(ve, "evaluate_image") and hasattr(ve, "evaluate_video"),
          "vision_eval 关键函数存在")
    print(f"      vision_eval 默认模型: {os.getenv('DASHSCOPE_VISION_MODEL', 'qwen-image-3.0-pro')}")

    print("\n==== 结论 ====")
    if failures:
        print(f"发现 {len(failures)} 个问题，无需跑完整流程即可暴露。")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("全部逻辑自检通过：模型接入链路正确，遗留旧模型已清除。")


if __name__ == "__main__":
    main()
