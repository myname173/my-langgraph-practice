# src/agent/multimedia/rag/ingestors/knowledge_generator.py
"""
LLM 驱动的知识生成器
====================
一次性运行，系统性生成专业摄影/布光/运镜知识 markdown 文件，
存入 vision_knowledge/generated/ 目录，再由 markdown_ingestor 灌入 ChromaDB。

用法（独立运行）：
    python -m src.agent.multimedia.rag.ingestors.knowledge_generator

或通过 ingest.py：
    python -m src.agent.multimedia.rag.ingest --source gen
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ============================================================
# 输出目录
# ============================================================
_GENERATED_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "vision_knowledge"
    / "generated"
)

# ============================================================
# 知识主题清单 — 每个主题 = 一次 LLM 调用
# ============================================================

KNOWLEDGE_TOPICS: List[Dict[str, str]] = [
    {
        "filename": "lighting_setups.md",
        "title": "Professional Lighting Setups for Cinematic Trailers",
        "prompt": (
            "你是一位好莱坞 AAA 游戏宣传片的灯光总监。\n"
            "请系统性地写出 30 种专业布光方案，每种方案包含：\n"
            "- 名称（英文）\n"
            "- 适用场景（如：hero reveal、villain entrance、battle climax）\n"
            "- 主光类型和方向\n"
            "- 补光和背光设置\n"
            "- 色温（K 值）\n"
            "- 情绪效果\n"
            "- 对应的英文 AI 图像生成 prompt 关键词（1-2 个短语）\n\n"
            "用 markdown 表格格式输出，按 ## Key Light / ## Rim Light / ## Ambient Light / "
            "## Volumetric Light / ## Special Effects Light 五个大类组织。\n"
            "确保覆盖：自然光、studio 光、混合光、极端风格化光效。\n"
            "只输出专业内容，不要废话。全部用英文撰写技术术语，中文解释可保留。"
        ),
    },
    {
        "filename": "camera_psychology.md",
        "title": "Camera Movement Psychology & Narrative Impact",
        "prompt": (
            "你是一位电影心理学和摄影运动学专家。\n"
            "请详细写出运镜方式与观众心理反应之间的对应关系，覆盖以下维度：\n\n"
            "1. ## Speed and Emotion — 不同运动速度（极慢/慢/中/快/极快）对应的心理效果\n"
            "2. ## Direction and Meaning — 运动方向（左→右/右→左/上/下/旋转）的叙事含义\n"
            "3. ## Acceleration Patterns — 加速/减速/突然停止的情绪冲击\n"
            "4. ## Multi-Axis Movement — 复合运动（同时推进+上升+旋转）的戏剧效果\n"
            "5. ## Stillness vs Motion — 静止镜头 vs 运动镜头的心理对比\n"
            "6. ## Rhythm and Breathing — 运镜节奏与观众呼吸/心率的关系\n\n"
            "每个维度至少给出 5 个具体的例子和对应的 AI 视频生成 prompt 关键词。\n"
            "用 markdown 格式，每个大类用 ## 标题。全部用英文撰写技术术语。"
        ),
    },
    {
        "filename": "color_theory.md",
        "title": "Color Theory for Cinematic Visual Storytelling",
        "prompt": (
            "你是一位色彩理论和电影调色专家。\n"
            "请系统性地写出色彩在游戏宣传片中的叙事功能，包含：\n\n"
            "1. ## Color Palettes by Emotion — 列出 15 种情绪（恐惧/希望/愤怒/孤独/胜利/神秘/浪漫/压迫/自由/悲伤/狂热/紧张/平静/混乱/庄严），每种给出：\n"
            "   - 主色调 + 辅助色（具体 HEX 值或颜色名）\n"
            "   - 色彩搭配方案（complementary/analogous/triadic）\n"
            "   - 对应的 AI prompt 色彩关键词\n"
            "2. ## Color Transitions — 10 种色彩过渡策略（如：冷→暖表示希望升起），每种配 narrative 场景\n"
            "3. ## Genre Color Coding — 6 种游戏类型（奇幻/科幻/恐怖/军事/赛博朋克/东方仙侠）的标志性调色方案\n"
            "4. ## Time of Day Color — 12 个时段的色彩特征（从黎明到深夜）\n\n"
            "用 markdown 表格和 ## 标题组织。全部用英文撰写技术术语。"
        ),
    },
    {
        "filename": "composition_rules.md",
        "title": "Advanced Composition Rules for Cinematic Frames",
        "prompt": (
            "你是一位电影构图学大师。\n"
            "请系统性地写出 25 种高级构图法则，每种包含：\n"
            "- 名称（英文）\n"
            "- 原理说明（1-2 句）\n"
            "- 适用场景（hero shot、battle、environment reveal 等）\n"
            "- 视觉重心位置\n"
            "- 对应的 AI 图像 prompt 关键词\n\n"
            "包括但不限于：Rule of Thirds、Golden Ratio、Leading Lines、Frame within Frame、"
            "Symmetry Breaking、Negative Space、Diagonal Tension、Depth Layering、"
            "Foreground Anchor、Silhouette Composition、Dutch Angle、Overhead Pattern。\n"
            "按 ## Static Composition / ## Dynamic Composition / ## Group Composition / "
            "## Environmental Composition 四个大类组织。\n"
            "用 markdown 表格和 ## 标题。全部用英文撰写技术术语。"
        ),
    },
    {
        "filename": "mood_atmosphere.md",
        "title": "Mood & Atmosphere Design for Game Trailers",
        "prompt": (
            "你是一位 AAA 游戏宣传片的氛围设计总监。\n"
            "请写出 20 种预告片氛围模板，每种包含：\n"
            "- 氛围名称（如：Dread Building、Epic Triumph、Mysterious Discovery）\n"
            "- 光影方案（主光/环境光/体积光/粒子效果）\n"
            "- 色彩方案（主色/辅色/强调色）\n"
            "- 运镜策略（开场/高潮/结尾各推荐什么运镜）\n"
            "- 粒子/天气效果（雨/雾/尘/火星/花瓣等）\n"
            "- 节奏指南（慢→快？脉冲式？渐强？）\n"
            "- 完整的 AI prompt 示例（一段 40-60 词的英文 prompt，展示如何将以上元素融合）\n\n"
            "按 ## Tension & Horror / ## Epic & Heroic / ## Mystery & Wonder / "
            "## Melancholy & Loss / ## Chaos & Battle 五个大类组织。\n"
            "用 markdown 格式，每种氛围用 ### 子标题。全部用英文撰写技术术语。"
        ),
    },
    {
        "filename": "style_guides.md",
        "title": "Visual Style Guides for Game Genres",
        "prompt": (
            "你是一位跨风格的游戏美术总监，精通多种游戏类型的视觉设计。\n"
            "请为以下 8 种游戏类型各写一份视觉风格指南：\n"
            "1. Dark Fantasy（黑暗奇幻）\n"
            "2. Cyberpunk / Sci-Fi（赛博朋克/科幻）\n"
            "3. Post-Apocalyptic（末世废土）\n"
            "4. Eastern Fantasy / Xianxia（东方仙侠）\n"
            "5. Military / Tactical（军事战术）\n"
            "6. Horror / Survival（恐怖生存）\n"
            "7. Steampunk（蒸汽朋克）\n"
            "8. Mythological / Ancient（神话/古代文明）\n\n"
            "每种风格的指南包含：\n"
            "- ## [Genre Name]\n"
            "- Signature Visual Elements（标志性视觉元素 3-5 个）\n"
            "- Color Palette（色彩方案）\n"
            "- Lighting Style（光影风格）\n"
            "- Material & Texture Keywords（材质/纹理关键词）\n"
            "- Camera Preferences（偏好运镜）\n"
            "- AI Prompt Template（一段完整的 50 词英文 prompt 模板）\n"
            "- Negative Prompts（应该避免的视觉元素）\n\n"
            "用 markdown 格式组织。全部用英文撰写技术术语。"
        ),
    },
    {
        "filename": "lens_optics.md",
        "title": "Lens & Optics Guide for Cinematic Depth",
        "prompt": (
            "你是一位电影镜头光学专家。\n"
            "请系统性地写出镜头选择与画面效果的关系知识：\n\n"
            "1. ## Focal Length Categories — 列出 8 种焦段（14mm/24mm/35mm/50mm/85mm/135mm/200mm/400mm），"
            "每种说明：透视变形、景深特征、最佳使用场景、对应的 AI prompt 关键词\n"
            "2. ## Depth of Field Control — 5 种景深策略（极浅/浅/中/深/全景深）的叙事功能和 prompt 写法\n"
            "3. ## Lens Effects — 8 种镜头光学效果（lens flare/bokeh/anamorphic streak/chromatic aberration/"
            "vignette/barrel distortion/motion blur/rack focus），每种配叙事用法和 prompt 关键词\n"
            "4. ## Lens Selection by Scene Type — 为 10 种常见场景（hero reveal/chase/intimate dialogue/"
            "battle panorama/mystery object/establishing shot 等）推荐镜头组合\n\n"
            "用 markdown 表格和 ## 标题。全部用英文撰写技术术语。"
        ),
    },
    {
        "filename": "transition_techniques.md",
        "title": "Visual Transition Techniques for Trailer Editing",
        "prompt": (
            "你是一位预告片剪辑大师，精通镜头间的视觉过渡技法。\n"
            "请写出 20 种预告片转场技法，每种包含：\n"
            "- 名称（英文）\n"
            "- 原理说明\n"
            "- 叙事功能（时间跳跃？空间转换？情绪转变？因果衔接？）\n"
            "- 实现方式（在 AI 生成中如何通过 prompt 实现前后帧的视觉衔接）\n"
            "- 前帧 prompt 要点 + 后帧 prompt 要点\n\n"
            "包括但不限于：Match Cut、Smash Cut、Whip Pan、Light Flash、"
            "Object Wipe、Color Dissolve、Motion Continuity、Scale Shift、"
            "Silhouette Bridge、Depth Rack、Particle Wipe。\n\n"
            "按 ## Hard Cuts / ## Soft Transitions / ## Motion-Based / ## Light-Based / "
            "## Creative Transitions 五个大类组织。\n"
            "用 markdown 表格和 ## 标题。全部用英文撰写技术术语。"
        ),
    },
    {
        "filename": "pacing_editing.md",
        "title": "Pacing & Editing Rhythm for Game Trailers",
        "prompt": (
            "你是一位游戏宣传片的节奏设计师。\n"
            "请系统性地写出预告片节奏控制知识：\n\n"
            "1. ## Pacing Templates — 5 种预告片节奏模板（Epic Build-up / Rapid Montage / "
            "Pulse Rhythm / Slow Burn / Crescendo），每种给出：\n"
            "   - 时间轴结构（各阶段时长比例）\n"
            "   - 镜头数建议\n"
            "   - 每个阶段的运镜速度\n"
            "   - 情绪曲线描述\n"
            "2. ## Shot Duration Strategy — 不同镜头时长（0.5s/1s/2s/3s/5s+）的心理效果\n"
            "3. ## Rhythm Breaking — 何时打破节奏以及打破方式的戏剧效果\n"
            "4. ## Multi-Scene Orchestration — 4 镜头/5 镜头/6 镜头序列的编排策略\n"
            "5. ## Emotional Arc Design — 设计预告片情绪弧线的 8 种模式\n\n"
            "用 markdown 格式，每个大类用 ## 标题。全部用英文撰写技术术语。"
        ),
    },
    {
        "filename": "prompt_engineering.md",
        "title": "AI Image Prompt Engineering for Cinematic Quality",
        "prompt": (
            "你是一位 AI 图像生成 prompt 工程专家，精通 Midjourney/Stable Diffusion/DALL-E 等模型。\n"
            "请写出游戏宣传片场景的高质量 prompt 工程知识：\n\n"
            "1. ## Prompt Structure — 最佳 prompt 结构模板（主体/环境/光影/镜头/风格/质量词 的排列顺序）\n"
            "2. ## Quality Boosters — 30 个能显著提升画面质量的 prompt 后缀词，每个说明效果\n"
            "3. ## Style Anchors — 20 个风格锚定词（如 'Unreal Engine 5'/'Octane Render'/'ray tracing'），"
            "说明各自对画面的影响\n"
            "4. ## Negative Prompt Strategy — 15 种常见画面问题及其对应的 negative prompt 解决方案\n"
            "5. ## Weight & Emphasis — prompt 中各元素的权重分配策略\n"
            "6. ## Complete Prompt Examples — 10 个完整的 60-80 词高质量 prompt 示例，"
            "覆盖不同场景类型（hero reveal/battle/environment/mystery/climax）\n\n"
            "用 markdown 格式，每个大类用 ## 标题。\n"
            "所有 prompt 示例必须是英文。解释说明可以用中文。"
        ),
    },
]


def _generate_single(topic: Dict[str, str], verbose: bool = True) -> bool:
    """生成单个知识文件。"""
    # 惰性导入：只在真正生成时才需要 openai 依赖
    from src.agent.multimedia.tools.text_llm import call_llm

    filepath = _GENERATED_DIR / topic["filename"]

    # 如果文件已存在且非空，跳过（避免重复生成）
    if filepath.exists() and filepath.stat().st_size > 100:
        if verbose:
            print(f"  ⏭  跳过（已存在）: {topic['filename']}")
        return True

    if verbose:
        print(f"  ⏳ 生成中: {topic['filename']} ...")

    try:
        content = call_llm(topic["prompt"], role_name="知识生成器")

        # 添加文件头
        header = (
            f"# {topic['title']}\n\n"
            f"> Auto-generated cinematography knowledge for AI trailer pipeline.\n"
            f"> Generated by LLM knowledge generator.\n\n"
            f"---\n\n"
        )
        full_content = header + content

        filepath.write_text(full_content, encoding="utf-8")

        if verbose:
            size = filepath.stat().st_size
            print(f"  ✅ 完成: {topic['filename']} ({size:,} bytes)")
        return True

    except Exception as e:
        if verbose:
            print(f"  ❌ 失败: {topic['filename']} — {e}")
        return False


def generate_all_knowledge(verbose: bool = True) -> int:
    """
    生成全部知识文件。

    Returns:
        成功生成的文件数
    """
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LLM 知识生成器")
        print(f"  输出目录: {_GENERATED_DIR}")
        print(f"  主题数: {len(KNOWLEDGE_TOPICS)}")
        print(f"{'='*60}\n")

    success = 0
    for i, topic in enumerate(KNOWLEDGE_TOPICS, 1):
        if verbose:
            print(f"[{i}/{len(KNOWLEDGE_TOPICS)}]")
        if _generate_single(topic, verbose=verbose):
            success += 1
        # 限速，避免 API 过载
        if i < len(KNOWLEDGE_TOPICS):
            time.sleep(1.5)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  生成完成: {success}/{len(KNOWLEDGE_TOPICS)} 个文件")
        print(f"{'='*60}")

    return success


def ingest_generated_knowledge(verbose: bool = True) -> int:
    """
    生成知识文件并灌入 ChromaDB。

    这是 ingest.py 注册表调用的入口函数。
    """
    # Step 1: 生成
    gen_count = generate_all_knowledge(verbose=verbose)
    if gen_count == 0:
        if verbose:
            print("  ⚠ 没有生成任何知识文件，跳过灌入")
        return 0

    # Step 2: 灌入（调用 markdown_ingestor，它会自动扫描 vision_knowledge/ 下所有 md 文件）
    from src.agent.multimedia.rag.ingestors.markdown_ingestor import ingest_markdown_files
    chunk_count = ingest_markdown_files(verbose=verbose)

    if verbose:
        print(f"\n  📦 知识生成 + 灌入完成: {gen_count} 个文件 → {chunk_count} chunks")

    return chunk_count


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM 知识生成器")
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有知识主题，不生成",
    )
    parser.add_argument(
        "--only", "-o",
        type=int,
        default=None,
        help="只生成指定序号的主题（从 1 开始）",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新生成（忽略已有文件）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="减少输出",
    )

    args = parser.parse_args()

    if args.list:
        for i, topic in enumerate(KNOWLEDGE_TOPICS, 1):
            print(f"  {i}. {topic['filename']} — {topic['title']}")
        sys.exit(0)

    if args.force:
        # 清除已有文件以强制重新生成
        for topic in KNOWLEDGE_TOPICS:
            fp = _GENERATED_DIR / topic["filename"]
            if fp.exists():
                fp.unlink()

    if args.only is not None:
        if 1 <= args.only <= len(KNOWLEDGE_TOPICS):
            topic = KNOWLEDGE_TOPICS[args.only - 1]
            # 强制删除已有文件
            fp = _GENERATED_DIR / topic["filename"]
            if fp.exists():
                fp.unlink()
            _generate_single(topic, verbose=not args.quiet)
        else:
            print(f"序号超出范围（1-{len(KNOWLEDGE_TOPICS)}）")
            sys.exit(1)
    else:
        generate_all_knowledge(verbose=not args.quiet)
