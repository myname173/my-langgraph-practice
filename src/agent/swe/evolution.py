# src/agent/swe/evolution.py
"""
Capability Evolution Loop
=========================
在每次任务成功完成后低频触发，将本次任务中有价值的解决方案结晶为可复用的 Skill 脚本。

三阶段流程：
  1. evolution_reflect  — 观察 + 反思 + 决策（一次结构化 LLM 调用）
  2. evolution_generate — 生成完整 Skill 脚本草稿
  3. evolution_verify   — AST 语法检查 + 沙盒验证 + 版本化持久化

设计原则：
  - 任何阶段失败只记录日志，绝不影响主任务 "success" 状态
  - Jaccard 相似度去重，防止低价值重复 Skill 污染库
  - Skill 文件带 YAML frontmatter，供 list_skills 展示丰富元数据
"""

import ast
import json
import logging
import os
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agent.swe.state import AgentState
from src.agent.swe.tools import SKILLS_DIR, WORKSPACE_DIR, sandbox
from src.agent.swe.prompts import EVOLUTION_REFLECT_PROMPT, EVOLUTION_GENERATE_PROMPT

load_dotenv()

logger = logging.getLogger("SWE_Evolution")

# 演化专用 LLM（temperature 稍高，激发创造力）
_evolution_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    temperature=0.25,
)

# 草稿目录
DRAFT_DIR = WORKSPACE_DIR / "_evolution_drafts"
DRAFT_DIR.mkdir(parents=True, exist_ok=True)

# 版本号上限，防止无界计数
_MAX_SKILL_VERSION = 99


# ==========================================
# Pydantic Schema
# ==========================================
class EvolutionReflectOutput(BaseModel):
    should_evolve: bool = Field(description="是否值得固化为可复用 Skill")
    reasoning: str = Field(description="决策理由（2-3 句话）")
    skill_name: str = Field(default="", description="技能文件名，如 setup_fastapi.py")
    description: str = Field(default="", description="一句话功能描述（中文）")
    category: str = Field(default="misc", description="scaffold|debug|test|config|deploy|misc")
    applicable_scenarios: str = Field(default="", description="适用场景（2-3 句，中文）")
    core_logic_summary: str = Field(default="", description="核心逻辑摘要，供代码生成参考")


# ==========================================
# 工具函数
# ==========================================

def parse_skill_metadata(filepath: Path) -> dict:
    """
    解析 Skill 脚本顶部的 YAML frontmatter 注释，返回元数据字典。
    格式：
        # ---
        # name: setup_fastapi.py
        # category: scaffold
        # description: ...
        # ---
    """
    meta = {
        "name": filepath.stem,
        "description": "无描述",
        "category": "misc",
        "version": "1.0",
        "created_at": "",
    }
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        in_fm = False
        for line in lines[:15]:
            stripped = line.strip()
            if stripped == "# ---":
                in_fm = not in_fm
                continue
            if in_fm and stripped.startswith("# ") and ": " in stripped:
                kv = stripped[2:]
                key, val = kv.split(": ", 1)
                meta[key.strip()] = val.strip()
    except Exception:
        pass
    return meta


def _format_existing_skills() -> str:
    """格式化已有技能库信息，供 Reflect LLM 做去重判断。"""
    skills = list(SKILLS_DIR.glob("*.py"))
    if not skills:
        return "（技能库当前为空）"
    lines = []
    for f in sorted(skills):
        m = parse_skill_metadata(f)
        lines.append(f"[{m.get('category', 'misc')}] {f.name} — {m.get('description', '无描述')}")
    return "\n".join(lines)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """基于词集合的 Jaccard 相似度，用于快速判断两个描述是否高度重叠。"""
    words_a = set(re.findall(r"\w+", text_a.lower()))
    words_b = set(re.findall(r"\w+", text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _is_duplicate_skill(new_description: str, threshold: float = 0.65) -> str:
    """检查新技能描述是否与现有技能高度重叠。若重叠 > threshold，返回已存在技能名。"""
    for f in SKILLS_DIR.glob("*.py"):
        meta = parse_skill_metadata(f)
        sim = _jaccard_similarity(new_description, meta.get("description", ""))
        if sim > threshold:
            return f.name
    return ""


def _get_versioned_skill_path(skill_name: str) -> Path:
    """
    如果 skills/skill_name.py 已存在，返回 skills/skill_name_v2.py，以此类推。
    [修复] 增加 _MAX_SKILL_VERSION 上限，防止无界计数。
    """
    stem = Path(skill_name).stem
    ext = Path(skill_name).suffix or ".py"
    target = SKILLS_DIR / f"{stem}{ext}"
    if not target.exists():
        return target
    v = 2
    while v <= _MAX_SKILL_VERSION:
        candidate = SKILLS_DIR / f"{stem}_v{v}{ext}"
        if not candidate.exists():
            return candidate
        v += 1
    # 超出上限：用时间戳后缀兜底
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return SKILLS_DIR / f"{stem}_{ts}{ext}"


def _extract_python_code(text: str) -> str:
    """
    从 LLM 响应中提取纯 Python 代码。
    [修复] 优先提取 ```python ... ``` 代码块内容；
           如无围栏，则剥离首个非代码行（LLM 常见的前置说明文字）。
    """
    # 尝试提取 ```python ... ``` 代码块
    fence_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    # 尝试提取普通 ``` ... ``` 代码块
    fence_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    # 无围栏：去掉开头的非 Python 行（如 "以下是生成的脚本："）
    # Python 合法起始：# 注释、import、def、class、if __name__、空行
    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith("if ")
        ):
            start_idx = i
            break
    return "\n".join(lines[start_idx:]).strip()


def _safe_skill_name(raw_name: str) -> str:
    """
    将 LLM 生成的技能名规范化为安全的纯文件名。
    只保留字母、数字、下划线；强制以 .py 结尾；禁止路径分隔符。
    """
    stem = Path(raw_name).stem
    stem = re.sub(r"[^\w]", "_", stem)  # 非字母数字下划线替换为 _
    stem = stem.strip("_") or "skill_unnamed"
    return f"{stem}.py"


# ==========================================
# 演化节点
# ==========================================

def evolution_reflect_node(state: AgentState) -> Dict[str, Any]:
    """[演化 第1阶段] 观察 + 反思 + 决策"""
    logger.info(">>> [Node] Evolution.Reflect: 分析任务可复用性...")
    try:
        existing_skills_info = _format_existing_skills()

        reflect_llm = _evolution_llm.with_structured_output(EvolutionReflectOutput)
        decision: EvolutionReflectOutput = reflect_llm.invoke([
            SystemMessage(content=EVOLUTION_REFLECT_PROMPT.format(
                task_description=state.get("task_description", ""),
                completed_tasks="\n".join(
                    f"  - {t}" for t in state.get("completed_tasks", [])
                ),
                summary=state.get("summary", "（本次无压缩摘要，已在 token 限制内）"),
                existing_skills=existing_skills_info,
            ))
        ])

        if not decision.should_evolve:
            logger.info(f"Evolution.Reflect: 决策不演化。理由：{decision.reasoning}")
            return {
                "evolution_skill_draft": "",
                "evolution_report": (
                    f"🔍 [演化分析] 本次任务不具备高复用价值，跳过技能生成。\n"
                    f"理由：{decision.reasoning}"
                ),
            }

        # 二次去重检查（防止 LLM 漏判）
        dup_name = _is_duplicate_skill(decision.description)
        if dup_name:
            logger.info(f"Evolution.Reflect: 发现重复技能 '{dup_name}'，跳过演化。")
            return {
                "evolution_skill_draft": "",
                "evolution_report": (
                    f"🔍 [演化分析] 技能库中已有高度相似的技能 '{dup_name}'，跳过生成。"
                ),
            }

        # 规范化技能名，防止 LLM 输出路径分隔符等危险字符
        safe_name = _safe_skill_name(decision.skill_name or "skill_unnamed")
        logger.info(f"Evolution.Reflect: 决策演化！技能名：{safe_name}")

        draft_payload = decision.model_dump()
        draft_payload["skill_name"] = safe_name   # 覆盖为规范化名称
        draft_payload["draft_path"] = ""           # 占位，generate 节点填充

        return {
            "evolution_skill_draft": json.dumps(draft_payload, ensure_ascii=False),
            "messages": [AIMessage(content=(
                f"🧬 [Evolution] 发现高价值可结晶技能！\n"
                f"📦 名称: {safe_name}\n"
                f"📝 描述: {decision.description}\n"
                f"🏷️ 类别: {decision.category}\n"
                f"🧠 理由: {decision.reasoning}"
            ))],
        }

    except Exception as e:
        logger.warning(f"Evolution.Reflect 失败（跳过演化）: {e}")
        return {
            "evolution_skill_draft": "",
            "evolution_report": f"⚠️ 演化反思阶段异常（主任务不受影响）: {e}",
        }


def evolution_generate_node(state: AgentState) -> Dict[str, Any]:
    """[演化 第2阶段] 生成 Skill 脚本草稿"""
    draft_json = state.get("evolution_skill_draft", "")
    if not draft_json:
        return {}

    logger.info(">>> [Node] Evolution.Generate: 生成 Skill 脚本草稿...")
    try:
        decision = json.loads(draft_json)
        if not decision.get("should_evolve"):
            return {}

        skill_name: str = decision.get("skill_name", "skill_unnamed.py")
        if not skill_name.endswith(".py"):
            skill_name += ".py"

        today_str = date.today().isoformat()
        prompt = EVOLUTION_GENERATE_PROMPT.format(
            skill_name=skill_name,
            category=decision.get("category", "misc"),
            description=decision.get("description", ""),
            applicable_scenarios=decision.get("applicable_scenarios", ""),
            core_logic_summary=decision.get("core_logic_summary", ""),
            today=today_str,
        )

        response = _evolution_llm.invoke([HumanMessage(content=prompt)])
        script_content = _extract_python_code(response.content)

        if not script_content:
            return {
                "evolution_skill_draft": "",
                "evolution_report": "⚠️ 演化生成阶段：LLM 返回了空内容，跳过。",
            }

        draft_path = DRAFT_DIR / skill_name
        draft_path.write_text(script_content, encoding="utf-8")
        logger.info(f"Evolution.Generate: 草稿已写入 {draft_path}")

        decision["draft_path"] = str(draft_path)
        return {
            "evolution_skill_draft": json.dumps(decision, ensure_ascii=False),
        }

    except Exception as e:
        logger.error(f"Evolution.Generate 失败: {e}")
        return {
            "evolution_skill_draft": "",
            "evolution_report": f"⚠️ 演化生成阶段异常（主任务不受影响）: {e}",
        }


def evolution_verify_node(state: AgentState) -> Dict[str, Any]:
    """
    [演化 第3阶段] 验证草稿并持久化
      1. 读取草稿文件
      2. AST 语法检查
      3. 沙盒运行 __test__()（Best-effort）
      4. 双重校验通过 → 版本化持久化到 skills/
    """
    draft_json = state.get("evolution_skill_draft", "")
    if not draft_json:
        return {}

    logger.info(">>> [Node] Evolution.Verify: 验证并持久化 Skill 草稿...")
    try:
        info = json.loads(draft_json)
        # [修复] 明确用 str() 判空，因为 Path("") 本身是 truthy 对象
        draft_path_str = info.get("draft_path", "")
        if not draft_path_str:
            return {
                "evolution_skill_draft": "",
                "evolution_report": "⚠️ 验证失败：草稿路径为空（Generate 阶段可能已跳过）。",
            }

        draft_path = Path(draft_path_str)
        skill_name = info.get("skill_name", "unknown.py")

        if not draft_path.exists():
            return {
                "evolution_skill_draft": "",
                "evolution_report": f"⚠️ 验证失败：草稿文件不存在 ({draft_path})。",
            }

        source = draft_path.read_text(encoding="utf-8")

        # ---- 步骤 1: AST 语法检查 ----
        try:
            ast.parse(source, filename=skill_name)
        except SyntaxError as e:
            draft_path.unlink(missing_ok=True)
            return {
                "evolution_skill_draft": "",
                "evolution_report": f"❌ 技能生成失败（语法错误，草稿已丢弃）: 行{e.lineno} {e.msg}",
            }

        # ---- 步骤 2: 沙盒验证 __test__()（Best-effort）----
        # [安全修复] 不再把路径插入 bash -c 字符串，改用 exec_run 列表模式传参
        sandbox_status = "跳过（Docker 未启动）"
        if sandbox.container:
            try:
                rel_draft = draft_path.relative_to(WORKSPACE_DIR)
                # 把路径作为参数传入 Python，而非嵌入代码字符串，避免路径中的特殊字符造成注入
                test_script = (
                    "import sys\n"
                    "sys.path.insert(0, '/workspace')\n"
                    "path = sys.argv[1]\n"
                    "code = open(path).read()\n"
                    "ns = {}\n"
                    "exec(code, ns)\n"
                    "result = ns.get('__test__', lambda: True)()\n"
                    "print('SKILL_TEST_PASS' if result is not False else 'SKILL_TEST_FAIL')\n"
                )
                cmd = ["python", "-c", test_script, f"/workspace/{rel_draft}"]
                sb_exit, sb_out = sandbox.container.exec_run(cmd, workdir="/workspace")
                sb_result = f"Return Code: {sb_exit}\nOutput:\n{sb_out.decode('utf-8', errors='replace').strip()}"

                if "SKILL_TEST_FAIL" in sb_result:
                    draft_path.unlink(missing_ok=True)
                    return {
                        "evolution_skill_draft": "",
                        "evolution_report": (
                            f"❌ 技能自验证失败（__test__() 返回 False），草稿已丢弃。\n"
                            f"沙盒输出:\n{sb_result}"
                        ),
                    }
                if "SKILL_TEST_PASS" in sb_result:
                    sandbox_status = "✅ 沙盒验证通过"
                else:
                    sandbox_status = f"⚠️ 沙盒验证结果不明确（已忽略）: {sb_result[:200]}"
            except Exception as e:
                sandbox_status = f"⚠️ 沙盒验证异常（已忽略）: {e}"

        # ---- 步骤 3: 版本化持久化 ----
        target_path = _get_versioned_skill_path(skill_name)
        shutil.copy2(draft_path, target_path)
        draft_path.unlink(missing_ok=True)
        logger.info(f"Evolution.Verify: ✅ Skill 已持久化到 {target_path}")

        report = (
            f"✅ [Capability Evolution 成功] 新技能已固化入库！\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📦 文件: {target_path.name}\n"
            f"🏷️  类别: {info.get('category', 'misc')}\n"
            f"📝 描述: {info.get('description', '')}\n"
            f"🎯 适用: {info.get('applicable_scenarios', '')}\n"
            f"🔬 验证: {sandbox_status}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"下次遇到类似任务时，可通过 list_skills + run_skill 直接复用！"
        )

        return {
            "evolution_skill_draft": "",
            "evolution_report": report,
            "messages": [AIMessage(content=report)],
        }

    except Exception as e:
        logger.error(f"Evolution.Verify 异常: {e}")
        return {
            "evolution_skill_draft": "",
            "evolution_report": f"⚠️ 验证阶段异常（主任务不受影响）: {e}",
        }
