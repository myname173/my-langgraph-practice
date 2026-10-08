# src/agent/multimedia/llm_cognition.py
"""LLM 认知层（LLM Cognition Layer）
=================================

把原先纯规则驱动的「理解 / 创作 / 语义判断」环节升级为大模型驱动，
规则层**降级**为两件事：① 输出校验与修复；② 无 LLM 时的兜底路径。

设计原则（与项目既有约定一致）：
    - **绝不阻断**：任何异常 / 超时 / JSON 解析失败 → 返回 None，调用方回退规则实现。
    - **可开关**：每个环节独立 env 开关，默认全开；置 0 即回到改造前的纯规则行为。
    - **闸口不动**：质量闸门（cinematic_critic 的 PASS/FAIL）仍由规则决定，
      LLM 只作为「建议层」提供语义问题清单，不参与 pass_gates 判定。
    - **零新增依赖**：全部经 tools/text_llm.call_llm 走既有 fallback 链
      （DashScope → SiliconFlow → env 注入通道），自带看门狗超时与多模型降级。

覆盖环节：
    Phase 1  classify_intent_llm    场景意图理解（规则 → LLM 结构化分类）
    Phase 1  detect_genres_llm      全局风格识别（关键词匹配 → LLM 语义识别）
    Phase 2  generate_shot_plan_llm 镜头计划创作（16 模板硬选 → LLM 创作 + 规则校验）
    Phase 4  semantic_critic_llm    语义级问题检测（仅建议层，不参与闸门）
    Phase 5  studio_agents_llm      三专家真·LLM 协作与辩论（规则伪辩论 → LLM 协商）

环境变量：
    MULTIMEDIA_LLM_COGNITION    总开关，默认 1；置 0 一次性关闭全部 LLM 认知层
    MULTIMEDIA_LLM_INTENT       Phase 1 意图 + 风格识别，默认 1
    MULTIMEDIA_LLM_SHOT_PLAN    Phase 2 镜头计划创作，默认 1
    MULTIMEDIA_LLM_STUDIO       Phase 5 三专家协作，默认 1
    MULTIMEDIA_LLM_SEMANTIC_CRITIC  Phase 4 语义建议层，默认 1
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .tools.text_llm import call_llm


# ============================================================
# 0. 开关
# ============================================================

def _flag(name: str, default: str = "1") -> bool:
    return (os.getenv(name, default) or "").strip().lower() not in ("0", "false", "no", "off", "")


def llm_cognition_enabled() -> bool:
    """总开关。

    语义：
      - 显式 "0"/false/no/off → 关闭
      - 显式 "1"/true/yes/on  → 开启（即使在 pytest 下也强制开启）
      - 未设置：默认开启；但**在 pytest 下默认关闭**，避免既有单测意外触发真实网络调用。
    """
    v = (os.getenv("MULTIMEDIA_LLM_COGNITION", "") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    # 未显式设置：pytest 环境默认关闭，保护既有测试套件
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return True


def llm_enabled(layer: str) -> bool:
    """分环节开关。layer ∈ {intent, shot_plan, studio, semantic_critic}。"""
    if not llm_cognition_enabled():
        return False
    _map = {
        "intent": "MULTIMEDIA_LLM_INTENT",
        "shot_plan": "MULTIMEDIA_LLM_SHOT_PLAN",
        "studio": "MULTIMEDIA_LLM_STUDIO",
        "semantic_critic": "MULTIMEDIA_LLM_SEMANTIC_CRITIC",
    }
    env_name = _map.get(layer)
    if env_name is None:
        return False
    return _flag(env_name, "1")


# ============================================================
# 1. 健壮的 JSON 解析（不依赖 nodes.common，规避循环导入）
# ============================================================

def _extract_json(text: str) -> Optional[Any]:
    """从 LLM 输出中抽取第一个合法 JSON 对象/数组。

    容忍 markdown 代码围栏、前后解释文字、尾随逗号。
    """
    if not text:
        return None
    s = text.strip()

    # 去掉 ```json ... ``` 围栏
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    # 直接解析
    try:
        return json.loads(s)
    except Exception:
        pass

    # 截取第一个出现的 { ... } 或 [ ... ] 平衡区间。
    # 关键：顶层类型必须由"最先出现"的那个分隔符决定，不能固定先试 {。
    # 反面案例（实测）：对象被 max_tokens 截断后永远配不平，若退而试 [，
    # 会把输出里某个数组片段（如 "modifications": ["..."]）配平成 list 返回，
    # 用"解析出个 list"掩盖了"其实被截断"的事实，排障时极难定位。
    _cands = [(s.find("{"), "{", "}"), (s.find("["), "[", "]")]
    _cands = [c for c in _cands if c[0] >= 0]
    _cands.sort(key=lambda c: c[0])
    for start, open_ch, close_ch in _cands:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    chunk = s[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        # 尾随逗号容错
                        try:
                            return json.loads(re.sub(r",(\s*[}\]])", r"\1", chunk))
                        except Exception:
                            break
    return None


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except Exception:
        return default
    if f != f:  # NaN
        return default
    return f


def _as_list_of_str(v: Any) -> List[str]:
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        sx = str(x).strip() if x is not None else ""
        if sx and sx not in out:
            out.append(sx)
    return out


# ============================================================
# 2. Phase 1 — 场景意图理解
# ============================================================

_INTENT_PROMPT = """你是一位资深电影分镜师。请判断下面这个场景在整支宣传片里承担哪种**戏剧意图**。

可选意图（必须从中选一个，只能输出英文 key）：
{intent_menu}

判断要点：
- world_building        ：建立世界观 / 环境展示，主体是空间与氛围
- character_introduction：角色登场，主体是人物形象与身份
- action                ：动作冲突 / 打斗 / 追逐，主体是运动与事件
- emotional             ：情绪表达 / 静场 / 内心戏，主体是感受
- dialogue              ：对话 / 交锋，主体是人物之间的交流
- montage               ：快速剪辑的片段组接，强调节奏与并列信息
- reveal                ：反转 / 揭晓 / 高光揭示，强调"意料之外"

场景剧本：
\"\"\"{script}\"\"\"

全局视觉设定：
\"\"\"{global_setting}\"\"\"

只输出 JSON，不要任何解释文字：
{{"intent": "<key>", "confidence": 0.0-1.0, "reason": "<一句话中文理由>"}}"""


def classify_intent_llm(
    script: str,
    global_setting: str,
    intent_keys: List[str],
) -> Optional[Dict[str, Any]]:
    """LLM 场景意图分类。失败返回 None（调用方回退规则）。"""
    if not llm_enabled("intent") or not intent_keys:
        return None

    menu = _intent_menu_lines(intent_keys)
    prompt = _INTENT_PROMPT.format(
        intent_menu=menu,
        script=(script or "")[:1200],
        global_setting=(global_setting or "")[:600],
    )
    try:
        raw = call_llm(prompt, role_name="场景意图分析", temperature=0.2, max_tokens=384)
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return None
        intent = str(data.get("intent", "")).strip()
        # 归一化：容忍大小写 / 前后空白 / 代码风格写法
        _norm = {k.lower().replace("-", "_"): k for k in intent_keys}
        intent = _norm.get(intent.lower().replace("-", "_"), intent)
        if intent not in intent_keys:
            print(f"    [!] LLM 意图 '{data.get('intent')}' 不在合法集合，回退规则分类")
            return None
        return {
            "intent": intent,
            "confidence": _as_float(data.get("confidence"), 0.7),
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception as e:  # noqa: BLE001
        print(f"    [!] LLM 意图分类失败（{type(e).__name__}: {str(e)[:80]}），回退规则")
        return None


def _intent_menu_lines(intent_keys: List[str]) -> str:
    _zh = {
        "world_building": "建立世界观 / 环境展示",
        "character_introduction": "角色登场",
        "action": "动作冲突",
        "emotional": "情绪表达 / 静场",
        "dialogue": "对话 / 交锋",
        "montage": "快速剪辑组接",
        "reveal": "反转 / 揭晓",
    }
    return "\n".join(f"- {k}：{_zh.get(k, '')}".rstrip("：") for k in intent_keys)


# ============================================================
# 3. Phase 1 — 全局风格识别
# ============================================================

_GENRE_PROMPT = """你是一位电影美术指导。请从下面这段全局视觉设定中，识别出**适用的风格标签**。

可选风格标签（可多选，只能输出这些英文 key；若一个都不适用则输出空数组）：
{genre_menu}

规则：
- 只选**在设定里有明确依据**的风格，不要臆测叠加。
- 一个设定同时具备多种风格特质时可多选（例：赛博朋克 + 废土）。

全局视觉设定：
\"\"\"{global_setting}\"\"\"

只输出 JSON：
{{"genres": ["<key>", ...], "reason": "<一句话中文理由>"}}"""


def detect_genres_llm(
    global_setting: str,
    genre_keys: List[str],
) -> Optional[List[str]]:
    """LLM 全局风格识别。失败返回 None（调用方回退关键词匹配）。"""
    if not llm_enabled("intent") or not genre_keys:
        return None

    prompt = _GENRE_PROMPT.format(
        genre_menu="\n".join(f"- {k}" for k in genre_keys),
        global_setting=(global_setting or "")[:900],
    )
    try:
        raw = call_llm(prompt, role_name="风格识别", temperature=0.2, max_tokens=384)
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return None
        genres = _as_list_of_str(data.get("genres"))
        _norm = {k.lower().replace("-", "_"): k for k in genre_keys}
        out = []
        for g in genres:
            gk = _norm.get(g.lower().replace("-", "_"))
            if gk and gk not in out:
                out.append(gk)
        # 空数组是合法答案（明确表示"无匹配风格"），区别于解析失败
        return out
    except Exception as e:  # noqa: BLE001
        print(f"    [!] LLM 风格识别失败（{type(e).__name__}: {str(e)[:80]}），回退关键词匹配")
        return None


# ============================================================
# 4. Phase 2 — 镜头计划创作
# ============================================================

_SHOT_PLAN_PROMPT = """你是一位好莱坞级动作片摄影指导 + 剪辑师，正在为下面这个场景设计**分镜拍摄方案**。

要求：设计恰好 {n_shots} 个镜头，构成一条有起伏的视觉序列。

场景剧本：
\"\"\"{script}\"\"\"

场景意图（Phase 1 结论）：{intent}
节奏阶段：{pacing_phase}
环境光（昼夜/天气）：{ambient}
全局风格：{genres}

可用的摄影词汇（**建议**从中挑选，保证下游运镜/转场规则能识别；也可在合理时自创更贴切的英文描述）：
- 景别 type：{types}
- 运镜 movement：{movements}
- 机位角度 angle：{angles}
- 焦段 lens：{lenses}
- 朝向 orientation：{orientations}
- 追踪方式 tracking：{trackings}
- 情绪 emotion：{emotions}

创作约束：
1. 第 1 镜建立空间/情境，最后一镜收束；高潮（hero）镜头应放在序列后段。
2. 相邻镜头的景别与角度必须拉开差异，禁止连续同景别。
3. 情绪（emotion）要形成一条曲线，不要平铺。
4. pacing_weight 是剪辑权重 0.0–1.0，高权重=短促快剪，低权重=舒缓长镜。
5. 每镜的 focus_category 从 environment / character / action / detail 中选一个。

只输出 JSON，不要解释文字：
{{
  "shots": [
    {{
      "shot_name": "<英文短名，如 establish / impact / reveal>",
      "camera": {{"type": "...", "movement": "...", "angle": "...", "lens": "...", "orientation": "...", "tracking": "..."}},
      "lighting": {{"setup": "<英文光影描述>", "contrast": "low|medium|high|extreme"}},
      "emotion": "<emotion key>",
      "pacing_weight": 0.0,
      "focus_category": "environment|character|action|detail"
    }}
  ],
  "hero_shot_index": 0,
  "end_shot_index": 0,
  "rationale": "<一句话中文说明整体设计思路>"
}}"""


def generate_shot_plan_llm(
    script: str,
    visual_context: Dict[str, Any],
    n_shots: int,
    vocab: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    """LLM 镜头计划创作。

    Returns（校验通过时）:
        {"shots": [...], "hero_shot_index": int|None, "end_shot_index": int|None,
         "rationale": str}
    失败 / 校验不通过 → None（调用方回退模板硬选）。
    """
    if not llm_enabled("shot_plan") or n_shots <= 0:
        return None

    def _j(v):
        return "、".join(v) if v else "（不限）"

    prompt = _SHOT_PLAN_PROMPT.format(
        n_shots=n_shots,
        script=(script or "")[:1500],
        intent=visual_context.get("intent", "world_building"),
        pacing_phase=visual_context.get("pacing_phase", "opening"),
        ambient=visual_context.get("ambient_context") or "未指定",
        genres="、".join(visual_context.get("genres") or []) or "未指定",
        types=_j(vocab.get("types", [])),
        movements=_j(vocab.get("movements", [])),
        angles=_j(vocab.get("angles", [])),
        lenses=_j(vocab.get("lenses", [])),
        orientations=_j(vocab.get("orientations", [])),
        trackings=_j(vocab.get("trackings", [])),
        emotions=_j(vocab.get("emotions", [])),
    )
    try:
        raw = call_llm(prompt, role_name="分镜创作", temperature=0.75, max_tokens=3200)
        data = _extract_json(raw)
        if not isinstance(data, dict):
            print("    [!] LLM 分镜创作未返回合法 JSON，回退模板")
            return None

        shots = data.get("shots")
        if not isinstance(shots, list) or not shots:
            print("    [!] LLM 分镜创作 shots 为空，回退模板")
            return None

        valid = []
        for s in shots:
            if not isinstance(s, dict):
                continue
            cam = s.get("camera")
            if not isinstance(cam, dict) or not str(cam.get("type", "")).strip():
                continue
            light = s.get("lighting")
            if not isinstance(light, dict):
                light = {"setup": "", "contrast": "medium"}
            contrast = str(light.get("contrast", "medium")).strip().lower()
            if contrast not in ("low", "medium", "high", "extreme"):
                contrast = "medium"
            valid.append({
                "shot_name": str(s.get("shot_name") or f"shot{len(valid)+1}").strip(),
                "camera": {k: str(cam.get(k) or "").strip() for k in
                           ("type", "movement", "angle", "lens", "orientation", "tracking")},
                "lighting": {"setup": str(light.get("setup") or "").strip(), "contrast": contrast},
                "emotion": str(s.get("emotion") or "").strip() or "presence",
                "pacing_weight": max(0.0, min(1.0, _as_float(s.get("pacing_weight"), 0.25))),
                "focus_category": str(s.get("focus_category") or "character").strip(),
            })

        if len(valid) != n_shots:
            print(f"    [!] LLM 分镜数量 {len(valid)} ≠ 期望 {n_shots}，回退模板以保证结构完整")
            return None

        hero = data.get("hero_shot_index")
        end = data.get("end_shot_index")
        hero_i = int(hero) if isinstance(hero, (int, float)) and 0 <= int(hero) < n_shots else None
        end_i = int(end) if isinstance(end, (int, float)) and 0 <= int(end) < n_shots else None

        return {
            "shots": valid,
            "hero_shot_index": hero_i,
            "end_shot_index": end_i,
            "rationale": str(data.get("rationale", ""))[:300],
        }
    except Exception as e:  # noqa: BLE001
        print(f"    [!] LLM 分镜创作失败（{type(e).__name__}: {str(e)[:80]}），回退模板")
        return None


# ============================================================
# 5. Phase 5 — 三专家真·LLM 协作与辩论
# ============================================================

_STUDIO_PROMPT = """你现在同时扮演三位电影主创，对同一份分镜方案做联席评审与协商，并把分歧**辩到收敛**。

【席位 1｜摄影指导】审查镜头系统的摄影语言：焦距与景别是否服务叙事、运镜是否有动机、机位角度的叙事功能、是否有跳跃剪辑、与上一场次的风格是否连贯。
【席位 2｜灯光指导】审查光影：氛围光照是否匹配情绪、对比度是否有进程（不是一条直线）、大气层次、光照连续性、与上一场次的一致性。
【席位 3｜剪辑指导】审查节奏：转场选择是否匹配能量差、pacing_weight 是否符合当前节奏阶段、情绪弧是否平滑、是否有视觉疲劳（连续高强镜头）、与上一场次衔接。

协商规则（这是**辩论**，不是各说各话）：
- 每位专家先给出自己的意见（concerns）与修改建议。
- 当两位专家的建议相互冲突时（例：摄影要手持强运动、剪辑要稳定长镜），必须在 debate 里明确指出冲突双方，并给出**双方都能接受的折中结论**。
- 每位专家的 consistency_score / pacing_score 是 0–1 的自我评估。
- 剪辑指导还要给出 refined_transitions：逐条对应输入连接（长度必须一致）的转场类型。

【输出长度约束（必须遵守）】
- modifications 每条不超过 25 字，每位专家最多 3 条。
- agent_concerns 每位专家最多 2 条，concern / suggestion 各不超过 30 字。
- debate 最多 3 条，resolution 不超过 40 字。
- summary 不超过 60 字。
- 目标：整体输出控制在 2500 token 以内，避免因过长而被截断导致整个 JSON 作废。

输入方案：
shot_plan（JSON）：{shot_plan}
sequence_graph（JSON）：{sequence_graph}
上一场次风格（可能为空）：{prev_style}

只输出 JSON：
{{
  "cinematography": {{"modifications": ["..."], "consistency_score": 0.0, "agent_concerns": [{{"type":"...","concern":"...","suggestion":"..."}}]}},
  "lighting": {{"modifications": ["..."], "consistency_score": 0.0, "agent_concerns": [{{"type":"...","concern":"...","suggestion":"..."}}]}},
  "editing": {{"modifications": ["..."], "pacing_score": 0.0, "refined_transitions": ["..."], "agent_concerns": [{{"type":"...","concern":"...","suggestion":"..."}}]}},
  "debate": [
    {{"from_agent": "cinematography|lighting|editor", "to_agent": "cinematography|lighting|editor",
      "conflict_type": "...", "concern": "...", "suggestion": "...", "resolution": "<折中结论>"}}
  ],
  "summary": "<一句话中文总结>"
}}"""


def _compact(obj: Any, limit: int = 6000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "...(截断)"


def studio_agents_llm(
    shot_plan: Dict[str, Any],
    sequence_graph: Dict[str, Any],
    visual_context: Dict[str, Any],
    prev_style: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """三专家 LLM 联席评审 + 辩论。失败返回 None（调用方回退规则检测）。"""
    if not llm_enabled("studio"):
        return None

    # 只把评审必需的字段喂给 LLM，控制 token 与噪声
    _shots_brief = [
        {
            "i": i,
            "name": s.get("shot_name"),
            "camera": s.get("camera"),
            "lighting": s.get("lighting"),
            "emotion": s.get("emotion"),
            "pacing_weight": s.get("pacing_weight"),
        }
        for i, s in enumerate((shot_plan or {}).get("shots", []) or [])
    ]
    _graph_brief = {
        "connections": [
            {"from": c.get("from_name"), "to": c.get("to_name"), "transition": c.get("transition")}
            for c in ((sequence_graph or {}).get("connections", []) or [])
        ],
        "emotional_arc": (sequence_graph or {}).get("emotional_arc"),
    }

    prompt = _STUDIO_PROMPT.format(
        shot_plan=_compact({
            "intent": (shot_plan or {}).get("intent"),
            "pacing_phase": (shot_plan or {}).get("pacing_phase"),
            "hero_shot_index": (shot_plan or {}).get("hero_shot_index"),
            "shots": _shots_brief,
        }),
        sequence_graph=_compact(_graph_brief, 3000),
        prev_style=_compact(prev_style, 1500) if prev_style else "（无）",
    )
    try:
        raw = call_llm(prompt, role_name="电影工作室", temperature=0.6, max_tokens=4096)
        data = _extract_json(raw)
        if not isinstance(data, dict):
            print("    [!] LLM 工作室协作未返回合法 JSON，回退规则协商")
            return None

        def _agent_block(node: Any, score_key: str) -> Dict[str, Any]:
            node = node if isinstance(node, dict) else {}
            concerns = []
            for c in (node.get("agent_concerns") or []):
                if isinstance(c, dict):
                    concerns.append({
                        "agent": node.get("_agent", ""),
                        "type": str(c.get("type", "general")),
                        "concern": str(c.get("concern", "")),
                        "suggestion": str(c.get("suggestion", "")),
                    })
            out = {
                "modifications": _as_list_of_str(node.get("modifications")),
                score_key: round(max(0.0, min(1.0, _as_float(node.get(score_key), 0.85))), 2),
                "agent_concerns": concerns,
            }
            return out

        cine = _agent_block(data.get("cinematography"), "consistency_score")
        light = _agent_block(data.get("lighting"), "consistency_score")
        edit = _agent_block(data.get("editing"), "pacing_score")
        edit["refined_transitions"] = _as_list_of_str((data.get("editing") or {}).get("refined_transitions"))

        debate = []
        for d in (data.get("debate") or []):
            if not isinstance(d, dict):
                continue
            debate.append({
                "from_agent": str(d.get("from_agent", "")),
                "to_agent": str(d.get("to_agent", "")),
                "conflict_type": str(d.get("conflict_type", "cross_domain")),
                "concern": str(d.get("concern", "")),
                "suggestion": str(d.get("suggestion", "")),
                "resolution": str(d.get("resolution", "")),
            })

        return {
            "cinematography": cine,
            "lighting": light,
            "editing": edit,
            "debate": debate,
            "summary": str(data.get("summary", ""))[:300],
        }
    except Exception as e:  # noqa: BLE001
        print(f"    [!] LLM 工作室协作失败（{type(e).__name__}: {str(e)[:80]}），回退规则协商")
        return None


# ============================================================
# 6. Phase 4 — 语义级问题检测（仅建议层）
# ============================================================

_SEMANTIC_CRITIC_PROMPT = """你是一位挑剔的电影评论家。规则评分器已经检查过**结构性问题**（镜头是否完整、角度是否多样、情绪是否推进），
你要做的是它做不了的事：**语义层面的判断**。

请针对下面的分镜方案，找出规则查不出来的问题，例如：
- 某个镜头的情绪与剧本内容是否相符（例：剧本在写悲伤，镜头却给了 epic triumph）
- 镜头顺序是否讲得通（因果是否连贯）
- prompt 语义与场景意图是否一致
- 是否存在与场景设定矛盾的视觉元素
- 整体是否真的讲出了这个故事

场景剧本：
\"\"\"{script}\"\"\"

内容类型：{content_type}

shotted 方案（JSON）：{shot_plan}
镜头连接（JSON）：{sequence_graph}

注意：
- **不要复述结构性问题**（如"镜头数不足""角度单一"），那些已由规则层负责。
- 只报你有把握的语义问题；没有就给空数组，不要凑数。
- 这是**建议**，不会直接拦截流程。

只输出 JSON：
{{
  "score": 0.0-1.0,
  "issues": [{{"severity": "low|medium|high", "type": "<英文短标签>", "desc": "<中文说明>"}}],
  "summary": "<一句话中文总评>"
}}"""


def semantic_critic_llm(
    shot_plan: Dict[str, Any],
    sequence_graph: Dict[str, Any],
    script: str,
    content_type: str,
) -> Optional[Dict[str, Any]]:
    """LLM 语义级问题检测（建议层，绝不参与闸门判定）。失败返回 None。"""
    if not llm_enabled("semantic_critic"):
        return None

    _shots_brief = [
        {"i": i, "name": s.get("shot_name"), "emotion": s.get("emotion"),
         "camera": s.get("camera", {}).get("type") if isinstance(s.get("camera"), dict) else None}
        for i, s in enumerate((shot_plan or {}).get("shots", []) or [])
    ]
    prompt = _SEMANTIC_CRITIC_PROMPT.format(
        script=(script or "")[:1200],
        content_type=content_type or "trailer",
        shot_plan=_compact(_shots_brief, 2500),
        sequence_graph=_compact({
            "connections": [
                {"from": c.get("from_name"), "to": c.get("to_name"), "transition": c.get("transition")}
                for c in ((sequence_graph or {}).get("connections", []) or [])
            ],
        }, 2000),
    )
    try:
        raw = call_llm(prompt, role_name="语义评审", temperature=0.4, max_tokens=1600)
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return None
        issues = []
        for it in (data.get("issues") or []):
            if not isinstance(it, dict):
                continue
            sev = str(it.get("severity", "low")).strip().lower()
            if sev not in ("low", "medium", "high"):
                sev = "low"
            issues.append({
                "severity": sev,
                "type": str(it.get("type", "semantic")),
                "desc": str(it.get("desc", "")),
            })
        return {
            "score": round(max(0.0, min(1.0, _as_float(data.get("score"), 0.7))), 2),
            "issues": issues,
            "summary": str(data.get("summary", ""))[:300],
            "source": "llm",
        }
    except Exception as e:  # noqa: BLE001
        print(f"    [!] LLM 语义评审失败（{type(e).__name__}: {str(e)[:80]}），跳过建议层")
        return None
