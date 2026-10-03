# src/agent/multimedia/review_flywheel.py
"""
审核飞轮（Review Flywheel）—— P1-6
=====================================

把每次 HITL（Human-in-the-Loop）人审决策结构化成可复用的 **(rejected, chosen)
偏好对**，落盘为 JSONL；攒够后离线蒸馏一个「自动审核员」，逐步把 7 道人审收敛为
1 道 final review。

架构评审原文（P1-6）：
  「每次 HITL 决策自动落一条结构化记录（镜头产物 + 决策 + 修改后 prompt），
   rewrite 天然构成 chosen/rejected 对；攒够后离线蒸馏一个『自动审核员』
   （小模型或规则+embedding 混合），先接管 image_review 这种模式固定的门控，
   把 7 道人审逐步收敛为 1 道 final review。」

设计原则（对齐 run_metrics.py 的工程约定）：
  - 零外部依赖：仅标准库；不引入任何第三方包，不影响现有流水线。
  - 零侵入：graph 侧仅在各审核闸口消费决策后追加一次调用，不改任何现有逻辑。
  - 绝不阻断：所有落盘/蒸馏路径 try/except 兜底，埋点失败只打印不抛。
  - 可开关：MULTIMEDIA_FLYWHEEL_ENABLED / MULTIMEDIA_FLYWHEEL_PATH 控制。

三层结构：
  1) 收集端  record_decision() —— 闸口消费决策时调用，自动配对 chosen/rejected
  2) 存储端  PreferencePair + JSONL（+ DPO/SFT 导出）
  3) 收敛端  suggest() —— 读取离线蒸馏出的 review_policy.json，影子模式给建议

偏好对的构成规则：
  - edit_prompt / 编辑后提交：rejected = 原产物，chosen = 人改后的产物（立即成对）
  - rewrite：先在 pending 里记 rejected；等该 (thread,gate,scene) 下一次
    approve 到来时，把「最终被通过的那一版」作为 chosen，配对成对
  - 直接 approve（无 pending）：不构对（没有偏好信号）
  - auto_mode 自动放行：不构对，但计入 auto 计数（供收敛率统计）
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

# ── 路径与开关 ────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

# 默认落到与 SWE 侧同一训练数据目录（workspace/_training_data），便于 LlamaFactory 统一消费
_DEFAULT_DIR = os.path.join(_PROJECT_ROOT, "workspace", "_training_data")

_ENABLED = os.getenv("MULTIMEDIA_FLYWHEEL_ENABLED", "1") != "0"
_DIR = os.getenv("MULTIMEDIA_FLYWHEEL_DIR", _DEFAULT_DIR)
_PATH = os.path.join(_DIR, "multimedia_review_pairs.jsonl")
_POLICY_PATH = os.getenv(
    "MULTIMEDIA_REVIEW_POLICY_PATH", os.path.join(_DIR, "review_policy.json")
)

SCHEMA = "multimedia.review_pair.v1"
POLICY_SCHEMA = "multimedia.review_policy.v1"

_APPROVE_ACTIONS = {"approve", "pass", "ok", "accept", "yes", "通过", "批准"}

_lock = threading.Lock()
# 待配对的 rewrite：key = f"{thread_id}|{gate}|{scene_index}" → pending record
_pending: Dict[str, Dict[str, Any]] = {}
# 影子模式开关：MULTIMEDIA_AUTO_REVIEW=shadow|off|<gate,g渲染>
_AUTO_REVIEW = os.getenv("MULTIMEDIA_AUTO_REVIEW", "off").strip().lower()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


# ---------------------------------------------------------------- 产物归一
def _artifact(kind: str, text: str = "", url: str = "", **meta: Any) -> Dict[str, Any]:
    """归一化一个「镜头产物」：kind ∈ prompt/image/video/script/end_frame。"""
    return {
        "kind": kind,
        "text": (text or "").strip(),
        "url": (url or "").strip(),
        "meta": {k: v for k, v in meta.items() if v is not None},
    }


def artifact_from_payload(gate: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """从审核卡的 payload 里抽出「当前被审核的产物」——即 rejected 候选。

    各闸口产物形态不同，这里做 gate → 产物的映射；抽不到则返回空 prompt 产物。
    """
    payload = payload or {}
    if gate in ("prompt_preview", "end_frame_prompt_preview"):
        # 生图前的 prompt 审核
        txt = payload.get("image_prompt") or payload.get("last_image_prompt") or ""
        return _artifact("prompt", text=txt)
    if gate == "image_review":
        return _artifact(
            "image",
            text=payload.get("image_prompt", ""),
            url=payload.get("image_url", ""),
            embedding_similarity=payload.get("embedding_similarity"),
            portrait_similarity=payload.get("portrait_similarity"),
            continuity_score=payload.get("continuity_score"),
        )
    if gate == "end_frame_review":
        return _artifact(
            "end_frame",
            text=payload.get("last_image_prompt", ""),
            url=payload.get("end_frame_url", "") or payload.get("image_url", ""),
        )
    if gate == "video_review":
        return _artifact(
            "video",
            text=payload.get("last_image_prompt", "") or payload.get("video_prompt", ""),
            url=payload.get("video_url", ""),
        )
    if gate == "draft_review":
        return _artifact("video", url=payload.get("movie_path", "") or payload.get("final_movie_path", ""))
    if gate == "showrunner_review":
        return _artifact("script", text=payload.get("global_setting", ""))
    # 兜底：把常见字段拼成文本产物
    return _artifact(
        "prompt",
        text=payload.get("image_prompt") or payload.get("last_image_prompt") or "",
        url=payload.get("image_url") or payload.get("video_url") or "",
    )


def _chosen_from_decision(gate: str, decision: Dict[str, Any], rejected: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从决策里抽出「人改后的产物」（chosen）。仅当人显式提交了编辑内容时才有值。"""
    decision = decision or {}
    # edit_prompt / 编辑后提交：决策携带修改后的 prompt
    for key in ("image_prompt", "last_image_prompt", "video_prompt", "prompt"):
        val = decision.get(key)
        if val is not None and str(val).strip() and str(val).strip() != rejected.get("text", ""):
            return _artifact(rejected.get("kind", "prompt"), text=str(val).strip(), url=rejected.get("url", ""))
    # 挑选候选（F-1 pick_candidate）：chosen 是被选中的那一版产物
    pick = decision.get("chosen_artifact")
    if isinstance(pick, dict) and (pick.get("url") or pick.get("text")):
        return _artifact(
            pick.get("kind", rejected.get("kind", "image")),
            text=pick.get("text", ""),
            url=pick.get("url", ""),
        )
    return None


# ---------------------------------------------------------------- 收集端
def _pkey(thread_id: str, gate: str, scene_index: Any) -> str:
    return f"{thread_id}|{gate}|{scene_index}"


def _emit(pair: Dict[str, Any]) -> None:
    """落盘一条偏好对（线程安全；失败静默）。"""
    if not _ENABLED:
        return
    try:
        os.makedirs(_DIR, exist_ok=True)
        with _lock:
            with open(_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    except Exception as e:  # 埋点绝不阻断主流程
        print(f"    [review_flywheel] 写入失败（已忽略）: {e}")


def record_decision(
    gate: str,
    decision: Dict[str, Any],
    *,
    thread_id: str = "",
    scene_index: Any = None,
    rejected: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    run_id: str = "",
) -> Optional[Dict[str, Any]]:
    """闸口消费决策时调用：自动配对并落盘偏好对。

    返回本次「新产生的偏好对」（未产生则为 None），便于单测断言。
    """
    if not _ENABLED:
        return None
    try:
        decision = decision or {}
        action = str(decision.get("action", "approve")).strip().lower()
        rejected = rejected or _artifact("prompt")
        ctx = dict(context or {})
        key = _pkey(thread_id, gate, scene_index)

        base = {
            "schema": SCHEMA,
            "created_at": _now_iso(),
            "run_id": run_id or "",
            "thread_id": thread_id or "",
            "gate": gate,
            "scene_index": scene_index,
            "decision": {
                "action": action,
                "fix_type": decision.get("fix_type"),
                "edited_by_human": bool(decision.get("image_prompt") is not None
                                        or decision.get("last_image_prompt") is not None
                                        or decision.get("chosen_artifact") is not None),
                "auto_approved": bool(decision.get("auto_approved")),
                "reason": str(decision.get("reason") or "").strip(),
            },
            "context": ctx,
        }

        # ① 自动放行：不构对，只清理 pending（供收敛率统计）
        if decision.get("auto_approved"):
            _pending.pop(key, None)
            return None

        # ② 编辑后提交：立即成对（rejected=原产物，chosen=人改产物）
        chosen = _chosen_from_decision(gate, decision, rejected)
        if chosen is not None:
            pair = dict(base)
            pair["pair_id"] = uuid.uuid4().hex[:12]
            pair["source"] = "edit"
            pair["rejected"] = rejected
            pair["chosen"] = chosen
            _pending.pop(key, None)
            _emit(pair)
            return pair

        # ③ rewrite：记 pending，等下一次 approve 时把「最终通过版」作为 chosen
        if action == "rewrite":
            _pending[key] = dict(base, rejected=rejected)
            return None

        # ④ approve：若此前有 pending（=重写后这次通过了），配对成对
        if action in _APPROVE_ACTIONS:
            pend = _pending.pop(key, None)
            if pend is not None:
                pair = dict(base)
                pair["pair_id"] = uuid.uuid4().hex[:12]
                pair["source"] = "rewrite"
                pair["rejected"] = pend.get("rejected", rejected)
                pair["chosen"] = rejected  # 重写后最终被通过的这一版
                # 保留首次 rewrite 的理由
                if pend.get("decision", {}).get("reason"):
                    pair["decision"]["reason"] = pend["decision"]["reason"]
                _emit(pair)
                return pair
            return None
        return None
    except Exception as e:  # 埋点绝不阻断主流程
        print(f"    [review_flywheel] record_decision 失败（已忽略）: {e}")
        return None


# ---------------------------------------------------------------- 读取 / 统计
def _read_pairs(path: Optional[str] = None) -> List[Dict[str, Any]]:
    p = path or _PATH
    out: List[Dict[str, Any]] = []
    if not os.path.exists(p):
        return out
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return out
    return out


def summarize(path: Optional[str] = None) -> Dict[str, Any]:
    """聚合统计：按 gate 统计配对数与来源分布，供前端 / CLI 展示飞轮进度。"""
    pairs = _read_pairs(path)
    by_gate: Dict[str, Dict[str, int]] = {}
    by_source: Dict[str, int] = {}
    for p in pairs:
        g = p.get("gate", "?")
        s = p.get("source", "?")
        by_gate.setdefault(g, {"total": 0, "edit": 0, "rewrite": 0})
        by_gate[g]["total"] += 1
        if s in ("edit", "rewrite"):
            by_gate[g][s] += 1
        by_source[s] = by_source.get(s, 0) + 1
    return {
        "schema": "multimedia.flywheel_stats.v1",
        "path": path or _PATH,
        "total_pairs": len(pairs),
        "by_gate": by_gate,
        "by_source": by_source,
        "threads": len({p.get("thread_id") for p in pairs if p.get("thread_id")}),
    }


# ---------------------------------------------------------------- 导出
def export_dpo(path: Optional[str] = None, out_path: Optional[str] = None) -> str:
    """导出为 LlamaFactory DPO 格式（prompt / chosen / rejected）。

    仅导出「文本可比」的产物（prompt 类）。图片/视频类偏好对跳过——它们需要
    视觉特征而非文本，留给 distill_reviewer 的规则侧消费。
    """
    pairs = _read_pairs(path)
    out_path = out_path or os.path.join(_DIR, "multimedia_review_dpo.jsonl")
    n = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            rej, cho = p.get("rejected", {}), p.get("chosen", {})
            if rej.get("kind") not in ("prompt", "script") or not rej.get("text"):
                continue
            if not cho.get("text"):
                continue
            rec = {
                "system": _DPO_SYSTEM,
                "instruction": _build_instruction(p),
                "input": "",
                "chosen": cho["text"],
                "rejected": rej["text"],
                "meta": {
                    "gate": p.get("gate"),
                    "scene_index": p.get("scene_index"),
                    "fix_type": (p.get("decision") or {}).get("fix_type"),
                    "reason": (p.get("decision") or {}).get("reason", ""),
                    "source": p.get("source"),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return out_path


_DPO_SYSTEM = (
    "你是影视生成流水线的 prompt 审核员。给定镜头剧本与全局设定，"
    "判断 image_prompt 是否达标；不达标则给出改进版。"
)


def _build_instruction(pair: Dict[str, Any]) -> str:
    ctx = pair.get("context") or {}
    sc = ctx.get("script") or ""
    idx = pair.get("scene_index")
    style = ctx.get("style_key") or ""
    parts = [f"镜头 {idx}：{sc}" if idx is not None else f"镜头剧本：{sc}"]
    if ctx.get("global_setting"):
        parts.append(f"全局设定：{ctx['global_setting']}")
    if style:
        parts.append(f"视觉风格：{style}")
    return "\n".join(parts)


def export_sft(path: Optional[str] = None, out_path: Optional[str] = None) -> str:
    """导出为 SFT 格式（只取 chosen 侧，供微调「会写好 prompt」的导演小模型）。"""
    pairs = _read_pairs(path)
    out_path = out_path or os.path.join(_DIR, "multimedia_review_sft.jsonl")
    n = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            cho = p.get("chosen", {})
            if cho.get("kind") not in ("prompt", "script") or not cho.get("text"):
                continue
            rec = {
                "system": _DPO_SYSTEM,
                "instruction": _build_instruction(p),
                "input": "",
                "output": cho["text"],
                "meta": {"gate": p.get("gate"), "source": p.get("source")},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return out_path


# ---------------------------------------------------------------- 收敛端（影子模式）
_POLICY_CACHE: Optional[Dict[str, Any]] = None


def load_policy(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """读取离线蒸馏产物 review_policy.json（不存在返回 None）。"""
    global _POLICY_CACHE
    p = path or _POLICY_PATH
    if _POLICY_CACHE is not None and path is None:
        return _POLICY_CACHE
    try:
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if path is None:
            _POLICY_CACHE = obj
        return obj
    except Exception:
        return None


def auto_review_enabled(gate: str) -> bool:
    """该闸口是否启用自动审核（默认 off）；支持 shadow（只建议不拦）与 all/on。"""
    if _AUTO_REVIEW in ("", "off", "0", "false"):
        return False
    if _AUTO_REVIEW in ("all", "on", "1", "true", "shadow"):
        return True
    return gate in {g.strip() for g in _AUTO_REVIEW.split(",") if g.strip()}


def suggest(gate: str, artifact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """影子模式建议：返回 {decision: pass|flag, confidence, reason, score}。

    仅用于记录建议做 A/B（不改主流程），由调用方决定是否采纳。policy 缺失返回 None。
    """
    try:
        policy = load_policy()
        if not policy:
            return None
        gp = (policy.get("gates") or {}).get(gate)
        if not gp:
            return None
        text = (artifact or {}).get("text", "") or ""
        score = 0.0
        hits: List[str] = []
        for tok, w in (gp.get("chosen_tokens") or {}).items():
            if tok and tok in text:
                score += float(w)
                hits.append(tok)
        for tok, w in (gp.get("rejected_tokens") or {}).items():
            if tok and tok in text:
                score -= float(w)
                hits.append(f"-{tok}")
        thr = float(gp.get("threshold", 0.0))
        decision = "pass" if score >= thr else "flag"
        conf = min(1.0, abs(score) / (abs(thr) + 1.0)) if thr else 0.0
        return {
            "decision": decision,
            "confidence": round(conf, 3),
            "score": round(score, 3),
            "threshold": thr,
            "hits": hits[:12],
            "reason": gp.get("reason", ""),
        }
    except Exception:
        return None


def record_suggestion(gate: str, artifact: Dict[str, Any], suggestion: Optional[Dict[str, Any]] = None) -> None:
    """把影子建议落到独立文件，供离线评估「自动审核员 vs 人审」的一致率。"""
    if not _ENABLED:
        return
    sug = suggestion if suggestion is not None else suggest(gate, artifact)
    if not sug:
        return
    try:
        os.makedirs(_DIR, exist_ok=True)
        rec = {
            "schema": "multimedia.review_suggestion.v1",
            "created_at": _now_iso(),
            "gate": gate,
            "artifact_kind": (artifact or {}).get("kind"),
            **sug,
        }
        with _lock:
            with open(os.path.join(_DIR, "multimedia_review_suggestions.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def reset_pending() -> None:
    """清空待配对缓存（测试用）。"""
    _pending.clear()


def pending_count() -> int:
    return len(_pending)


def paths() -> Dict[str, str]:
    return {"dir": _DIR, "pairs": _PATH, "policy": _POLICY_PATH}
