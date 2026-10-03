# src/agent/multimedia/tools/backend_health.py
"""后端熔断器 + 成本台账（S1 / 架构评审 P0-4）。

职责（单一、只读产物之外零副作用）：
1. **熔断器（Circuit Breaker）**：按 provider 记录连续失败次数，超过阈值进入
   冷却期（down），冷却期满自动半开重试。video_gen / image_gen 在派发前经
   :func:`provider_status` 查询，避免把请求浪费在已知不可用的后端上。
2. **成本台账（Cost Ledger）**：每次生成调用（成功/失败）追加一条 JSONL 记录
   （时间、thread、provider、类型、时延、成功与否），供前端成本仪表盘与
   预算治理消费。台账按 thread 隔离落盘 ``data/cost_ledger/<thread_id>.jsonl``，
   另有全局流 ``_global.jsonl``。

设计守则（与 qc.py 一致）：
- 只记录、不修改任何生成产物；
- 零第三方依赖（threading + json + pathlib）；
- 任何记录失败都静默降级（打印 WARN），绝不影响生成主链路。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()

# 熔断参数（env 可覆盖）
_FAILURE_THRESHOLD = int(os.getenv("BACKEND_FAILURE_THRESHOLD", "3"))
_COOLDOWN_SECONDS = float(os.getenv("BACKEND_COOLDOWN_SECONDS", "300"))

# provider -> 状态
#   {"consecutive_failures": int, "last_failure_ts": float, "state": "healthy"|"down"}
_BREAKERS: Dict[str, Dict[str, Any]] = {}

# 项目根目录：backend_health.py -> tools -> multimedia -> agent -> src -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LEDGER_DIR = _PROJECT_ROOT / "data" / "cost_ledger"


def _safe_thread_tag(thread_id: str) -> str:
    tag = "".join(c for c in (thread_id or "") if c.isalnum() or c in "-_")
    return tag or "_global"


def _ledger_file(thread_id: str) -> Path:
    return _LEDGER_DIR / f"{_safe_thread_tag(thread_id)}.jsonl"


def _status_unlocked(provider: str) -> str:
    """在已持有 _LOCK 的前提下返回状态（内部复用，避免同锁重入死锁）。

    语义与 provider_status 一致；含半开恢复副作用（冷却期满翻回 healthy）。
    """
    info = _BREAKERS.get(provider)
    if not info:
        return "unknown"
    if info["state"] == "down":
        elapsed = time.time() - info["last_failure_ts"]
        if elapsed >= _COOLDOWN_SECONDS:
            # 半开：冷却期满，允许重试；失败会再次进入 down
            info["state"] = "healthy"
            info["consecutive_failures"] = 0
            return "healthy"
        return "down"
    return "healthy"


def provider_status(provider: str) -> str:
    """返回 provider 当前状态：healthy / down / unknown。

    down = 连续失败达阈值且仍在冷却期；冷却期满自动恢复 healthy（半开重试）。
    注意：本函数自行加锁；已在 _LOCK 临界区内的代码请调用 _status_unlocked。
    """
    with _LOCK:
        return _status_unlocked(provider)


def record_result(
    provider: str,
    ok: bool,
    latency_s: float = 0.0,
    kind: str = "video",
    thread_id: str = "",
    meta: str = "",
) -> None:
    """记录一次后端调用结果：更新熔断器 + 追加台账。绝不抛异常。"""
    try:
        now = time.time()
        with _LOCK:
            info = _BREAKERS.setdefault(
                provider, {"consecutive_failures": 0, "last_failure_ts": 0.0, "state": "healthy"}
            )
            if ok:
                info["consecutive_failures"] = 0
                info["state"] = "healthy"
            else:
                info["consecutive_failures"] += 1
                info["last_failure_ts"] = now
                if info["consecutive_failures"] >= _FAILURE_THRESHOLD:
                    info["state"] = "down"
                    print(
                        f"    [熔断] {provider} 连续失败 {info['consecutive_failures']} 次，"
                        f"冷却 {_COOLDOWN_SECONDS:.0f}s 内跳过"
                    )

        entry = {
            "ts": round(now, 3),
            "provider": provider,
            "kind": kind,
            "ok": bool(ok),
            "latency_s": round(float(latency_s), 2),
            "thread_id": thread_id or "",
        }
        if meta:
            entry["meta"] = str(meta)[:300]

        _LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with open(_ledger_file(thread_id), "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if thread_id:
            with open(_ledger_file(""), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:  # noqa: BLE001 — 台账绝不拖垮主链路
        print(f"    [WARN] 成本台账记录失败（已忽略）: {e}")


def snapshot() -> Dict[str, Any]:
    """全部 provider 的熔断状态快照（供 /settings/health 接口）。"""
    with _LOCK:
        out: Dict[str, Any] = {}
        for provider, info in _BREAKERS.items():
            remaining = 0.0
            if info["state"] == "down":
                remaining = max(0.0, _COOLDOWN_SECONDS - (time.time() - info["last_failure_ts"]))
            out[provider] = {
                "state": _status_unlocked(provider),
                "consecutive_failures": info["consecutive_failures"],
                "cooldown_remaining_s": round(remaining, 1),
            }
        return out


def read_ledger(thread_id: str = "", limit: int = 500) -> List[Dict[str, Any]]:
    """读取某个 thread（或全局）的台账最近记录。"""
    path = _ledger_file(thread_id)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        tail = lines[-max(1, limit):]
        out = []
        for ln in tail:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] 台账读取失败: {e}")
        return []


def summarize_ledger(thread_id: str = "") -> Dict[str, Any]:
    """台账汇总：按 provider 统计成功/失败/累计生成时长/记录数。"""
    entries = read_ledger(thread_id)
    by_provider: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        p = e.get("provider", "unknown")
        agg = by_provider.setdefault(
            p, {"ok": 0, "fail": 0, "total_latency_s": 0.0, "count": 0}
        )
        agg["count"] += 1
        if e.get("ok"):
            agg["ok"] += 1
        else:
            agg["fail"] += 1
        agg["total_latency_s"] = round(agg["total_latency_s"] + float(e.get("latency_s") or 0), 2)
    return {
        "thread_id": thread_id or "_global",
        "total_calls": len(entries),
        "by_provider": by_provider,
        "failure_threshold": _FAILURE_THRESHOLD,
        "cooldown_seconds": _COOLDOWN_SECONDS,
    }
