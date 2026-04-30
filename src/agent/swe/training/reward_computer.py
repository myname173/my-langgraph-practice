# src/agent/swe/training/reward_computer.py
"""
Reward Computer
===============
从轨迹记录中计算标量奖励信号，用于 DPO/GRPO 训练数据标注。

设计思路：
  - 完全离线（不依赖 LLM 或网络），从 outcome 字段直接计算
  - 多维度加权：测试结果 + 效率 + 工具准确率
  - 支持 Process Reward（逐步奖励）扩展接口，当前用 Outcome Reward 实现

奖励分解：
  test_reward      [0.0, 1.0]  测试是否通过（最高权重）
  efficiency_bonus [-0.3, 0.3] 迭代效率（越少轮次越高）
  tool_accuracy    [0.0, 0.2]  工具调用成功率
  completion_bonus [0.0, 0.2]  任务步骤完成比例
  total            ∈ [-0.3, 1.7] → clamp 到 [-1.0, 1.0]
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from src.agent.swe.training.trajectory_logger import TrajectoryRecord

logger = logging.getLogger("SWE_RewardComputer")


# ==========================================
# 奖励权重配置
# ==========================================
REWARD_WEIGHTS = {
    "test_pass": 1.0,        # 测试通过是最高价值信号
    "efficiency": 0.3,       # 效率奖励权重
    "tool_accuracy": 0.2,    # 工具准确率权重
    "completion": 0.2,       # 任务完成率权重
}


@dataclass
class RewardBreakdown:
    test_reward: float = 0.0
    efficiency_bonus: float = 0.0
    tool_accuracy: float = 0.0
    completion_bonus: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "test_reward": round(self.test_reward, 4),
            "efficiency_bonus": round(self.efficiency_bonus, 4),
            "tool_accuracy": round(self.tool_accuracy, 4),
            "completion_bonus": round(self.completion_bonus, 4),
            "total": round(self.total, 4),
        }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_reward(record: TrajectoryRecord) -> RewardBreakdown:
    """
    计算单条轨迹的多维奖励。

    返回 RewardBreakdown，total 是最终归一化到 [-1, 1] 的标量奖励。
    """
    bd = RewardBreakdown()

    # ── 1. 测试奖励（最重要的信号） ────────────────────────
    if record.status == "success" and record.test_passed:
        bd.test_reward = REWARD_WEIGHTS["test_pass"]
    elif record.status == "success" and not record.test_passed:
        # 代码通过了 reviewer 但测试未明确通过 → 部分奖励
        bd.test_reward = REWARD_WEIGHTS["test_pass"] * 0.4
    elif record.status == "failed":
        bd.test_reward = -0.3
    else:
        # 未知状态：中性
        bd.test_reward = 0.0

    # ── 2. 效率奖励（激励 Agent 用更少轮次完成任务） ────────
    if record.max_iterations > 0 and record.status == "success":
        iteration_ratio = record.iteration_count / record.max_iterations
        # 用完 < 30% 的轮次 → 最高效率奖励
        # 用完 > 80% 的轮次 → 效率惩罚
        efficiency = 1.0 - iteration_ratio
        bd.efficiency_bonus = _clamp(
            (efficiency - 0.5) * REWARD_WEIGHTS["efficiency"],
            -REWARD_WEIGHTS["efficiency"],
            REWARD_WEIGHTS["efficiency"],
        )

    # ── 3. 工具调用准确率 ─────────────────────────────────
    steps = record.tool_call_steps
    if steps:
        success_rate = sum(1 for s in steps if s.success) / len(steps)
        bd.tool_accuracy = success_rate * REWARD_WEIGHTS["tool_accuracy"]

    # ── 4. 任务步骤完成率 ─────────────────────────────────
    total_steps = len(record.completed_tasks) + len(
        # 通过 summary 推断待办数量（粗估）
        [line for line in record.summary.split("\n") if "待处理" in line or "未完成" in line]
    )
    if total_steps > 0 and record.completed_tasks:
        completion_ratio = len(record.completed_tasks) / max(total_steps, len(record.completed_tasks))
        bd.completion_bonus = completion_ratio * REWARD_WEIGHTS["completion"]

    # ── 5. 归一化并 clamp ────────────────────────────────
    raw = bd.test_reward + bd.efficiency_bonus + bd.tool_accuracy + bd.completion_bonus
    bd.total = _clamp(raw, -1.0, 1.0)

    return bd


def annotate_trajectories(records: List[TrajectoryRecord]) -> List[TrajectoryRecord]:
    """
    批量计算奖励并回填到 TrajectoryRecord.reward 和 reward_breakdown。
    """
    annotated = []
    for rec in records:
        bd = compute_reward(rec)
        rec.reward = bd.total
        rec.reward_breakdown = bd.to_dict()
        annotated.append(rec)
        logger.debug(
            f"轨迹 {rec.trajectory_id[:8]}: "
            f"reward={bd.total:.3f} "
            f"(test={bd.test_reward:.2f}, eff={bd.efficiency_bonus:.2f}, "
            f"tool={bd.tool_accuracy:.2f}, comp={bd.completion_bonus:.2f})"
        )
    return annotated


def split_by_reward(
    records: List[TrajectoryRecord],
    positive_threshold: float = 0.5,
    negative_threshold: float = 0.0,
) -> tuple:
    """
    将轨迹列表按奖励分为正样本、负样本、中性样本三组。
    返回 (positive, negative, neutral)。
    """
    positive = [r for r in records if r.reward >= positive_threshold]
    negative = [r for r in records if r.reward < negative_threshold]
    neutral = [r for r in records if negative_threshold <= r.reward < positive_threshold]
    logger.info(
        f"奖励分组: 正={len(positive)}, 负={len(negative)}, 中性={len(neutral)}"
        f" | 总计={len(records)}"
    )
    return positive, negative, neutral


def build_dpo_pairs(
    records: List[TrajectoryRecord],
    positive_threshold: float = 0.5,
    negative_threshold: float = 0.0,
    max_pairs: int = 1000,
) -> List[tuple]:
    """
    从正负样本中构造 DPO 训练对 (chosen, rejected)。
    使用 task_description 做对齐（同任务的成功/失败对最理想）。
    无法对齐时，跨任务构造松散对（任何正样本 vs 任何负样本）。
    """
    positive, negative, _ = split_by_reward(records, positive_threshold, negative_threshold)
    if not positive or not negative:
        logger.warning("DPO 对构造失败：正样本或负样本数量不足。")
        return []

    # 优先：同 task_description 的成功/失败对
    pairs = []
    neg_by_task = {}
    for neg in negative:
        key = neg.task_description[:80]
        neg_by_task.setdefault(key, []).append(neg)

    for pos in positive:
        key = pos.task_description[:80]
        if key in neg_by_task:
            for neg in neg_by_task[key]:
                pairs.append((pos, neg))
                if len(pairs) >= max_pairs:
                    return pairs

    # 备选：跨任务松散配对（reward 差距 > 0.4）
    if len(pairs) < max_pairs:
        positive_sorted = sorted(positive, key=lambda r: r.reward, reverse=True)
        negative_sorted = sorted(negative, key=lambda r: r.reward)
        for pos in positive_sorted:
            for neg in negative_sorted:
                if pos.reward - neg.reward >= 0.4:
                    pairs.append((pos, neg))
                if len(pairs) >= max_pairs:
                    return pairs

    return pairs
