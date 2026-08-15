# src/agent/swe/training/import_public_data.py
"""
公开 SWE 轨迹数据导入器
========================
从公开 SWE 轨迹数据集（SWE-agent / Nemotron-SFT-SWE 等）拉取轨迹，
转换为本项目统一的 TrajectoryRecord 格式，并 **追加** 写入：
    workspace/_training_data/trajectories.jsonl

设计要点：
  - 不修改现有 reward_computer / data_pipeline / trajectory_logger 逻辑：
    公开轨迹先归一化为 TrajectoryRecord（保留原始 success/failed 标签），
    后续 reward 标注 + SFT/DPO 导出全部复用现有 run_pipeline。
  - reward 正负由原始 status / test_passed 决定（success&test_passed→1.0，
    failed→-0.3），从而自然凑出 DPO 所需的「正样本 ≥0.5 / 负样本 <0」分组。
  - 抽样控制：公开数据集巨大，小样验证只需各取若干条即可达标，
    通过 --max-success / --max-failed 控制，避免全量下载。
  - 多源适配：--source 选择适配器（sweagent / nemotron / synth）。
    当本地没有真实数据集文件且不想联网时，synth 源会基于公开 SWE-bench
    风格任务描述合成合理的成功/失败轨迹，保证离线也能达标。

用法：
  # 从 HuggingFace 拉取 SWE-agent 轨迹（需 huggingface_hub + 网络）
  python -m src.agent.swe.training.import_public_data --source sweagent \
      --hf-repo SWE-agent/SWE-agent-trajectories --max-success 35 --max-failed 25

  # 从本地已下载的 jsonl 文件读取
  python -m src.agent.swe.training.import_public_data --source sweagent \
      --local-path ./data/swe_agent_trajectories.jsonl --max-success 35 --max-failed 25

  # 完全离线兜底：合成公开风格轨迹
  python -m src.agent.swe.training.import_public_data --source synth \
      --max-success 35 --max-failed 25
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.agent.swe.training.trajectory_logger import (
    ToolCallStep,
    TrajectoryRecord,
    get_training_dir,
    load_trajectories,
    save_trajectory,
)
from src.agent.swe.training.data_pipeline import run_pipeline

logger = logging.getLogger("SWE_ImportPublic")

_DATA_DIR = Path(__file__).resolve().parents[4]  # 项目根


# ==========================================
# 工具函数
# ==========================================

def _truncate(text: str, max_len: int = 400) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...[截断，原长 {len(text)} 字符]"


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ==========================================
# 适配器：SWE-agent 格式
# ==========================================

def convert_swe_agent_trajectory(raw: dict) -> Optional[TrajectoryRecord]:
    """
    将 SWE-agent 原始轨迹 dict 映射为 TrajectoryRecord。

    SWE-agent 典型字段（不同导出略有差异，做健壮兜底）：
      - problem_statement / task: 任务描述
      - success / resolved / test_passed: 是否成功
      - trajectory: 主轨迹文本 或 actions/observations 列表
    """
    task = (raw.get("problem_statement") or raw.get("task")
            or raw.get("instance_id") or "").strip()
    if not task:
        return None

    # 成功判定
    success = bool(raw.get("success", raw.get("resolved", raw.get("test_passed", False))))
    test_passed = bool(raw.get("test_passed", success))

    # 工具步骤：优先用结构化 actions/observations
    steps: List[ToolCallStep] = []
    actions = raw.get("actions") or raw.get("trajectory", {}).get("actions") if isinstance(raw.get("trajectory"), dict) else None
    observations = raw.get("observations") or (raw.get("trajectory", {}).get("observations") if isinstance(raw.get("trajectory"), dict) else None)

    if actions and observations and len(actions) == len(observations):
        for act, obs in zip(actions, observations):
            act_str = act if isinstance(act, str) else json.dumps(act, ensure_ascii=False)
            # SWE-agent action 形如：<command>... 或 edit/...，提取工具名
            tool_name = "shell" if act_str.startswith("<command>") else "edit"
            steps.append(ToolCallStep(
                tool_name=tool_name,
                tool_args={"action": _truncate(act_str, 200)},
                tool_result=_truncate(str(obs), 400),
                success=not str(obs).startswith("ERROR"),
                result_len=len(str(obs)),
            ))
    else:
        # 退化为单步：把整段 trajectory 文本作为一次“reason”工具
        traj_text = raw.get("trajectory")
        traj_text = traj_text if isinstance(traj_text, str) else json.dumps(traj_text, ensure_ascii=False)
        if traj_text:
            steps.append(ToolCallStep(
                tool_name="agent_trace",
                tool_args={"source": "swe_agent"},
                tool_result=_truncate(traj_text, 400),
                success=success,
                result_len=len(traj_text),
            ))

    if not steps:
        steps.append(ToolCallStep(
            tool_name="agent_trace",
            tool_args={"source": "swe_agent"},
            tool_result=_truncate(task, 200),
            success=success,
            result_len=len(task),
        ))

    return TrajectoryRecord(
        trajectory_id=f"sa-{abs(hash(json.dumps(raw, sort_keys=True, ensure_ascii=False))) % 10**12:012d}",
        task_description=task,
        timestamp=raw.get("timestamp") or _now(),
        tool_call_steps=steps,
        llm_reasoning_snippets=[_truncate(task, 200)],
        status="success" if success else "failed",
        test_passed=test_passed,
        iteration_count=len(steps),
        max_iterations=max(len(steps), 1) * 2,
        summary=raw.get("summary", "") or _truncate(task, 200),
    )


# ==========================================
# 适配器：Nemotron-SFT-SWE 格式
# ==========================================

def _degrade_patch(patch: str) -> str:
    """
    构造 DPO 负样本用的「退化 patch」：保留 gold patch 的结构骨架（diff 头 + 文件/函数名），
    但移除关键修改内容，得到一个「看起来像修了、实则未真正修复」的最弱解。

    设计说明（面试可讲）：
    SWE-bench 本身是「问题陈述 + gold patch」的评测集，不含 agent 失败轨迹。DPO 偏好学习
    需要 chosen/rejected 对，这里 rejected = gold patch 的退化版本（剔除 hunk 正文、仅留骨架），
    对应「模型未真正修复」的劣质解。这是 SWE 偏好微调的通用构造方式，并非伪装成真实失败轨迹。
    """
    out_lines = []
    for line in patch.splitlines():
        # 保留 diff 元数据行（--- / +++ / @@ / index 等），剔除真正的代码改动
        if line.startswith(("diff --git", "--- ", "+++ ", "@@", "index ", "new file", "deleted file", "rename ")):
            out_lines.append(line)
        elif line.startswith((" ",)) :
            # 上下文行保留，代码改动行（+/ -开头）一律丢弃，从而 patch 不再产生有效修改
            continue
        else:
            # + / - 开头的实际改动行：丢弃，制造「未修复」
            continue
    return "\n".join(out_lines).strip() + "\n"


def convert_swebench_instance(raw: dict, success: bool = True) -> Optional[TrajectoryRecord]:
    """
    将 SWE-bench（Lite/Verified）任务实例映射为 TrajectoryRecord。

    说明：SWE-bench 数据集仅含「问题陈述 + gold patch + 测试标签」，没有 agent 的
    中间工具调用步骤。这里把 patch 构造为一个最终的 submit_patch 步骤，使
    trajectory_to_sharegpt 能将其导出为「给定 bug 描述 -> 生成 patch」的样本。

    - success=True：使用 gold patch 作为正样本（status=success, reward≈1.0，SFT/DPO chosen）。
    - success=False：使用退化 patch（_degrade_patch）作为负样本（status=failed, reward<0，DPO rejected）。
    """
    try:
        instance_id = raw.get("instance_id") or raw.get("instance_ID") or "unknown"
        repo = raw.get("repo") or ""
        problem = (raw.get("problem_statement") or "").strip()
        patch = (raw.get("patch") or "").strip()
        hints = (raw.get("hints_text") or "").strip()
        if not problem or not patch:
            return None
        # 任务描述：带 repo 上下文，贴近真实 agent 接收的 prompt
        task = f"Repository: {repo}\nInstance: {instance_id}\n\n{problem}"
        if hints:
            task += f"\n\nHints:\n{hints}"
        used_patch = patch if success else _degrade_patch(patch)
        # 模拟 agent 完成修复的最终步骤：提交 patch（成功=gold，失败=退化）
        steps = [
            ToolCallStep(
                tool_name="analyze_issue",
                tool_args={"instance_id": instance_id, "repo": repo},
                tool_result=(problem[:300] + "…") if len(problem) > 300 else problem,
                success=True,
                result_len=len(problem),
            ),
            ToolCallStep(
                tool_name="submit_patch",
                tool_args={"patch": used_patch},
                tool_result=f"FAIL_TO_PASS={raw.get('FAIL_TO_PASS')} PASS_TO_PASS={raw.get('PASS_TO_PASS')}",
                success=success,
                result_len=len(used_patch),
            ),
        ]
        return TrajectoryRecord(
            trajectory_id=f"swebench-{instance_id}" + ("" if success else "-rej"),
            task_description=task,
            timestamp=datetime.now().isoformat(),
            tool_call_steps=steps,
            status="success" if success else "failed",
            test_passed=success,
            summary=f"source=swebench success={success} instance_id={instance_id} base_commit={raw.get('base_commit')}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("convert_swebench_instance 失败: %s", exc)
        return None


def _iter_swebench(max_success: int, max_failed: int, local_path: str | None = None):
    """
    流式迭代 SWE-bench 任务实例（parquet），产出「成功正样本 + 失败负样本」。

    - 成功样本：max_success 条，使用 gold patch（SFT 正样本 / DPO chosen）。
    - 失败样本：max_failed 条，使用退化 patch（DPO rejected，见 _degrade_patch）。

    默认从 ModelScope 拉取 AI-ModelScope/SWE-bench_Verified 的 test parquet
    （国内可达，无需代理；HuggingFace 在受限网络下常超时，故优先 ModelScope）。
    也可通过 local_path 指定本地 parquet 文件。Verified 子集人工校验过、噪声更低，更适合训练。
    """
    import pandas as pd  # 惰性导入，避免无 pandas 时整体失败

    path = local_path
    if not path:
        # 优先 ModelScope（国内直连可达）；备选 HuggingFace 官方/HF 镜像
        candidates = [
            ("ModelScope", "https://modelscope.cn/datasets/AI-ModelScope/SWE-bench_Verified/resolve/master/data/test-00000-of-00001.parquet"),
            ("HuggingFace", "https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/resolve/main/data/test-00000-of-00001.parquet"),
        ]
        for name, url in candidates:
            try:
                import tempfile
                import urllib.request
                logger.info("尝试从 %s 下载 SWE-bench_Verified parquet ...", name)
                suffix = ".parquet"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                urllib.request.urlretrieve(url, tmp.name)
                path = tmp.name
                logger.info("下载成功: %s -> %s", name, path)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s 下载失败: %s", name, exc)
                continue
        if not path:
            logger.error("所有数据源下载失败（ModelScope/HuggingFace 均不可达）")
            return
    df = pd.read_parquet(path)
    logger.info("SWE-bench_Verified 数据集加载 %d 条，成功 %d / 失败 %d", len(df), max_success, max_failed)

    # 第一轮：成功正样本
    taken = 0
    for _, row in df.iterrows():
        if taken >= max_success:
            break
        rec = convert_swebench_instance(row.to_dict(), success=True)
        if rec is None:
            continue
        yield rec
        taken += 1

    # 第二轮：失败负样本（来自同一批实例的退化 patch，避免引入外部噪声）
    taken = 0
    for _, row in df.iterrows():
        if taken >= max_failed:
            break
        rec = convert_swebench_instance(row.to_dict(), success=False)
        if rec is None:
            continue
        yield rec
        taken += 1


def convert_nemotron_trajectory(raw: dict) -> Optional[TrajectoryRecord]:
    """
    将 Nemotron-SFT-SWE 原始轨迹 dict 映射为 TrajectoryRecord。

    该数据集多为 ShareGPT / messages 结构：
      - messages: [system?, user, assistant(tool_calls), tool, ...]
      - 通常带 reward / score / success 字段
    """
    messages = raw.get("messages") or []
    # 找第一条 user 消息作为 task
    task = ""
    for m in messages:
        if m.get("role") in ("user", "human") and m.get("content"):
            task = str(m["content"]).strip()
            break
    if not task:
        task = (raw.get("instruction") or raw.get("problem") or "").strip()
    if not task:
        return None

    success = bool(raw.get("success", raw.get("resolved", raw.get("test_passed", False))))
    # Nemotron 常带 reward/score：>0 视为成功
    if "reward" in raw or "score" in raw:
        sc = float(raw.get("reward", raw.get("score", 0.0)))
        success = sc > 0
    test_passed = bool(raw.get("test_passed", success))

    steps: List[ToolCallStep] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                    name = fn.get("name", "tool")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"raw": args}
                    steps.append(ToolCallStep(
                        tool_name=name,
                        tool_args={"args": _truncate(json.dumps(args, ensure_ascii=False), 200)},
                        tool_result="",
                        success=True,
                        result_len=0,
                    ))
            elif content:
                steps.append(ToolCallStep(
                    tool_name="reasoning",
                    tool_args={"text": _truncate(str(content), 200)},
                    tool_result="",
                    success=True,
                    result_len=0,
                ))
        elif role == "tool":
            if steps:
                steps[-1].tool_result = _truncate(str(content), 400)
                steps[-1].result_len = len(str(content))
                steps[-1].success = not str(content).startswith("ERROR")

    if not steps:
        steps.append(ToolCallStep(
            tool_name="agent_trace",
            tool_args={"source": "nemotron"},
            tool_result=_truncate(task, 200),
            success=success,
            result_len=len(task),
        ))

    return TrajectoryRecord(
        trajectory_id=f"nm-{abs(hash(json.dumps(raw, sort_keys=True, ensure_ascii=False))) % 10**12:012d}",
        task_description=task,
        timestamp=raw.get("timestamp") or _now(),
        tool_call_steps=steps,
        llm_reasoning_snippets=[
            _truncate(str(m.get("content", "")), 200)
            for m in messages if m.get("role") == "assistant" and m.get("content")
        ][:5],
        status="success" if success else "failed",
        test_passed=test_passed,
        iteration_count=len(steps),
        max_iterations=max(len(steps), 1) * 2,
        summary=raw.get("summary", "") or _truncate(task, 200),
    )


# ==========================================
# 兜底：合成公开风格轨迹（离线）
# ==========================================

_SYNTH_TASKS = [
    "在 flask 项目中修复 `Request.environ` 在多线程下偶发返回 None 的问题。",
    "为 requests 库增加连接池在异常时的优雅回收逻辑。",
    "修复 numpy `np.load` 在 mmap_mode='r' 时大文件内存泄漏。",
    "pylint 误报 `unused-argument` 在 pytest 装饰器场景，需要抑制。",
    "django ORM `select_for_update()` 在 postgresql 上 skip_locked 支持。",
    "scikit-learn `StandardScaler` 处理稀疏矩阵时 warning 噪声过大。",
    "pandas `read_csv` 解析含 BOM 的 utf-8-sig 文件乱码。",
    "matplotlib 在 agg 后端保存透明背景 PNG 失效。",
    "sqlalchemy 2.0 `selectinload` 与 `limit` 联用产生额外查询。",
    "torch `DataLoader` 多进程在 Windows 下 fork 报错，需 spawn。",
]


def _synth_steps(success: bool, seed: int) -> List[ToolCallStep]:
    rnd = random.Random(seed)
    n = rnd.randint(3, 7)
    steps: List[ToolCallStep] = []
    for i in range(n):
        tool = rnd.choice(["shell", "edit", "search_code", "run_test"])
        if tool == "shell":
            args = {"command": f"grep -rn 'TODO' src/pkg{i}/ | head"}
            res = f"src/pkg{i}/mod.py:42:    # TODO handle edge case\nfound {rnd.randint(1,9)} matches"
        elif tool == "edit":
            args = {"file": f"src/pkg{i}/mod.py", "old": "return None", "new": "return cached_value"}
            res = "1 file changed, 3 insertions(+), 1 deletion(-)"
        elif tool == "search_code":
            args = {"query": f"def process_request_{i}()"}
            res = f"src/pkg{i}/mod.py\ndef process_request_{i}(): ..."
        else:
            args = {"command": f"pytest tests/test_pkg{i}.py -q"}
            res = "3 passed" if (success or i < n - 1) else "1 failed, 2 passed"
        steps.append(ToolCallStep(
            tool_name=tool,
            tool_args={"args": _truncate(json.dumps(args, ensure_ascii=False), 200)},
            tool_result=_truncate(res, 400),
            success=not (tool == "run_test" and not success and i == n - 1),
            result_len=len(res),
        ))
    return steps


def _synth_record(idx: int, success: bool) -> TrajectoryRecord:
    task = _SYNTH_TASKS[idx % len(_SYNTH_TASKS)]
    return TrajectoryRecord(
        trajectory_id=f"sy-{idx:012d}",
        task_description=task,
        timestamp=_now(),
        tool_call_steps=_synth_steps(success, idx),
        llm_reasoning_snippets=[_truncate(task, 200)],
        status="success" if success else "failed",
        test_passed=success,
        iteration_count=random.Random(idx).randint(3, 7),
        max_iterations=15,
        summary=_truncate(task, 200),
    )


# ==========================================
# 数据获取
# ==========================================

def _iter_hf_dataset(repo: str, split: str, max_items: int) -> Iterator[dict]:
    """使用 huggingface_hub 流式读取数据集（按需下载，不落全量）。"""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("❌ 未安装 huggingface_hub，请 pip install huggingface_hub 或使用 --local-path / --source synth")
        return iter([])

    api = HfApi()
    ds = api.dataset_info(repo)
    logger.info(f"📦 数据集: {repo} (files={len(ds.siblings)})")
    # 尝试读取 parquet / jsonl
    import io
    try:
        from datasets import load_dataset
        hf_ds = load_dataset(repo, split=split, streaming=True)
        count = 0
        for row in hf_ds:
            if count >= max_items:
                break
            yield row
            count += 1
    except Exception as e:
        logger.error(f"❌ 通过 datasets 读取失败: {e}")
        return iter([])


def _iter_local_jsonl(path: str) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        logger.error(f"❌ 本地文件不存在: {path}")
        return iter([])
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


# ==========================================
# 主导入流程
# ==========================================

def import_and_export(
    source: str = "synth",
    local_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_split: str = "train",
    max_success: int = 35,
    max_failed: int = 25,
    workspace_dir: Optional[Path] = None,
) -> Dict:
    """
    下载/读取 → 转换 → 追加写入 → 重新导出，返回数据量报告。
    """
    workspace_dir = Path(workspace_dir) if workspace_dir else (_DATA_DIR / "workspace")
    training_dir = get_training_dir(workspace_dir)

    # 统计导入前的条数（用于对比）
    before = len(load_trajectories(workspace_dir))

    # 选择适配器 + 数据迭代器
    if source == "sweagent":
        adapter = convert_swe_agent_trajectory
        it = _iter_local_jsonl(local_path) if local_path else _iter_hf_dataset(hf_repo or "SWE-agent/SWE-agent-trajectories", hf_split, max_success + max_failed)
    elif source == "nemotron":
        adapter = convert_nemotron_trajectory
        it = _iter_local_jsonl(local_path) if local_path else _iter_hf_dataset(hf_repo or "nvidia/Nemotron-SFT-SWE-v2", hf_split, max_success + max_failed)
    elif source == "swebench":
        adapter = None  # SWE-bench 自带记录生成，无需外部适配器
        it = iter([])
    else:
        raise ValueError(f"未知 source: {source}")

    added_success = 0
    added_failed = 0
    skipped = 0

    def _write_record(rec: TrajectoryRecord) -> bool:
        nonlocal added_success, added_failed
        if rec.status == "success":
            if added_success >= max_success:
                return False
            added_success += 1
        else:
            if added_failed >= max_failed:
                return False
            added_failed += 1
        save_trajectory(rec, workspace_dir)
        return True

    # 真实数据源
    if source in ("sweagent", "nemotron"):
        for raw in it:
            rec = adapter(raw)
            if rec is None:
                skipped += 1
                continue
            if not _write_record(rec):
                break  # 已达配额
    # SWE-bench 真实数据源（仅 SFT 正样本，无 synth 兜底，避免合成假数据污染训练集）
    if source == "swebench":
        it = _iter_swebench(max_success, max_failed, local_path=local_path)
        for rec in it:
            if not _write_record(rec):
                break  # 已达配额

    after = len(load_trajectories(workspace_dir))
    logger.info(f"📥 导入完成: 新增 {after - before} 条（成功 {added_success} / 失败 {added_failed}，跳过 {skipped}）")

    # ── 复用现有 pipeline 重新导出 ──
    report = run_pipeline(workspace_dir=workspace_dir)
    report["import"] = {
        "source": source,
        "before": before,
        "after": after,
        "added_success": added_success,
        "added_failed": added_failed,
        "skipped": skipped,
    }
    return report


# ==========================================
# CLI 入口
# ==========================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="导入公开 SWE 轨迹数据集")
    parser.add_argument("--source", choices=["sweagent", "nemotron", "swebench"], default="swebench",
                        help="公开数据源与适配器（仅使用真实公开数据，无 synth 兜底）")
    parser.add_argument("--local-path", default=None, help="本地数据集 jsonl 路径")
    parser.add_argument("--hf-repo", default=None, help="HuggingFace 数据集 repo id")
    parser.add_argument("--hf-split", default="train", help="HuggingFace 数据集 split")
    parser.add_argument("--max-success", type=int, default=35, help="最多导入成功轨迹条数")
    parser.add_argument("--max-failed", type=int, default=25, help="最多导入失败轨迹条数")
    parser.add_argument("--workspace", default=str(_DATA_DIR / "workspace"), help="工作区目录")
    args = parser.parse_args()

    try:
        rep = import_and_export(
            source=args.source,
            local_path=args.local_path,
            hf_repo=args.hf_repo,
            hf_split=args.hf_split,
            max_success=args.max_success,
            max_failed=args.max_failed,
            workspace_dir=Path(args.workspace),
        )
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.exception(f"导入失败: {e}")
        sys.exit(1)
