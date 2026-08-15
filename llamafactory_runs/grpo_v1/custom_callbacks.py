
# custom_callbacks.py（放到 LlamaFactory 根目录，通过 --callbacks 加载）
# 计算 SWE Agent 专属指标：工具调用准确率、测试通过率、迭代效率
import json, re
from transformers import TrainerCallback

class SWEAgentMetricsCallback(TrainerCallback):
    """
    向 WandB/TensorBoard 注入 SWE Agent 专属训练指标。
    在 trainer.predict() 输出的 predictions 中提取：
      - tool_call_accuracy: 工具调用成功率（正确调用 / 总调用）
      - test_pass_rate:     包含 TASK_COMPLETED 的预测比例
      - iteration_efficiency: 1 - mean(iterations / max_iterations)
    """

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        # 从 eval_loss 辅助字段中读取自定义指标（由 compute_metrics 注入）
        for k in ["tool_call_accuracy", "test_pass_rate", "iteration_efficiency"]:
            if k in metrics:
                # LlamaFactory 的 trainer 会自动 log 到 WandB
                pass

def compute_swe_metrics(eval_preds):
    """
    传入 compute_metrics 的函数。
    eval_preds.predictions: List[str]，模型生成的文本
    eval_preds.label_ids:   List[str]，参考答案
    """
    predictions, labels = eval_preds
    if not isinstance(predictions[0], str):
        return {}

    task_completed = sum(1 for p in predictions if "TASK_COMPLETED" in p)
    test_pass_rate = task_completed / max(len(predictions), 1)

    # 工具调用格式合法性
    valid_tool_calls = 0
    total_tool_calls = 0
    for pred in predictions:
        matches = re.findall(r'\{"name":\s*"(\w+)"', pred)
        total_tool_calls += len(matches)
        valid_tool_calls += sum(
            1 for m in matches
            if m in ["execute_command", "write_file", "read_file",
                     "search_code", "edit_file", "search_codebase"]
        )
    tool_call_accuracy = valid_tool_calls / max(total_tool_calls, 1)

    return {
        "test_pass_rate": round(test_pass_rate, 4),
        "tool_call_accuracy": round(tool_call_accuracy, 4),
    }
