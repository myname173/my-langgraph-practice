#!/bin/bash
cd ~/my-langgraph-practice

# 正确激活 conda 环境（适用于脚本）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

export PYTHONPATH=src:$PYTHONPATH

# 清除代理（防止干扰）
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "🚀 正在启动 LangGraph Studio (端口 8123)..."

DATABASE_URI=sqlite:///:memory: \
REDIS_URI=redis://localhost:6379/0 \
LANGGRAPH_RUNTIME_EDITION=inmem \
LANGGRAPH_STORE_TYPE=inmem \
python -m langgraph_api.server --config langgraph.json --host 0.0.0.0 --port 8123 --reload
