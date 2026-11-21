#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  BTC 24H 极值磁吸带 & Deribit GEX 监控看板 (V4 全自动ETF版)"
echo "============================================================"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[错误] 未检测到 python3，请先安装 Python 3.10+。"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[1/3] 首次运行，正在创建虚拟环境 .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi

echo "[2/3] 校验并安装依赖包..."
.venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt

PORT="${PORT:-5050}"
echo "[3/3] 启动看板服务 -> http://127.0.0.1:${PORT}"
exec .venv/bin/python dashboard_app.py
