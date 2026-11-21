@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   BTC 24H 极值磁吸带 ^& Deribit GEX 监控看板 (V3 美国直连版)
echo ============================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+ 并勾选 Add to PATH。
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 首次运行，正在创建独立虚拟环境 .venv ...
    python -m venv .venv
)

echo [2/3] 校验并安装依赖包 (Flask / aiohttp / gunicorn)...
".venv\Scripts\pip.exe" install --quiet --disable-pip-version-check -r requirements.txt

echo [3/3] 启动看板服务 -> http://127.0.0.1:5050
start "" http://127.0.0.1:5050
".venv\Scripts\python.exe" dashboard_app.py

pause
