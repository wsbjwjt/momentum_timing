# 量化择时策略回测系统 - Windows PowerShell 启动脚本
# 自动设置 UTF-8 编码，确保中文和 emoji 正常输出

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

uv run python run_backtest.py @args
