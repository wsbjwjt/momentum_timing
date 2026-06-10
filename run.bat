@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
uv run python run_backtest.py %*
