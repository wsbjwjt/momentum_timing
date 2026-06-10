"""
工具函数模块
包含回测评估指标计算、性能统计等通用函数
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def evaluate_investment(
    df: pd.DataFrame,
    equity_col: str = "equity_curve",
    date_col: str = "date",
    risk_free: float = 0.03,
) -> Dict:
    """
    评估投资表现，计算各种关键指标。

    参数:
        df: 包含净值曲线和日期的DataFrame
        equity_col: 净值曲线列名
        date_col: 日期列名
        risk_free: 无风险利率（默认3%）

    返回:
        包含各项评估指标的字典
    """
    equity = df[equity_col].values
    dates = df[date_col].values

    # 日收益率
    daily_returns = np.diff(equity) / equity[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)

    # 累计收益率
    total_return = (equity[-1] / equity[0] - 1) * 100

    # 年化收益率
    n_days = len(equity)
    n_years = n_days / 252
    annual_return = ((equity[-1] / equity[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # 日胜率 — 仅计算持仓日
    if "pos" in df.columns:
        pos = df["pos"].values
        pos_days = pos == 1
        pos_returns = daily_returns[pos_days] if pos_days.sum() > 0 else daily_returns
    else:
        pos_returns = daily_returns
    win_days = np.sum(pos_returns > 0)
    loss_days = np.sum(pos_returns < 0)
    total_days = len(pos_returns)
    daily_win_rate = win_days / total_days * 100 if total_days > 0 else 0

    # 夏普比率
    excess_returns = daily_returns - risk_free / 252
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

    # 最大回撤
    cumulative_max = np.maximum.accumulate(equity)
    drawdowns = (equity - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdowns) * 100

    # 最大回撤起止时间
    dd_end_idx = np.argmin(drawdowns)
    dd_start_idx = np.argmax(cumulative_max[: dd_end_idx + 1])
    dd_start = pd.Timestamp(dates[dd_start_idx])
    dd_end = pd.Timestamp(dates[dd_end_idx])

    # 最大回撤恢复天数
    recovery_days = 0
    if dd_end_idx < len(equity) - 1:
        for i in range(dd_end_idx + 1, len(equity)):
            if equity[i] >= cumulative_max[dd_end_idx]:
                recovery_days = i - dd_end_idx
                break
        else:
            recovery_days = len(equity) - dd_end_idx - 1

    # Calmar比率
    calmar = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

    # 年化波动率
    annual_vol = np.std(daily_returns) * np.sqrt(252) * 100

    # 收益回撤比
    return_dd_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

    # 获取持仓相关的pos列
    if "pos" in df.columns:
        pos = df["pos"].values
        holding_days = np.sum(pos == 1)
        empty_days = np.sum(pos == 0)
        holding_ratio = holding_days / total_days * 100 if total_days > 0 else 0
    else:
        holding_days = total_days
        empty_days = 0
        holding_ratio = 100

    # 计算年化收益序列
    if n_years >= 1:
        annual_returns = []
        for y in range(int(n_years)):
            y_start = int(y * 252)
            y_end = min(int((y + 1) * 252), len(equity))
            if y_end > y_start:
                yr = (equity[y_end - 1] / equity[y_start] - 1) * 100
                annual_returns.append({
                    "year": str(pd.Timestamp(dates[y_start]).year),
                    "return": round(yr, 2),
                })
    else:
        annual_returns = []

    result = {
        "累计收益率(%)": round(total_return, 2),
        "年化收益率(%)": round(annual_return, 2),
        "年化波动率(%)": round(annual_vol, 2),
        "夏普比率": round(sharpe, 2),
        "最大回撤(%)": round(max_drawdown, 2),
        "最大回撤开始日": str(dd_start.date()),
        "最大回撤结束日": str(dd_end.date()),
        "最大回撤恢复天数": int(recovery_days),
        "Calmar比率": round(calmar, 2),
        "收益回撤比": round(return_dd_ratio, 2),
        "日胜率(%)": round(daily_win_rate, 2),
        "持仓天数": int(holding_days),
        "空仓天数": int(empty_days),
        "持仓占比(%)": round(holding_ratio, 2),
        "总天数": int(total_days),
        "年化收益序列": annual_returns,
    }

    return result


def evaluate_trades(
    df: pd.DataFrame,
    equity_col: str = "equity_curve",
    date_col: str = "date",
) -> Dict:
    """
    分析交易记录，计算交易维度的统计指标。

    参数:
        df: 包含signal和pos列的DataFrame
        equity_col: 净值列名
        date_col: 日期列名

    返回:
        包含交易统计的字典
    """
    if "signal" not in df.columns:
        return {"交易次数": 0, "错误": "无signal列"}

    signal = df["signal"].values
    dates = df[date_col].values
    equity = df[equity_col].values

    # 识别交易段（连续持仓 = 一次交易）
    pos = df["pos"].fillna(0).values if "pos" in df.columns else signal

    trades = []
    in_position = False
    entry_idx = 0

    for i in range(len(pos)):
        if pos[i] == 1 and not in_position:
            entry_idx = i
            in_position = True
        elif pos[i] == 0 and in_position:
            exit_idx = i - 1 if i > 0 else i
            if exit_idx > entry_idx:
                trade_return = (equity[exit_idx] / equity[entry_idx] - 1) * 100
                trades.append({
                    "开始日期": str(pd.Timestamp(dates[entry_idx]).date()),
                    "结束日期": str(pd.Timestamp(dates[exit_idx]).date()),
                    "持仓天数": exit_idx - entry_idx,
                    "收益率(%)": round(trade_return, 2),
                })
            in_position = False

    # 如果最后还在持仓
    if in_position and len(pos) > entry_idx:
        exit_idx = len(pos) - 1
        trade_return = (equity[exit_idx] / equity[entry_idx] - 1) * 100
        trades.append({
            "开始日期": str(pd.Timestamp(dates[entry_idx]).date()),
            "结束日期": str(pd.Timestamp(dates[exit_idx]).date()),
            "持仓天数": exit_idx - entry_idx,
            "收益率(%)": round(trade_return, 2),
        })

    if not trades:
        return {"交易次数": 0}

    returns = [t["收益率(%)"] for t in trades]
    durations = [t["持仓天数"] for t in trades]
    win_trades = [r for r in returns if r > 0]
    loss_trades = [r for r in returns if r <= 0]

    result = {
        "交易次数": len(trades),
        "盈利次数": len(win_trades),
        "亏损次数": len(loss_trades),
        "胜率(%)": round(len(win_trades) / len(trades) * 100, 2) if trades else 0,
        "平均收益率(%)": round(np.mean(returns), 2),
        "平均盈利(%)": round(np.mean(win_trades), 2) if win_trades else 0,
        "平均亏损(%)": round(np.mean(loss_trades), 2) if loss_trades else 0,
        "盈亏比": round(abs(np.mean(win_trades) / np.mean(loss_trades)), 2) if win_trades and loss_trades and np.mean(loss_trades) != 0 else 0,
        "最大单笔盈利(%)": round(max(returns), 2),
        "最大单笔亏损(%)": round(min(returns), 2),
        "平均持仓天数": round(np.mean(durations), 1),
        "交易明细": trades,
    }

    return result


def compute_drawdown_series(equity: np.ndarray) -> np.ndarray:
    """计算回撤序列"""
    cumulative_max = np.maximum.accumulate(equity)
    drawdowns = (equity - cumulative_max) / cumulative_max
    return drawdowns


def compute_rolling_sharpe(returns: np.ndarray, window: int = 252) -> np.ndarray:
    """计算滚动夏普比率"""
    roll_mean = pd.Series(returns).rolling(window).mean()
    roll_std = pd.Series(returns).rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).values


def compute_rolling_volatility(returns: np.ndarray, window: int = 60) -> np.ndarray:
    """计算滚动波动率"""
    return (pd.Series(returns).rolling(window).std() * np.sqrt(252) * 100).values


def compute_rolling_max_drawdown(equity: np.ndarray, window: int = 252) -> np.ndarray:
    """计算滚动最大回撤"""
    result = np.full(len(equity), np.nan)
    for i in range(window - 1, len(equity)):
        segment = equity[i - window + 1 : i + 1]
        peak = np.maximum.accumulate(segment)
        dd = (segment - peak) / peak
        result[i] = np.min(dd) * 100
    return result
