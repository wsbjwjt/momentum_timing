"""
回测引擎模块
负责运行策略回测、计算净值曲线、评估策略表现。
支持单标的和多标的批量回测。
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm
import json
import os

from .strategy import SlopeTimingStrategy
from .utils import evaluate_investment, evaluate_trades, compute_drawdown_series


class BacktestEngine:
    """
    回测引擎。

    支持:
    - 单标的单策略回测
    - 多策略参数对比
    - 多标的批量回测
    - 交易成本模拟
    """

    def __init__(
        self,
        strategy: SlopeTimingStrategy,
        initial_capital: float = 1.0,
        commission_rate: float = 0.0,
        slippage: float = 0.0,
        benchmark_col: str = "close",
    ):
        """
        初始化回测引擎。

        参数:
            strategy: 策略对象
            initial_capital: 初始资金（用于净值计算基准）
            commission_rate: 单边交易费率（如 0.001 = 0.1%）
            slippage: 滑点比例
            benchmark_col: 基准价格列名（用于计算基准收益率）
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.benchmark_col = benchmark_col

    def run(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        运行单次回测。

        参数:
            df: 行情数据 DataFrame
            verbose: 是否输出进度

        返回:
            包含策略信号和净值曲线的 DataFrame
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 开始回测")
            config = self.strategy.get_config_summary()
            for k, v in config.items():
                print(f"   {k}: {v}")
            print(f"   数据范围: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
            print(f"   数据天数: {len(df)}")
            print(f"{'='*60}")

        # 1. 运行策略计算信号
        result = self.strategy.compute(df, verbose=verbose)

        # 2. 计算基础日收益率
        # 使用 close / pre_close 计算指数收益率
        if "pre_close" not in result.columns:
            result["pre_close"] = result[self.benchmark_col].shift(1)
        result["daily_return"] = result[self.benchmark_col] / result["pre_close"] - 1
        # 处理第一行
        result["daily_return"] = result["daily_return"].fillna(0)

        # 3. 计算策略日收益率
        result["strategy_return"] = result["daily_return"] * result["pos"]

        # 4. 计算交易信号变化点（用于计费）
        signal_changes = np.diff(result["pos"].values, prepend=0)
        trade_points = np.abs(signal_changes) > 0.5
        result["trade_event"] = trade_points.astype(int)

        # 5. 扣除交易成本后的日收益率
        cost_impact = np.zeros(len(result))
        for i in range(len(result)):
            if result["trade_event"].iloc[i] == 1:
                cost_impact[i] = -(self.commission_rate * 2 + self.slippage)
        result["cost_impact"] = cost_impact
        result["strategy_return_net"] = result["strategy_return"] + result["cost_impact"]

        # 6. 计算净值曲线
        result["equity_curve"] = (1 + result["strategy_return"]).cumprod() * self.initial_capital
        result["equity_curve_net"] = (1 + result["strategy_return_net"]).cumprod() * self.initial_capital
        result["benchmark_curve"] = (1 + result["daily_return"]).cumprod() * self.initial_capital

        # 7. 计算回撤序列
        result["drawdown"] = compute_drawdown_series(result["equity_curve_net"].values)

        return result

    def evaluate(self, result: pd.DataFrame) -> Dict:
        """
        评估回测结果。

        参数:
            result: 回测结果 DataFrame

        返回:
            包含评估指标的字典
        """
        # 策略表现 — 构建临时 DataFrame 避免列名冲突
        strat_df = pd.DataFrame({
            "date": result["date"],
            "equity_curve": result["equity_curve_net"],
            "pos": result["pos"],
            "signal": result["signal"],
        })
        strategy_metrics = evaluate_investment(strat_df, equity_col="equity_curve", date_col="date")

        # 基准表现
        bench_df = pd.DataFrame({
            "date": result["date"],
            "equity_curve": result["benchmark_curve"],
        })
        benchmark_metrics = evaluate_investment(bench_df, equity_col="equity_curve", date_col="date")

        # 交易统计
        trade_stats = evaluate_trades(strat_df, equity_col="equity_curve", date_col="date")

        # 超额收益
        excess_return = strategy_metrics.get("年化收益率(%)", 0) - benchmark_metrics.get("年化收益率(%)", 0)

        # 整理输出
        summary = {
            "策略表现": {
                "累计收益率(%)": strategy_metrics.get("累计收益率(%)"),
                "年化收益率(%)": strategy_metrics.get("年化收益率(%)"),
                "年化波动率(%)": strategy_metrics.get("年化波动率(%)"),
                "夏普比率": strategy_metrics.get("夏普比率"),
                "最大回撤(%)": strategy_metrics.get("最大回撤(%)"),
                "最大回撤开始日": strategy_metrics.get("最大回撤开始日"),
                "最大回撤结束日": strategy_metrics.get("最大回撤结束日"),
                "Calmar比率": strategy_metrics.get("Calmar比率"),
                "日胜率(%)": strategy_metrics.get("日胜率(%)"),
            },
            "基准表现": {
                "累计收益率(%)": benchmark_metrics.get("累计收益率(%)"),
                "年化收益率(%)": benchmark_metrics.get("年化收益率(%)"),
                "最大回撤(%)": benchmark_metrics.get("最大回撤(%)"),
            },
            "超额收益(%)": round(excess_return, 2),
            "交易统计": trade_stats,
            "策略配置": self.strategy.get_config_summary(),
            "数据日期": {
                "开始": str(result["date"].iloc[0].date()),
                "结束": str(result["date"].iloc[-1].date()),
                "天数": len(result),
            },
        }

        return summary

    def run_and_evaluate(self, df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """运行回测并返回结果和评估"""
        result = self.run(df, verbose=verbose)
        evaluation = self.evaluate(result)
        return result, evaluation

    def export_results(
        self,
        result: pd.DataFrame,
        evaluation: Dict,
        output_dir: str = "output",
        prefix: str = "backtest",
        verbose: bool = True,
    ) -> Dict[str, str]:
        """
        导出回测结果为 CSV 和 JSON 文件。

        参数:
            result: 回测结果 DataFrame
            evaluation: 评估结果字典
            output_dir: 输出目录
            prefix: 文件名前缀

        返回:
            Dict[str, str]: 各输出文件路径
        """
        os.makedirs(output_dir, exist_ok=True)

        # 导出净值曲线
        equity_path = os.path.join(output_dir, f"{prefix}_equity.csv")
        equity_cols = ["date", "equity_curve", "equity_curve_net", "benchmark_curve",
                       "drawdown", "daily_return", "strategy_return", "signal", "pos"]
        available_cols = [c for c in equity_cols if c in result.columns]
        result[available_cols].to_csv(equity_path, index=False, encoding="utf-8-sig")

        # 导出交易记录
        trades_path = os.path.join(output_dir, f"{prefix}_trades.csv")
        if "交易明细" in evaluation.get("交易统计", {}):
            trades_df = pd.DataFrame(evaluation["交易统计"]["交易明细"])
            trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame({"note": ["无交易"]}).to_csv(trades_path, index=False)

        # 导出评估概要
        summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2, default=str)

        paths = {
            "equity": equity_path,
            "trades": trades_path,
            "summary": summary_path,
        }

        if verbose:
            print(f"\n📁 结果已导出:")
            for name, path in paths.items():
                print(f"   {name}: {path}")

        return paths


class MultiBacktestRunner:
    """
    多标的批量回测运行器。
    支持同时对多个标的运行同一策略，汇总统计结果。
    """

    def __init__(
        self,
        strategy_class=None,
        strategy_params: Optional[Dict] = None,
        commission_rate: float = 0.0,
    ):
        self.strategy_class = strategy_class or SlopeTimingStrategy
        self.strategy_params = strategy_params or {}
        self.commission_rate = commission_rate

    def run_batch(
        self,
        data_dict: Dict[str, pd.DataFrame],
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """
        对多个标的运行回测。

        参数:
            data_dict: {标的代码: 行情DataFrame}
            verbose: 是否显示进度

        返回:
            {标的代码: 评估结果字典}
        """
        results = {}
        strategy_stats = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 批量回测 {len(data_dict)} 个标的")
            print(f"{'='*60}")

        for code, df in tqdm(data_dict.items(), desc="回测进度", disable=not verbose):
            try:
                strategy = self.strategy_class(**self.strategy_params)
                engine = BacktestEngine(strategy, commission_rate=self.commission_rate)
                result_df, evaluation = engine.run_and_evaluate(df, verbose=False)
                evaluation["标的代码"] = code
                results[code] = evaluation
                strategy_stats.append(evaluation)
            except Exception as e:
                tqdm.write(f"  ⚠️ {code}: 回测失败 - {e}")
                continue

        # 汇总统计
        if strategy_stats:
            self._print_batch_summary(strategy_stats)

        return results

    def _print_batch_summary(self, stats: List[Dict]):
        """打印批量回测汇总"""
        annual_returns = [s["策略表现"]["年化收益率(%)"] for s in stats if s["策略表现"]["年化收益率(%)"] is not None]
        sharpes = [s["策略表现"]["夏普比率"] for s in stats if s["策略表现"]["夏普比率"] is not None]
        drawdowns = [s["策略表现"]["最大回撤(%)"] for s in stats if s["策略表现"]["最大回撤(%)"] is not None]

        if not annual_returns:
            return

        print(f"\n{'='*60}")
        print(f"📊 批量回测汇总 ({len(stats)} 个标的)")
        print(f"{'='*60}")
        print(f"  平均年化收益: {np.mean(annual_returns):.2f}% (中位数: {np.median(annual_returns):.2f}%)")
        print(f"  平均夏普比率: {np.mean(sharpes):.2f} (中位数: {np.median(sharpes):.2f})")
        print(f"  平均最大回撤: {np.mean(drawdowns):.2f}% (中位数: {np.median(drawdowns):.2f}%)")

        # 胜率统计
        win_count = sum(1 for r in annual_returns if r > 0)
        print(f"  正收益标的: {win_count}/{len(annual_returns)} ({win_count/len(annual_returns)*100:.1f}%)")
        print(f"{'='*60}\n")
