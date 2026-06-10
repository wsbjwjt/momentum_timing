#!/usr/bin/env python3
"""
量化择时策略回测系统 - 主入口

基于最高价与最低价的斜率指标（量价相对强度）构建择时信号。
支持多种策略变体、价格过滤、成交量过滤，以及单标的/多标的回测。

使用示例:
    # 快速回测（沪深300指数，默认参数）
    python run_backtest.py --mode quick

    # 标准回测（多个指数对比）
    python run_backtest.py --mode standard

    # 中等规模回测（200只个股）
    python run_backtest.py --mode medium --pool-size 200

    # 自定义参数
    python run_backtest.py --mode quick --method right_biased --n 16 --m 300 --s 0.7 --price-filter

    # 显示所有可用选项
    python run_backtest.py --help
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from momentum_timing.strategy import SlopeTimingStrategy
from momentum_timing.backtest import BacktestEngine, MultiBacktestRunner
from momentum_timing.data import (
    fetch_index_data, fetch_multiple_indices, fetch_stock_pool,
    get_hs300_stocks, get_common_indices, PREDEFINED_INDICES,
    clear_cache,
)
from momentum_timing.dashboard import build_dashboard, create_comparison_dashboard


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="量化择时策略回测系统 - 基于量价斜率指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式说明:
  quick     - 快速回测：沪深300指数，默认参数，适合快速验证（3-5分钟）
  standard  - 标准回测：6大指数对比，多种策略变体（8-15分钟）
  medium    - 中等规模：N只个股批量回测，验证策略普适性（时间取决于pool-size）
  compare   - 策略对比：单一标的上运行所有策略变体
  custom    - 自定义：完全自定义参数

示例:
  python run_backtest.py --mode quick
  python run_backtest.py --mode standard
  python run_backtest.py --mode medium --pool-size 100
  python run_backtest.py --mode compare --index sh.000300
  python run_backtest.py --mode custom --index sh.000300 --method zscore --n 18 --m 600 --s 0.7
        """,
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="quick",
        choices=["quick", "standard", "medium", "compare", "custom"],
        help="回测模式 (默认: quick)",
    )

    # 策略参数
    parser.add_argument("--method", type=str, default="right_biased",
                        choices=["slope", "zscore", "corrected", "right_biased"],
                        help="指标方法 (默认: right_biased)")
    parser.add_argument("--n", type=int, default=16, help="回归周期 N (默认: 16)")
    parser.add_argument("--m", type=int, default=300, help="标准分周期 M (默认: 300)")
    parser.add_argument("--s", type=float, default=0.7, help="信号阈值 S (默认: 0.7)")
    parser.add_argument("--price-filter", action="store_true", default=True,
                        help="启用价格(均线)过滤 (默认: 启用)")
    parser.add_argument("--no-price-filter", action="store_true",
                        help="关闭价格过滤")
    parser.add_argument("--volume-filter", action="store_true",
                        help="启用成交量相关性过滤")
    parser.add_argument("--ma-period", type=int, default=20, help="均线周期 (默认: 20)")
    parser.add_argument("--ma-lookback", type=int, default=3, help="均线回看天数 (默认: 3)")

    # 数据参数
    parser.add_argument("--index", type=str, default="sh.000300",
                        help="回测标的代码 (baostock格式)")
    parser.add_argument("--start", type=str, default="2005-01-01",
                        help="回测开始日期 (默认: 2005-01-01)")
    parser.add_argument("--end", type=str, default=None,
                        help="回测结束日期 (默认: 今天)")
    parser.add_argument("--pool-size", type=int, default=200,
                        help="个股池数量 (medium模式, 默认: 200)")
    parser.add_argument("--no-cache", action="store_true",
                        help="不使用数据缓存")

    # 成本参数
    parser.add_argument("--commission", type=float, default=0.0,
                        help="单边交易费率 (默认: 0, 如 0.001=0.1%%)")

    # 输出参数
    parser.add_argument("--output", type=str, default="output",
                        help="输出目录 (默认: output)")
    parser.add_argument("--prefix", type=str, default="backtest",
                        help="输出文件名前缀 (默认: backtest)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="不生成HTML仪表盘")
    parser.add_argument("--no-export", action="store_true",
                        help="不导出CSV/JSON文件")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式")

    return parser.parse_args()


def run_quick_mode(args):
    """快速回测模式 - 沪深300指数默认参数"""
    print("\n" + "="*70)
    print("  ⚡ 快速回测模式 - 沪深300指数")
    print("="*70)

    strategy = SlopeTimingStrategy(
        n_regression=args.n if args.no_price_filter else args.n,
        m_standardize=args.m,
        threshold=args.s,
        method=args.method,
        use_price_filter=not args.no_price_filter,
        use_volume_filter=args.volume_filter,
        ma_period=args.ma_period,
        ma_lookback=args.ma_lookback,
    )

    df = fetch_index_data(
        args.index, args.start, args.end,
        use_cache=not args.no_cache,
    )

    if len(df) == 0:
        print("❌ 未获取到数据，请检查网络或代码是否正确")
        return

    engine = BacktestEngine(strategy, commission_rate=args.commission)
    result, evaluation = engine.run_and_evaluate(df)

    # 打印结果
    _print_results(evaluation)

    # 导出
    paths = {}
    if not args.no_export:
        paths = engine.export_results(result, evaluation, args.output, args.prefix)

    # 生成仪表盘
    dashboard_path = None
    if not args.no_dashboard:
        dashboard_path = build_dashboard(
            result, evaluation,
            output_path=os.path.join(args.output, f"{args.prefix}_dashboard.html"),
            title="量化择时策略回测仪表盘",
        )
        print(f"\n📊 仪表盘: {dashboard_path}")

    return result, evaluation, dashboard_path


def run_standard_mode(args):
    """标准回测模式 - 多指数对比"""
    print("\n" + "="*70)
    print("  📊 标准回测模式 - 六大指数对比")
    print("="*70)

    indices = list(PREDEFINED_INDICES.keys())
    all_results = {}

    for name in indices:
        print(f"\n--- {name} ---")
        code = PREDEFINED_INDICES[name]

        strategy = SlopeTimingStrategy(
            n_regression=args.n,
            m_standardize=args.m,
            threshold=args.s,
            method="right_biased",
            use_price_filter=not args.no_price_filter,
            use_volume_filter=args.volume_filter,
        )

        df = fetch_index_data(code, args.start, args.end, use_cache=not args.no_cache, verbose=False)
        if len(df) == 0:
            print(f"  ⚠️ 跳过（无数据）")
            continue

        engine = BacktestEngine(strategy, commission_rate=args.commission)
        result, evaluation = engine.run_and_evaluate(df, verbose=False)
        all_results[name] = result

        perf = evaluation["策略表现"]
        print(f"  年化收益: {perf.get('年化收益率(%)', 0):.1f}%  "
              f"夏普: {perf.get('夏普比率', 0):.2f}  "
              f"最大回撤: {perf.get('最大回撤(%)', 0):.1f}%  "
              f"超额: {evaluation.get('超额收益(%)', 0):.1f}%")

    # 生成对比仪表盘
    if all_results:
        comparison_path = create_comparison_dashboard(
            all_results,
            output_path=os.path.join(args.output, "comparison_dashboard.html"),
            title="六大指数策略对比",
        )
        print(f"\n📊 对比仪表盘: {comparison_path}")

    return all_results


def run_medium_mode(args):
    """中等规模回测 - 多只个股"""
    print("\n" + "="*70)
    print(f"  📦 中等规模回测模式 - {args.pool_size} 只个股")
    print("="*70)

    # 获取股票池
    all_codes = get_hs300_stocks()
    codes = all_codes[:args.pool_size]

    print(f"  股票池: {len(codes)} 只（从 {len(all_codes)} 候选池中选取）")
    print(f"  开始获取数据...")

    stock_data = fetch_stock_pool(
        codes,
        start_date=args.start,
        end_date=args.end,
        min_days=200,
        use_cache=not args.no_cache,
    )

    if not stock_data:
        print("❌ 未获取到任何有效数据")
        return

    # 批量回测
    runner = MultiBacktestRunner(
        strategy_params={
            "n_regression": args.n,
            "m_standardize": args.m,
            "threshold": args.s,
            "method": args.method,
            "use_price_filter": not args.no_price_filter,
            "use_volume_filter": args.volume_filter,
        },
        commission_rate=args.commission,
    )

    runner.run_batch(stock_data)

    # 保存汇总
    summary_path = os.path.join(args.output, f"batch_{args.prefix}_summary.json")
    os.makedirs(args.output, exist_ok=True)

    return stock_data


def run_compare_mode(args):
    """策略对比模式 - 单一标的运行所有策略变体"""
    print("\n" + "="*70)
    print(f"  🔬 策略对比模式 - {args.index}")
    print("="*70)

    df = fetch_index_data(args.index, args.start, args.end, use_cache=not args.no_cache)
    if len(df) == 0:
        print("❌ 无数据")
        return

    # 策略变体
    variants = [
        {"method": "slope", "use_price_filter": False, "use_volume_filter": False, "label": "斜率策略"},
        {"method": "zscore", "use_price_filter": False, "use_volume_filter": False, "label": "标准分策略"},
        {"method": "corrected", "use_price_filter": False, "use_volume_filter": False, "label": "修正标准分"},
        {"method": "right_biased", "use_price_filter": False, "use_volume_filter": False, "label": "右偏标准分"},
        {"method": "right_biased", "use_price_filter": True, "use_volume_filter": False, "label": "右偏+价格过滤"},
        {"method": "right_biased", "use_price_filter": False, "use_volume_filter": True, "label": "右偏+量相关性"},
    ]

    all_results = {}
    print(f"\n{'策略变体':<20} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'交易次数':>8}")
    print("-" * 60)

    for v in variants:
        strategy = SlopeTimingStrategy(
            n_regression=args.n, m_standardize=args.m, threshold=args.s,
            method=v["method"],
            use_price_filter=v["use_price_filter"],
            use_volume_filter=v["use_volume_filter"],
        )
        engine = BacktestEngine(strategy, commission_rate=args.commission)
        result, evaluation = engine.run_and_evaluate(df, verbose=False)
        all_results[v["label"]] = result

        perf = evaluation["策略表现"]
        trades = evaluation["交易统计"].get("交易次数", 0)
        print(f"{v['label']:<20} {perf.get('年化收益率(%)',0):>8.1f}% {perf.get('夏普比率',0):>8.2f} "
              f"{perf.get('最大回撤(%)',0):>8.1f}% {trades:>8}")

    # 对比图
    comparison_path = create_comparison_dashboard(
        all_results,
        output_path=os.path.join(args.output, "strategy_comparison.html"),
        title="策略变体对比",
    )
    print(f"\n📊 对比仪表盘: {comparison_path}")

    return all_results


def run_custom_mode(args):
    """自定义模式"""
    return run_quick_mode(args)


def _print_results(evaluation: dict):
    """打印回测结果"""
    print(f"\n{'='*60}")
    print("  📈 回测结果")
    print(f"{'='*60}")

    perf = evaluation["策略表现"]
    bench = evaluation["基准表现"]
    trades = evaluation["交易统计"]
    config = evaluation["策略配置"]

    print(f"\n  【策略绩效】")
    print(f"  累计收益率:   {perf.get('累计收益率(%)', 0):>10.2f}%")
    print(f"  年化收益率:   {perf.get('年化收益率(%)', 0):>10.2f}%")
    print(f"  年化波动率:   {perf.get('年化波动率(%)', 0):>10.2f}%")
    print(f"  夏普比率:     {perf.get('夏普比率', 0):>10.2f}")
    print(f"  最大回撤:     {perf.get('最大回撤(%)', 0):>10.2f}%")
    print(f"  最大回撤区间: {perf.get('最大回撤开始日', '')} ~ {perf.get('最大回撤结束日', '')}")
    print(f"  日胜率:       {perf.get('日胜率(%)', 0):>10.2f}%")
    print(f"  Calmar比率:   {perf.get('Calmar比率', 0):>10.2f}")
    print(f"  收益回撤比:   {perf.get('收益回撤比', 0):>10.2f}")

    print(f"\n  【基准表现】")
    print(f"  累计收益率:   {bench.get('累计收益率(%)', 0):>10.2f}%")
    print(f"  年化收益率:   {bench.get('年化收益率(%)', 0):>10.2f}%")
    print(f"  最大回撤:     {bench.get('最大回撤(%)', 0):>10.2f}%")

    print(f"\n  【超额收益】")
    print(f"  超额收益:     {evaluation.get('超额收益(%)', 0):>10.2f}%")

    print(f"\n  【交易统计】")
    print(f"  交易次数:     {trades.get('交易次数', 0):>10}")
    print(f"  盈利次数:     {trades.get('盈利次数', 0):>10}")
    print(f"  亏损次数:     {trades.get('亏损次数', 0):>10}")
    print(f"  胜率:         {trades.get('胜率(%)', 0):>10.2f}%")
    print(f"  平均盈利:     {trades.get('平均盈利(%)', 0):>10.2f}%")
    print(f"  平均亏损:     {trades.get('平均亏损(%)', 0):>10.2f}%")
    print(f"  盈亏比:       {trades.get('盈亏比', 0):>10.2f}")
    print(f"  平均持仓天数: {trades.get('平均持仓天数', 0):>10.1f}天")

    print(f"\n  【策略配置】")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*60}\n")


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("  量化择时策略回测系统 v1.0")
    print("  基于量价斜率指标的市场择时")
    print("="*70)

    if args.no_price_filter:
        args.price_filter = False

    print(f"\n配置:")
    print(f"  模式: {args.mode}")
    print(f"  方法: {args.method}")
    print(f"  参数: N={args.n}, M={args.m}, S={args.s}")
    print(f"  价格过滤: {'关闭' if args.no_price_filter else '启用'}")
    print(f"  成交量过滤: {'启用' if args.volume_filter else '关闭'}")
    if args.end:
        print(f"  数据范围: {args.start} ~ {args.end}")
    else:
        print(f"  数据范围: {args.start} ~ 今天")

    # 根据模式执行
    mode_funcs = {
        "quick": run_quick_mode,
        "standard": run_standard_mode,
        "medium": run_medium_mode,
        "compare": run_compare_mode,
        "custom": run_custom_mode,
    }

    try:
        result = mode_funcs[args.mode](args)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ 回测完成！")
    print("\n⚠️ 以上内容由量化策略模型生成，仅供学习研究参考，不构成任何投资建议。投资有风险，决策需谨慎。")


if __name__ == "__main__":
    main()
