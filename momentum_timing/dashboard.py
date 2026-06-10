"""
交互式 HTML 仪表盘模块
使用 Plotly 生成丰富的交互式图表，所有内容整合在一个独立的 HTML 文件中。
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def build_dashboard(
    result: pd.DataFrame,
    evaluation: Dict,
    output_path: str = "output/dashboard.html",
    title: str = "量化择时策略回测仪表盘",
    language: str = "zh",
) -> str:
    """
    构建完整的交互式仪表盘 HTML 文件。

    参数:
        result: 回测结果 DataFrame（含所有策略列）
        evaluation: 评估结果字典
        output_path: 输出文件路径
        title: 仪表盘标题
        language: 语言 ("zh" 或 "en")

    返回:
        str: 输出文件的绝对路径
    """
    is_zh = language == "zh"

    # 标签文本
    labels = {
        "equity_curve": "策略净值" if is_zh else "Strategy NAV",
        "benchmark": "基准净值" if is_zh else "Benchmark NAV",
        "net_equity": "策略净值(扣费)" if is_zh else "Strategy NAV (After Cost)",
        "drawdown": "回撤" if is_zh else "Drawdown",
        "signal_buy": "买入信号" if is_zh else "Buy Signal",
        "signal_sell": "卖出信号" if is_zh else "Sell Signal",
        "pos": "持仓" if is_zh else "Position",
        "indicator": "择时指标" if is_zh else "Timing Indicator",
        "beta": "斜率(Beta)" if is_zh else "Slope (Beta)",
        "zscore": "标准分(Z-score)" if is_zh else "Standard Score",
        "r2": "拟合优度(R²)" if is_zh else "R-squared",
        "volume": "成交量" if is_zh else "Volume",
        "daily_return": "日收益率" if is_zh else "Daily Return",
        "strategy_return": "策略日收益" if is_zh else "Strategy Daily Return",
        "threshold_upper": "上阈值" if is_zh else "Upper Threshold",
        "threshold_lower": "下阈值" if is_zh else "Lower Threshold",
    }

    # 准备数据
    df = result.copy()
    dates = df["date"].values

    # 策略配置信息
    config = evaluation.get("策略配置", {})
    perf = evaluation.get("策略表现", {})
    bench = evaluation.get("基准表现", {})
    trades_info = evaluation.get("交易统计", {})

    # ============================================================
    # 创建 KPI 卡片
    # ============================================================
    kpi_data = [
        {"label": "年化收益率" if is_zh else "Annual Return", "value": f"{perf.get('年化收益率(%)', 0):.1f}%", "delta": f"+{perf.get('年化收益率(%)', 0) - bench.get('年化收益率(%)', 0):.1f}% vs 基准" if is_zh else f"+{perf.get('年化收益率(%)', 0) - bench.get('年化收益率(%)', 0):.1f}% vs Benchmark"},
        {"label": "夏普比率" if is_zh else "Sharpe Ratio", "value": f"{perf.get('夏普比率', 0):.2f}", "delta": "越高越好" if is_zh else "Higher is better"},
        {"label": "最大回撤" if is_zh else "Max DD", "value": f"{perf.get('最大回撤(%)', 0):.1f}%", "delta": f"基准: {bench.get('最大回撤(%)', 0):.1f}%" if is_zh else f"Bench: {bench.get('最大回撤(%)', 0):.1f}%"},
        {"label": "日胜率" if is_zh else "Daily Win Rate", "value": f"{perf.get('日胜率(%)', 0):.1f}%", "delta": "持仓天数占比" if is_zh else "Holding %"},
        {"label": "交易次数" if is_zh else "Trades", "value": f"{trades_info.get('交易次数', 0)}", "delta": f"胜率{trades_info.get('胜率(%)', 0):.0f}%" if is_zh else f"WR {trades_info.get('胜率(%)', 0):.0f}%"},
        {"label": "超额收益" if is_zh else "Excess Return", "value": f"{evaluation.get('超额收益(%)', 0):.1f}%", "delta": "vs 买入持有" if is_zh else "vs B&H"},
    ]

    # ============================================================
    # 图1: 净值曲线 + 买卖信号 + 回撤（主图）
    # ============================================================
    fig1 = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            labels["equity_curve"],
            labels["drawdown"],
            labels["pos"],
        ),
    )

    # 净值曲线
    fig1.add_trace(
        go.Scatter(
            x=dates, y=df["equity_curve_net"].values,
            mode="lines", name=labels["net_equity"],
            line=dict(color="#1f77b4", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>净值: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=dates, y=df["benchmark_curve"].values,
            mode="lines", name=labels["benchmark"],
            line=dict(color="#d3d3d3", width=1.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>基准: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 买入信号标记
    buy_dates = df[df["signal_raw"] == 1]["date"]
    buy_prices = df.loc[df["signal_raw"] == 1, "equity_curve_net"]
    if len(buy_dates) > 0:
        fig1.add_trace(
            go.Scatter(
                x=buy_dates, y=buy_prices,
                mode="markers", name=labels["signal_buy"],
                marker=dict(symbol="triangle-up", size=10, color="#2ca02c", line=dict(width=1, color="darkgreen")),
                hovertemplate="买入<br>%{x|%Y-%m-%d}<br>净值: %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # 卖出信号标记
    sell_dates = df[df["signal_raw"] == 0]["date"]
    sell_prices = df.loc[df["signal_raw"] == 0, "equity_curve_net"]
    # 只显示那些signal从1变为0的点
    sell_mask = np.diff(np.concatenate([[0], df["signal_raw"].values])) < -0.5
    sell_dates_real = df.loc[sell_mask, "date"]
    sell_prices_real = df.loc[sell_mask, "equity_curve_net"]
    if len(sell_dates_real) > 0:
        fig1.add_trace(
            go.Scatter(
                x=sell_dates_real, y=sell_prices_real,
                mode="markers", name=labels["signal_sell"],
                marker=dict(symbol="triangle-down", size=10, color="#d62728", line=dict(width=1, color="darkred")),
                hovertemplate="卖出<br>%{x|%Y-%m-%d}<br>净值: %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # 回撤曲线
    fig1.add_trace(
        go.Scatter(
            x=dates, y=df["drawdown"].values * 100,
            mode="lines", name=labels["drawdown"],
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.15)",
            line=dict(color="#d62728", width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>回撤: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    # 持仓状态
    fig1.add_trace(
        go.Scatter(
            x=dates, y=df["pos"].values,
            mode="lines", name=labels["pos"],
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.2)",
            line=dict(color="#2ca02c", width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>持仓: %{y:.0f}<extra></extra>",
        ),
        row=3, col=1,
    )

    fig1.update_yaxes(title_text="净值" if is_zh else "NAV", row=1, col=1, type="log")
    fig1.update_yaxes(title_text="回撤 %" if is_zh else "DD %", row=2, col=1)
    fig1.update_yaxes(title_text="持仓" if is_zh else "Position", row=3, col=1, range=[-0.1, 1.1])
    fig1.update_layout(
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        margin=dict(l=60, r=30, t=60, b=30),
    )

    # ============================================================
    # 图2: 指标面板（信号指标、Beta、Z-score、R²）
    # ============================================================
    fig2 = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.35, 0.3, 0.35],
        subplot_titles=(
            labels["indicator"],
            labels["beta"],
            labels["r2"],
        ),
    )

    # 指标值
    if "indicator" in df.columns:
        fig2.add_trace(
            go.Scatter(
                x=dates, y=df["indicator"].values,
                mode="lines", name=labels["indicator"],
                line=dict(color="#9467bd", width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>指标: %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )
        # 添加上下阈值线
        thr_val = evaluation.get("策略配置", {}).get("信号阈值", 0.7)
        if isinstance(thr_val, str):
            # slope 方法的非对称阈值，跳过
            pass
        else:
            thr = float(thr_val)
            fig2.add_hline(y=thr, line_dash="dash", line_color="#2ca02c", row=1, col=1,
                           annotation_text=f"+{thr}", annotation_position="top right")
            fig2.add_hline(y=-thr, line_dash="dash", line_color="#d62728", row=1, col=1,
                           annotation_text=f"-{thr}", annotation_position="bottom right")
        # 零线
        fig2.add_hline(y=0, line_color="gray", line_width=0.5, row=1, col=1)

    # Beta
    if "beta" in df.columns:
        fig2.add_trace(
            go.Scatter(
                x=dates, y=df["beta"].values,
                mode="lines", name=labels["beta"],
                line=dict(color="#ff7f0e", width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>Beta: %{y:.4f}<extra></extra>",
            ),
            row=2, col=1,
        )
        # Beta均值线
        fig2.add_hline(y=1.0, line_dash="dot", line_color="gray", row=2, col=1)

    # Z-score
    if "zscore" in df.columns:
        fig2.add_trace(
            go.Scatter(
                x=dates, y=df["zscore"].values,
                mode="lines", name=labels["zscore"],
                line=dict(color="#1f77b4", width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>Z-score: %{y:.4f}<extra></extra>",
            ),
            row=3, col=1,
        )
        fig2.add_hline(y=0, line_color="gray", line_width=0.5, row=3, col=1)

    fig2.update_layout(
        height=650,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        margin=dict(l=60, r=30, t=60, b=30),
    )

    # ============================================================
    # 图3: 收益分布分析
    # ============================================================
    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "日收益率分布" if is_zh else "Daily Return Distribution",
            "月度收益热力图" if is_zh else "Monthly Return Heatmap",
        ),
    )

    # 日收益直方图
    strat_returns = df["strategy_return"].values * 100
    bench_returns = df["daily_return"].values * 100
    fig3.add_trace(
        go.Histogram(
            x=strat_returns, name=labels["strategy_return"],
            marker_color="#1f77b4", opacity=0.7,
            nbinsx=50,
            hovertemplate="收益: %{x:.2f}%<br>频率: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig3.add_trace(
        go.Histogram(
            x=bench_returns, name=labels["daily_return"],
            marker_color="#d3d3d3", opacity=0.5,
            nbinsx=50,
            hovertemplate="收益: %{x:.2f}%<br>频率: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 月度收益热力图
    monthly_returns = _compute_monthly_returns(df)
    if monthly_returns is not None and len(monthly_returns) > 0:
        fig3.add_trace(
            go.Heatmap(
                z=monthly_returns.values,
                x=monthly_returns.columns.tolist(),
                y=monthly_returns.index.tolist(),
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in monthly_returns.values],
                texttemplate="%{text}",
                textfont={"size": 8},
                hovertemplate="%{y}年%{x}月<br>收益: %{z:.2f}%<extra></extra>",
            ),
            row=1, col=2,
        )

    fig3.update_layout(
        height=400,
        template="plotly_dark",
        margin=dict(l=60, r=30, t=60, b=30),
    )
    fig3.update_yaxes(title_text="频率" if is_zh else "Frequency", row=1, col=1)
    fig3.update_xaxes(title_text="日收益率 (%)" if is_zh else "Daily Return (%)", row=1, col=1)

    # ============================================================
    # 图4: 滚动性能指标
    # ============================================================
    fig4 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "滚动年化收益率 (252日)" if is_zh else "Rolling Annual Return (252d)",
            "滚动最大回撤 (252日)" if is_zh else "Rolling Max Drawdown (252d)",
        ),
    )

    # 滚动年化收益
    roll_returns = _compute_rolling_annual_return(df["strategy_return"].values, 252)
    fig4.add_trace(
        go.Scatter(
            x=dates, y=roll_returns,
            mode="lines", name="滚动年化收益" if is_zh else "Rolling Return",
            line=dict(color="#1f77b4", width=1.5),
            fill="tozeroy", fillcolor="rgba(31, 119, 180, 0.1)",
            hovertemplate="%{x|%Y-%m-%d}<br>年化收益: %{y:.1f}%<extra></extra>",
        ),
        row=1, col=1,
    )
    fig4.add_hline(y=0, line_color="gray", line_width=0.5, row=1, col=1)

    # 滚动最大回撤
    roll_dd = _compute_rolling_max_dd(df["equity_curve_net"].values, 252)
    fig4.add_trace(
        go.Scatter(
            x=dates, y=roll_dd,
            mode="lines", name="滚动回撤" if is_zh else "Rolling DD",
            line=dict(color="#d62728", width=1.5),
            fill="tozeroy", fillcolor="rgba(214, 39, 40, 0.1)",
            hovertemplate="%{x|%Y-%m-%d}<br>最大回撤: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig4.update_yaxes(title_text="%", row=1, col=1)
    fig4.update_yaxes(title_text="%", row=2, col=1)
    fig4.update_layout(
        height=500,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=60, r=30, t=60, b=30),
    )

    # ============================================================
    # 图5: 交易分析
    # ============================================================
    trades = trades_info.get("交易明细", [])
    fig5 = None
    if trades and len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        fig5 = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "单笔收益分布" if is_zh else "Trade Return Distribution",
                "持仓天数分布" if is_zh else "Holding Days Distribution",
                "累计收益曲线" if is_zh else "Cumulative Trade Return",
                "收益 vs 持仓天数" if is_zh else "Return vs Holding Days",
            ),
        )

        # 单笔收益柱状图
        colors = ["#2ca02c" if r > 0 else "#d62728" for r in trades_df["收益率(%)"]]
        fig5.add_trace(
            go.Bar(
                x=list(range(len(trades_df))),
                y=trades_df["收益率(%)"],
                marker_color=colors,
                name="交易收益",
                hovertemplate="交易#%{x}<br>收益: %{y:.2f}%<extra></extra>",
            ),
            row=1, col=1,
        )

        # 持仓天数分布
        fig5.add_trace(
            go.Histogram(
                x=trades_df["持仓天数"],
                marker_color="#9467bd",
                nbinsx=20,
                name="持仓天数",
                hovertemplate="天数: %{x}<br>频率: %{y}<extra></extra>",
            ),
            row=1, col=2,
        )

        # 累计收益
        cum_returns = trades_df["收益率(%)"].cumsum()
        fig5.add_trace(
            go.Scatter(
                x=list(range(len(trades_df))),
                y=cum_returns,
                mode="lines+markers",
                marker=dict(size=6, color=colors),
                line=dict(color="#1f77b4", width=1.5),
                name="累计收益",
                hovertemplate="交易#%{x}<br>累计: %{y:.2f}%<extra></extra>",
            ),
            row=2, col=1,
        )
        fig5.add_hline(y=0, line_color="gray", line_width=0.5, row=2, col=1)

        # 收益 vs 持仓天数散点图
        fig5.add_trace(
            go.Scatter(
                x=trades_df["持仓天数"],
                y=trades_df["收益率(%)"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=trades_df["收益率(%)"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="收益%"),
                ),
                name="收益/天数",
                hovertemplate="持仓: %{x}天<br>收益: %{y:.2f}%<extra></extra>",
            ),
            row=2, col=2,
        )

        fig5.update_layout(
            height=650,
            template="plotly_dark",
            margin=dict(l=60, r=30, t=60, b=30),
            showlegend=False,
        )

    # ============================================================
    # 年化收益柱状图
    # ============================================================
    annual_returns_data = perf.get("年化收益序列", [])
    fig6 = None
    if annual_returns_data:
        years = [a["year"] for a in annual_returns_data]
        yr_returns = [a["return"] for a in annual_returns_data]
        colors_annual = ["#2ca02c" if r > 0 else "#d62728" for r in yr_returns]

        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            x=years, y=yr_returns,
            marker_color=colors_annual,
            text=[f"{r:.1f}%" for r in yr_returns],
            textposition="outside",
            hovertemplate="%{x}年<br>收益: %{y:.2f}%<extra></extra>",
        ))
        fig6.add_hline(y=0, line_color="gray", line_width=0.5)
        fig6.update_layout(
            title="各年度收益率" if is_zh else "Annual Returns",
            height=350,
            template="plotly_dark",
            margin=dict(l=60, r=30, t=60, b=30),
        )

    # ============================================================
    # 构建完整 HTML
    # ============================================================

    # 将 Plotly figures 转为 HTML div
    fig1_html = pio.to_html(fig1, full_html=False, include_plotlyjs="cdn", div_id="chart1")
    fig2_html = pio.to_html(fig2, full_html=False, include_plotlyjs=False, div_id="chart2")
    fig3_html = pio.to_html(fig3, full_html=False, include_plotlyjs=False, div_id="chart3")
    fig4_html = pio.to_html(fig4, full_html=False, include_plotlyjs=False, div_id="chart4")
    fig5_html = pio.to_html(fig5, full_html=False, include_plotlyjs=False, div_id="chart5") if fig5 else ""
    fig6_html = pio.to_html(fig6, full_html=False, include_plotlyjs=False, div_id="chart6") if fig6 else ""

    # KPI 卡片 HTML
    kpi_cards_html = ""
    for kpi in kpi_data:
        kpi_cards_html += f"""
        <div class="kpi-card">
            <div class="kpi-label">{kpi['label']}</div>
            <div class="kpi-value">{kpi['value']}</div>
            <div class="kpi-delta">{kpi['delta']}</div>
        </div>
        """

    # 配置信息表
    config_rows = ""
    for k, v in config.items():
        config_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    # 交易统计表
    trade_rows = ""
    exclude_keys = ["交易明细"]
    for k, v in trades_info.items():
        if k not in exclude_keys:
            trade_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    perf_rows = ""
    for k, v in perf.items():
        if k not in ["年化收益序列"]:
            perf_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    bench_rows = ""
    for k, v in bench.items():
        bench_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    # 交易历史表
    trade_history_rows = ""
    if trades:
        for t in trades[:30]:  # 最多显示30条
            cls = "win" if t["收益率(%)"] > 0 else "loss"
            trade_history_rows += f"""
            <tr class="{cls}">
                <td>{t['开始日期']}</td>
                <td>{t['结束日期']}</td>
                <td>{t['持仓天数']}</td>
                <td>{t['收益率(%)']:.2f}%</td>
            </tr>
            """

    data_range = evaluation.get("数据日期", {})
    date_info = f"{data_range.get('开始', '')} ~ {data_range.get('结束', '')} ({data_range.get('天数', '')}天)"

    html_content = f"""<!DOCTYPE html>
<html lang="{'zh-CN' if is_zh else 'en'}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        *,*::before,*::after {{ box-sizing: border-box; margin: 0; padding: 0 }}
        :root {{
          --bg: #080c14;
          --bg2: rgba(14,18,28,0.85);
          --bg3: rgba(20,26,40,0.7);
          --bd: rgba(255,255,255,0.06);
          --t1: #e8ecf1;
          --t2: #7a8599;
          --t3: #4a5568;
          --gold: #d4a562;
          --gr: #34d399;
          --rd: #f87171;
          --fd: Georgia, Times New Roman, serif;
          --fb: -apple-system, BlinkMacSystemFont, Segoe UI, PingFang SC, Microsoft YaHei, sans-serif;
          --fm: SF Mono, Cascadia Code, JetBrains Mono, monospace;
        }}
        html {{ scroll-behavior: smooth; background: var(--bg); color: var(--t1); font-family: var(--fb); font-size: 16px; line-height: 1.6; -webkit-font-smoothing: antialiased }}
        body {{ overflow-x: hidden }}
        ::-webkit-scrollbar {{ width: 6px }}
        ::-webkit-scrollbar-track {{ background: var(--bg) }}
        ::-webkit-scrollbar-thumb {{ background: var(--t3); border-radius: 3px }}

        .header {{
            position: relative; z-index: 1;
            min-height: 40vh; display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            padding: 60px 40px; text-align: center;
            background: linear-gradient(180deg, rgba(8,12,20,0) 0%, var(--bg) 100%);
        }}
        .header h1 {{ font-family: var(--fd); font-size: clamp(2rem, 5vw, 3.5rem); font-weight: 400; line-height: 1.1; letter-spacing: -0.02em; margin-bottom: 16px; color: var(--t1) }}
        .header .subtitle {{ font-size: 1.1rem; color: var(--t2); margin-bottom: 8px }}
        .header .date-range {{ font-family: var(--fm); font-size: 0.9rem; color: var(--t3); margin-top: 5px }}

        .container {{ position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; padding: 0 40px }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg3);
            border: 1px solid var(--bd);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: transform 0.35s, border-color 0.35s, box-shadow 0.35s;
            cursor: default;
        }}
        .kpi-card:hover {{ transform: translateY(-4px); border-color: rgba(212,165,98,0.25); box-shadow: 0 12px 40px rgba(0,0,0,0.3), 0 0 0 1px rgba(212,165,98,0.1) inset }}
        .kpi-label {{ font-size: 0.75rem; color: var(--t3); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em }}
        .kpi-value {{ font-family: var(--fd); font-size: 2rem; font-weight: 400; letter-spacing: -0.02em; line-height: 1.1; color: var(--t1); margin-bottom: 4px; transition: transform 0.3s }}
        .kpi-card:hover .kpi-value {{ transform: scale(1.05) }}
        .kpi-delta {{ font-size: 0.7rem; color: var(--t3); margin-top: 6px; opacity: 0.7; transition: opacity 0.3s }}
        .kpi-card:hover .kpi-delta {{ opacity: 1 }}

        .section {{
            background: var(--bg2);
            border: 1px solid var(--bd);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
            transition: border-color 0.4s, box-shadow 0.4s;
        }}
        .section:hover {{ border-color: rgba(212,165,98,0.15); box-shadow: 0 0 60px rgba(212,165,98,0.04) }}
        .section::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(212,165,98,0.15), transparent); opacity: 0.5 }}
        .section-title {{
            font-family: var(--fd);
            font-size: 1.15rem;
            color: var(--t1);
            margin-bottom: 20px;
            letter-spacing: -0.01em;
        }}

        .tables-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        .info-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-family: var(--fm);
            font-size: 0.85rem;
        }}
        .info-table td:first-child {{
            color: var(--t3);
            width: 40%;
            font-weight: 500;
        }}
        .info-table td:last-child {{
            color: var(--t1);
            font-weight: 600;
        }}

        .trade-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        .trade-table th {{
            text-align: left;
            padding: 12px 16px;
            color: var(--t3);
            font-weight: 500;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border-bottom: 1px solid var(--bd);
            position: sticky;
            top: 0;
            background: var(--bg2);
            z-index: 2;
        }}
        .trade-table td {{
            padding: 10px 16px;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            font-family: var(--fm);
            font-size: 0.8rem;
        }}
        .trade-table tbody tr:nth-child(even) td {{ background: rgba(255,255,255,0.01) }}
        .trade-table tr.win td {{ color: var(--gr) }}
        .trade-table tr.loss td {{ color: var(--rd) }}
        .trade-table tbody tr:hover td {{ background: rgba(212,165,98,0.06) !important }}

        .chart-container {{ margin: 15px 0; }}

        .footer {{
            text-align: center;
            padding: 60px 40px;
            border-top: 1px solid var(--bd);
            margin-top: 60px;
        }}
        .footer p {{ font-size: 0.8rem; color: var(--t3) }}
        .footer a {{ color: var(--t3) }}

        .disclaimer {{
            background: rgba(251,191,36,0.06);
            border: 1px solid rgba(251,191,36,0.15);
            border-radius: 12px;
            padding: 20px 24px;
            font-size: 0.8rem;
            color: #b49450;
            line-height: 1.6;
            margin-top: 32px;
        }}

        .nav-bar {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(8,12,20,0.92);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--bd);
            padding: 12px 0;
        }}
        .nav-inner {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            gap: 20px;
            padding: 0 40px;
            flex-wrap: wrap;
        }}
        .nav-link {{
            color: var(--t3);
            text-decoration: none;
            font-size: 0.9em;
            font-weight: 500;
            padding: 5px 10px;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .nav-link:hover {{ color: var(--gold); background: rgba(212,165,98,0.08) }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">{config.get('策略方法', '')} | N={config.get('回归周期(N)', '')} M={config.get('标准分周期(M)', '')} S={config.get('信号阈值', '')}</div>
        <div class="date-range">{date_info}</div>
    </div>

    <div class="nav-bar">
        <div class="nav-inner">
            <a href="#overview" class="nav-link">{'概览' if is_zh else 'Overview'}</a>
            <a href="#charts" class="nav-link">{'图表分析' if is_zh else 'Charts'}</a>
            <a href="#indicators" class="nav-link">{'指标分析' if is_zh else 'Indicators'}</a>
            <a href="#performance" class="nav-link">{'绩效分析' if is_zh else 'Performance'}</a>
            <a href="#trades" class="nav-link">{'交易记录' if is_zh else 'Trade Log'}</a>
        </div>
    </div>

    <div class="container">
        <!-- KPI 概览 -->
        <div id="overview" class="kpi-grid">
            {kpi_cards_html}
        </div>

        <!-- 统计表格 -->
        <div class="section">
            <span class="section-title">{'策略绩效对比' if is_zh else 'Performance Comparison'}</span>
            <div class="tables-grid">
                <table class="info-table">
                    <tr><th colspan="2" style="text-align:left; padding: 8px 12px; color: #1a1a2e; font-size:1.1em;">{'策略表现' if is_zh else 'Strategy'}</th></tr>
                    {perf_rows}
                </table>
                <table class="info-table">
                    <tr><th colspan="2" style="text-align:left; padding: 8px 12px; color: #636e72; font-size:1.1em;">{'基准表现' if is_zh else 'Benchmark'}</th></tr>
                    {bench_rows}
                </table>
                <table class="info-table">
                    <tr><th colspan="2" style="text-align:left; padding: 8px 12px; color: #1a1a2e; font-size:1.1em;">{'策略配置' if is_zh else 'Config'}</th></tr>
                    {config_rows}
                </table>
                <table class="info-table">
                    <tr><th colspan="2" style="text-align:left; padding: 8px 12px; color: #636e72; font-size:1.1em;">{'交易统计' if is_zh else 'Trade Stats'}</th></tr>
                    {trade_rows}
                </table>
            </div>
        </div>

        <!-- 图表区 -->
        <div id="charts" class="section">
            <span class="section-title">{'净值曲线与回撤' if is_zh else 'Equity Curve & Drawdown'}</span>
            <div class="chart-container">{fig1_html}</div>
        </div>

        <div id="indicators" class="section">
            <span class="section-title">{'择时指标分解' if is_zh else 'Timing Indicator Decomposition'}</span>
            <div class="chart-container">{fig2_html}</div>
        </div>

        <div id="performance" class="section">
            <span class="section-title">{'收益分布分析' if is_zh else 'Return Distribution Analysis'}</span>
            <div class="chart-container">{fig3_html}</div>
            {'<div class="chart-container">' + fig4_html + '</div>' if fig4 else ''}
            {'<div class="chart-container">' + fig6_html + '</div>' if fig6 else ''}
        </div>

        <div id="trades" class="section">
            <span class="section-title">{'交易分析' if is_zh else 'Trade Analysis'}</span>
            {'<div class="chart-container">' + fig5_html + '</div>' if fig5 else '<p>暂无交易记录</p>' if is_zh else '<p>No trades found</p>'}

            <h3 style="margin-top: 20px;">{'交易明细' if is_zh else 'Trade Details'}</h3>
            <div style="max-height: 400px; overflow-y: auto;">
                <table class="trade-table">
                    <thead>
                        <tr>
                            <th>{'开始日期' if is_zh else 'Entry'}</th>
                            <th>{'结束日期' if is_zh else 'Exit'}</th>
                            <th>{'持仓天数' if is_zh else 'Days'}</th>
                            <th>{'收益率' if is_zh else 'Return'}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trade_history_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 免责声明 -->
        <div class="disclaimer">
            ⚠️ {'以上内容由量化策略模型生成，仅供学习研究参考，不构成任何投资建议或个股推荐。回测收益不代表未来表现。投资有风险，决策需谨慎。' if is_zh else 'The above content is generated by a quantitative strategy model for educational and research purposes only. It does not constitute investment advice. Past performance does not guarantee future results. Investing carries risk; make your own decisions carefully.'}
        </div>
    </div>

    <div class="footer">
        <p>{'生成时间' if is_zh else 'Generated'}: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {'基于 baostock 免费数据' if is_zh else 'Data: baostock'}</p>
    </div>
</body>
</html>"""

    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return os.path.abspath(output_path)


def _compute_monthly_returns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """计算月度收益率矩阵"""
    temp = df.copy()
    temp["year"] = temp["date"].dt.year
    temp["month"] = temp["date"].dt.month
    monthly = temp.groupby(["year", "month"])["strategy_return"].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack()

    # 确保月份列完整
    for m in range(1, 13):
        if m not in monthly.columns:
            monthly[m] = np.nan
    monthly = monthly[sorted(monthly.columns)]

    monthly.index = monthly.index.astype(str)
    monthly.columns = [f"{m}月" for m in monthly.columns]

    return monthly


def _compute_rolling_annual_return(returns: np.ndarray, window: int = 252) -> np.ndarray:
    """计算滚动年化收益率"""
    result = np.full(len(returns), np.nan)
    for i in range(window - 1, len(returns)):
        segment = returns[i - window + 1 : i + 1]
        total_return = (1 + segment).prod()
        n_years = window / 252
        result[i] = (total_return ** (1 / n_years) - 1) * 100
    return result


def _compute_rolling_max_dd(equity: np.ndarray, window: int = 252) -> np.ndarray:
    """计算滚动最大回撤"""
    result = np.full(len(equity), np.nan)
    for i in range(window - 1, len(equity)):
        segment = equity[i - window + 1 : i + 1]
        peak = np.maximum.accumulate(segment)
        dd = (segment - peak) / peak
        result[i] = np.min(dd) * 100
    return result


def create_comparison_dashboard(
    results_dict: Dict[str, pd.DataFrame],
    output_path: str = "output/comparison_dashboard.html",
    title: str = "多标的策略对比仪表盘",
    language: str = "zh",
) -> str:
    """
    创建多标的对比仪表盘。

    参数:
        results_dict: {名称: 回测结果DataFrame}
        output_path: 输出路径
        title: 标题
        language: 语言
    """
    is_zh = language == "zh"
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (name, df) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        # 归一化到 1
        equity = df["equity_curve_net"].values / df["equity_curve_net"].values[0]
        fig.add_trace(go.Scatter(
            x=df["date"], y=equity,
            mode="lines", name=name,
            line=dict(color=color, width=2),
            hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<br>净值: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="日期" if is_zh else "Date",
        yaxis_title="净值" if is_zh else "NAV",
        yaxis_type="log",
        height=600,
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # 获取 Plotly 图表 HTML（不含完整 HTML 骨架）
    plotly_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    html_content = f"""<!DOCTYPE html>
<html lang="{'zh-CN' if is_zh else 'en'}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        *,*::before,*::after {{ box-sizing: border-box; margin: 0; padding: 0 }}
        :root {{
          --bg: #080c14;
          --bg2: rgba(14,18,28,0.85);
          --bg3: rgba(20,26,40,0.7);
          --bd: rgba(255,255,255,0.06);
          --t1: #e8ecf1;
          --t2: #7a8599;
          --t3: #4a5568;
          --gold: #d4a562;
          --fd: Georgia, Times New Roman, serif;
          --fb: -apple-system, BlinkMacSystemFont, Segoe UI, PingFang SC, Microsoft YaHei, sans-serif;
          --fm: SF Mono, Cascadia Code, JetBrains Mono, monospace;
        }}
        html {{ scroll-behavior: smooth; background: var(--bg); color: var(--t1); font-family: var(--fb); font-size: 16px; line-height: 1.6; -webkit-font-smoothing: antialiased }}
        body {{ overflow-x: hidden }}
        ::-webkit-scrollbar {{ width: 6px }}
        ::-webkit-scrollbar-track {{ background: var(--bg) }}
        ::-webkit-scrollbar-thumb {{ background: var(--t3); border-radius: 3px }}

        .hero {{
            position: relative; z-index: 1;
            min-height: 30vh; display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            padding: 60px 40px; text-align: center;
            background: linear-gradient(180deg, rgba(8,12,20,0) 0%, var(--bg) 100%);
        }}
        .hero-title {{ font-family: var(--fd); font-size: clamp(2rem, 5vw, 3.5rem); font-weight: 400; line-height: 1.1; letter-spacing: -0.02em; margin-bottom: 16px; color: var(--t1) }}
        .hero-title em {{ font-style: normal; color: var(--gold) }}
        .hero-subtitle {{ font-size: 1.1rem; color: var(--t2); max-width: 600px; }}

        .container {{ position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; padding: 0 40px 60px }}

        .chart-panel {{
            background: var(--bg2);
            border: 1px solid var(--bd);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
            transition: border-color 0.4s, box-shadow 0.4s;
        }}
        .chart-panel:hover {{ border-color: rgba(212,165,98,0.15); box-shadow: 0 0 60px rgba(212,165,98,0.04) }}
        .chart-panel::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(212,165,98,0.15), transparent); opacity: 0.5 }}

        .footer {{
            text-align: center;
            padding: 60px 40px;
            border-top: 1px solid var(--bd);
            margin-top: 60px;
        }}
        .footer p {{ font-size: 0.8rem; color: var(--t3) }}
    </style>
</head>
<body>
    <section class="hero">
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">多标的策略净值对比 · 对数坐标</p>
    </section>

    <div class="container">
        <div class="chart-panel">
            {plotly_html}
        </div>
    </div>

    <div class="footer">
        <p>{'生成时间' if is_zh else 'Generated'}: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {'基于 baostock 免费数据' if is_zh else 'Data: baostock'}</p>
    </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return os.path.abspath(output_path)
