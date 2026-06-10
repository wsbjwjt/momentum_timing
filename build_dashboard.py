#!/usr/bin/env python3
"""
生成炸场级交互式策略回测仪表盘 HTML
Cinematic Dashboard — single-file, D3-powered, scroll-driven narrative
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ── 路径 ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
EQUITY_CSV = os.path.join(BASE, "output", "equity.csv")
SUMMARY_JSON = os.path.join(BASE, "output", "summary.json")
OUTPUT_HTML = os.path.join(BASE, "output", "dashboard.html")

# ── 加载数据 ──────────────────────────────────────────
df = pd.read_csv(EQUITY_CSV, parse_dates=["date"])
with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
    summary = json.load(f)

perf = summary["策略表现"]
bench = summary["基准表现"]
trades_info = summary["交易统计"]
config = summary["策略配置"]
data_range = summary["数据日期"]
trades_list = trades_info.get("交易明细", [])

# ── 准备嵌入数据 ─────────────────────────────────────
# 1. 净值曲线（完整数据）
equity_data = []
for _, row in df.iterrows():
    equity_data.append({
        "d": row["date"].strftime("%Y-%m-%d"),
        "s": round(float(row["equity_curve_net"]), 4),
        "b": round(float(row["benchmark_curve"]), 4),
        "dd": round(float(row["drawdown"]) * 100, 2),
        "pos": int(row["pos"]),
        "sig": float(row["signal"]),
    })

# 2. 采样数据（用于部分图表，减少数据量）
sample_n = min(800, len(df))
sample_idx = np.linspace(0, len(df) - 1, sample_n, dtype=int)
sampled = []
for i in sample_idx:
    row = df.iloc[i]
    sampled.append({
        "d": row["date"].strftime("%Y-%m-%d"),
        "s": round(float(row["equity_curve_net"]), 3),
        "b": round(float(row["benchmark_curve"]), 3),
    })

# 3. 月度收益矩阵
temp = df.copy()
temp["year"] = temp["date"].dt.year
temp["month"] = temp["date"].dt.month
monthly = temp.groupby(["year", "month"])["strategy_return"].apply(lambda x: (1 + x).prod() - 1).unstack()
for m in range(1, 13):
    if m not in monthly.columns:
        monthly[m] = np.nan
monthly = monthly[sorted(monthly.columns)]
monthly_heatmap = {
    "years": monthly.index.astype(int).tolist(),
    "months": [f"{m}月" for m in monthly.columns],
    "data": [[round(float(v) * 100, 1) if not np.isnan(v) else None for v in row] for row in monthly.values],
}

# 4. 年度收益序列
annual_ret = []
for y in monthly.index:
    yr_data = monthly.loc[y]
    valid = yr_data.dropna()
    if len(valid) > 0:
        yr = (1 + valid).prod() - 1
        annual_ret.append({"year": int(y), "return": round(float(yr) * 100, 1)})

# 5. 滚动年化收益（252日）
window = 252
rolling_annual = []
for i in range(window - 1, len(df)):
    seg = df["strategy_return"].iloc[i - window + 1 : i + 1].values
    total = (1 + seg).prod()
    ann = (total ** (1 / (window / 252)) - 1) * 100
    rolling_annual.append({
        "d": df["date"].iloc[i].strftime("%Y-%m-%d"),
        "v": round(float(ann), 1),
    })

# 6. 滚动最大回撤
rolling_dd = []
for i in range(window - 1, len(df)):
    eq = df["equity_curve_net"].iloc[i - window + 1 : i + 1].values
    peak = np.maximum.accumulate(eq)
    dd = np.min((eq - peak) / peak) * 100
    rolling_dd.append({
        "d": df["date"].iloc[i].strftime("%Y-%m-%d"),
        "v": round(float(dd), 1),
    })

# 7. 交易数据
trades_json = []
for t in trades_list:
    trades_json.append({
        "start": t["开始日期"],
        "end": t["结束日期"],
        "days": t["持仓天数"],
        "ret": t["收益率(%)"],
    })

# 8. 累计交易收益序列
trades_cum = []
cum = 0
for t in trades_list:
    cum += t["收益率(%)"]
    trades_cum.append({"i": len(trades_cum) + 1, "cum": round(cum, 2), "ret": t["收益率(%)"]})

# 9. 策略配置用于展示
config_display = [
    {"label": "指标方法", "value": config["策略方法"]},
    {"label": "回归周期 N", "value": config["回归周期(N)"]},
    {"label": "标准分周期 M", "value": config["标准分周期(M)"]},
    {"label": "信号阈值 S", "value": config["信号阈值"]},
    {"label": "价格过滤", "value": config["价格过滤"]},
    {"label": "成交量过滤", "value": config["成交量过滤"]},
]

# ── 组装嵌入数据 ────────────────────────────────────
embedded = {
    "equity": equity_data,
    "sampled": sampled,
    "trades": trades_json,
    "tradesCum": trades_cum,
    "perf": {
        "totalReturn": perf["累计收益率(%)"],
        "annualReturn": perf["年化收益率(%)"],
        "volatility": perf["年化波动率(%)"],
        "sharpe": perf["夏普比率"],
        "maxDD": perf["最大回撤(%)"],
        "ddStart": perf["最大回撤开始日"],
        "ddEnd": perf["最大回撤结束日"],
        "calmar": perf["Calmar比率"],
        "dailyWin": perf["日胜率(%)"],
    },
    "bench": {
        "totalReturn": bench["累计收益率(%)"],
        "annualReturn": bench["年化收益率(%)"],
        "maxDD": bench["最大回撤(%)"],
    },
    "tradeStats": {
        "count": trades_info["交易次数"],
        "winCount": trades_info["盈利次数"],
        "lossCount": trades_info["亏损次数"],
        "winRate": trades_info["胜率(%)"],
        "avgWin": trades_info["平均盈利(%)"],
        "avgLoss": trades_info["平均亏损(%)"],
        "plRatio": trades_info["盈亏比"],
        "bestTrade": trades_info["最大单笔盈利(%)"],
        "worstTrade": trades_info["最大单笔亏损(%)"],
        "avgDays": trades_info["平均持仓天数"],
    },
    "excessReturn": summary["超额收益(%)"],
    "dateRange": {
        "start": data_range["开始"],
        "end": data_range["结束"],
        "days": data_range["天数"],
    },
    "monthlyHeatmap": monthly_heatmap,
    "annualReturns": annual_ret,
    "rollingAnnual": rolling_annual,
    "rollingDD": rolling_dd,
    "config": config_display,
}

embedded_json = json.dumps(embedded, ensure_ascii=False)

# ── 写入 HTML ────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化择时策略 · 回测仪表盘</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
/* ── CSS Reset & Variables ────────────────────────── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg-deep:#080c14;
  --bg-panel:rgba(14,18,28,0.85);
  --bg-card:rgba(20,26,40,0.7);
  --border:rgba(255,255,255,0.06);
  --text-primary:#e8ecf1;
  --text-secondary:#7a8599;
  --text-muted:#4a5568;
  --accent-gold:#d4a562;
  --accent-gold-dim:rgba(212,165,98,0.15);
  --accent-green:#34d399;
  --accent-red:#f87171;
  --accent-cyan:#22d3ee;
  --accent-blue:#60a5fa;
  --font-display:'Georgia','Times New Roman',serif;
  --font-body:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
  --font-mono:'SF Mono','Cascadia Code','JetBrains Mono',monospace;
  --ease-out-expo:cubic-bezier(0.16,1,0.3,1);
  --ease-out-quart:cubic-bezier(0.25,1,0.5,1);
}}
html{{scroll-behavior:smooth;background:var(--bg-deep);color:var(--text-primary);font-family:var(--font-body);font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}}
body{{overflow-x:hidden}}
::-webkit-scrollbar{{width:6px}}
::-webkit-scrollbar-track{{background:var(--bg-deep)}}
::-webkit-scrollbar-thumb{{background:var(--text-muted);border-radius:3px}}

/* ── Background ──────────────────────────────────── */
.bg-canvas{{position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none}}
.bg-glow{{position:fixed;width:600px;height:600px;border-radius:50%;filter:blur(120px);opacity:0.08;pointer-events:none;z-index:0}}
.bg-glow-1{{top:-200px;right:-100px;background:radial-gradient(circle,var(--accent-gold),transparent 70%)}}
.bg-glow-2{{bottom:-200px;left:-100px;background:radial-gradient(circle,var(--accent-cyan),transparent 70%)}}
.bg-glow-3{{top:50%;left:50%;transform:translate(-50%,-50%);background:radial-gradient(circle,var(--accent-blue),transparent 70%);width:800px;height:800px}}

/* ── Hero Section ────────────────────────────────── */
.hero{{position:relative;z-index:1;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 40px;text-align:center}}
.hero-eyebrow{{font-family:var(--font-mono);font-size:0.75rem;letter-spacing:0.35em;text-transform:uppercase;color:var(--accent-gold);margin-bottom:24px;animation:heroFadeIn 0.8s cubic-bezier(0.16,1,0.3,1) 0.2s both}}
.hero-title{{font-family:var(--font-display);font-size:clamp(2.5rem,6vw,5rem);font-weight:400;line-height:1.1;letter-spacing:-0.02em;margin-bottom:16px;animation:heroFadeIn 0.8s cubic-bezier(0.16,1,0.3,1) 0.4s both}}
.hero-title em{{font-style:normal;color:var(--accent-gold)}}
.hero-subtitle{{font-size:1.1rem;color:var(--text-secondary);max-width:600px;margin-bottom:48px;animation:heroFadeIn 0.8s cubic-bezier(0.16,1,0.3,1) 0.6s both}}
.hero-meta{{display:flex;gap:32px;font-family:var(--font-mono);font-size:0.8rem;color:var(--text-muted);animation:heroFadeIn 0.8s cubic-bezier(0.16,1,0.3,1) 0.8s both}}
.hero-meta span{{display:flex;align-items:center;gap:8px}}
.hero-meta .dot{{width:6px;height:6px;border-radius:50%;background:var(--accent-gold)}}

@keyframes heroFadeIn{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
/* Reveal defaults — visible by default, animation as enhancement */
.reveal{{transition:opacity 0.8s cubic-bezier(0.16,1,0.3,1),transform 0.8s cubic-bezier(0.16,1,0.3,1)}}
.reveal.staggered{{opacity:0;transform:translateY(30px)}}
.reveal.staggered.visible{{opacity:1;transform:translateY(0)}}

/* ── KPI Bar ─────────────────────────────────────── */
.kpi-bar{{position:sticky;top:0;z-index:100;background:rgba(8,12,20,0.92);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border-bottom:1px solid var(--border);padding:14px 0;transition:all 0.3s var(--ease-out-expo)}}
.kpi-bar-inner{{max-width:1400px;margin:0 auto;padding:0 40px;display:flex;gap:0;justify-content:space-between;align-items:center}}
.kpi-item{{text-align:center;flex:1}}
.kpi-item-value{{font-family:var(--font-display);font-size:1.6rem;font-weight:400;color:var(--text-primary);line-height:1.2;letter-spacing:-0.01em}}
.kpi-item-value.gold{{color:var(--accent-gold)}}
.kpi-item-value.green{{color:var(--accent-green)}}
.kpi-item-label{{font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-top:2px}}
.kpi-divider{{width:1px;height:30px;background:var(--border)}}

/* ── Content Sections ────────────────────────────── */
.container{{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:0 40px}}
.section{{margin-bottom:100px;padding-top:20px}}
.section-header{{margin-bottom:40px}}
.section-label{{font-family:var(--font-mono);font-size:0.7rem;letter-spacing:0.25em;text-transform:uppercase;color:var(--accent-gold);margin-bottom:8px}}
.section-title{{font-family:var(--font-display);font-size:2rem;font-weight:400;letter-spacing:-0.01em;color:var(--text-primary)}}
.section-desc{{font-size:0.95rem;color:var(--text-secondary);max-width:600px;margin-top:8px}}

/* ── Chart Containers ────────────────────────────── */
.chart-panel{{background:var(--bg-panel);border:1px solid var(--border);border-radius:16px;padding:32px;margin-bottom:24px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);position:relative;overflow:hidden;transition:border-color 0.4s var(--ease-out-expo),box-shadow 0.4s}}
.chart-panel:hover{{border-color:rgba(212,165,98,0.15);box-shadow:0 0 60px rgba(212,165,98,0.04)}}
.chart-panel::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--accent-gold-dim),transparent);opacity:0.5}}
.chart-panel-title{{font-family:var(--font-display);font-size:1.15rem;color:var(--text-primary);margin-bottom:20px;letter-spacing:-0.01em}}
.chart-panel-desc{{font-size:0.8rem;color:var(--text-muted);margin-bottom:24px}}
.chart-svg{{width:100%;display:block}}
.chart-svg text{{font-family:var(--font-body);fill:var(--text-secondary)}}
.chart-svg .axis line,.chart-svg .axis path{{stroke:var(--border)}}
.chart-svg .axis .tick line{{stroke:rgba(255,255,255,0.04)}}

/* ── Stats Grid ──────────────────────────────────── */
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:24px}}
.stat-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:20px;position:relative;overflow:hidden;transition:transform 0.35s var(--ease-out-expo),border-color 0.35s,box-shadow 0.35s;cursor:default}}
.stat-card::before{{content:'';position:absolute;inset:0;border-radius:12px;background:radial-gradient(600px circle at var(--mx,50%) var(--my,50%),rgba(212,165,98,0.06),transparent 50%);opacity:0;transition:opacity 0.3s}}
.stat-card:hover{{transform:translateY(-4px);border-color:rgba(212,165,98,0.25);box-shadow:0 12px 40px rgba(0,0,0,0.3),0 0 0 1px rgba(212,165,98,0.1) inset}}
.stat-card:hover::before{{opacity:1}}
.stat-card-value{{font-family:var(--font-display);font-size:2rem;font-weight:400;letter-spacing:-0.02em;line-height:1.1;transition:transform 0.3s var(--ease-out-expo)}}
.stat-card:hover .stat-card-value{{transform:scale(1.05)}}
.stat-card-label{{font-size:0.75rem;color:var(--text-muted);margin-top:4px;text-transform:uppercase;letter-spacing:0.08em}}
.stat-card-note{{font-size:0.7rem;color:var(--text-muted);margin-top:6px;opacity:0.7;transition:opacity 0.3s}}
.stat-card:hover .stat-card-note{{opacity:1}}

/* ── Two Column Layout ───────────────────────────── */
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
.grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}}

/* ── Trade Table ─────────────────────────────────── */
.trade-table{{width:100%;border-collapse:collapse;font-size:0.85rem}}
.trade-table th{{text-align:left;padding:12px 16px;color:var(--text-muted);font-weight:500;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--bg-panel);z-index:2}}
.trade-table td{{padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.03);font-family:var(--font-mono);font-size:0.8rem;transition:background 0.2s}}
.trade-table tbody tr:nth-child(even) td{{background:rgba(255,255,255,0.01)}}
.trade-table tr.win td{{color:var(--accent-green)}}
.trade-table tr.loss td{{color:var(--accent-red)}}
.trade-table tbody tr{{transition:transform 0.2s var(--ease-out-expo)}}
.trade-table tbody tr:hover td{{background:rgba(212,165,98,0.06)!important}}
.trade-table tbody tr:hover{{transform:translateX(4px)}}
.trade-scroll{{max-height:500px;overflow-y:auto;border-radius:0 0 16px 16px}}

/* ── Config Chips ────────────────────────────────── */
.config-row{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px}}
.config-chip{{display:inline-flex;align-items:center;gap:8px;padding:8px 16px;background:var(--bg-card);border:1px solid var(--border);border-radius:100px;font-size:0.8rem;font-family:var(--font-mono);transition:all 0.3s var(--ease-out-expo);cursor:default}}
.config-chip:hover{{border-color:var(--accent-gold);background:rgba(212,165,98,0.08);transform:translateY(-1px)}}
.config-chip .chip-dot{{width:6px;height:6px;border-radius:50%;background:var(--accent-gold);transition:transform 0.3s}}
.config-chip:hover .chip-dot{{transform:scale(1.5)}}
.config-chip .chip-label{{color:var(--text-muted);font-size:0.7rem}}
.config-chip .chip-value{{color:var(--text-primary)}}

/* ── Footer ──────────────────────────────────────── */
.footer{{text-align:center;padding:60px 40px;border-top:1px solid var(--border);margin-top:60px}}
.footer-text{{font-size:0.8rem;color:var(--text-muted)}}
.disclaimer{{background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.15);border-radius:12px;padding:20px 24px;font-size:0.8rem;color:#b49450;line-height:1.6;margin-top:32px}}

/* ── Filter Controls ──────────────────────────────── */
.filter-bar{{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:16px}}
.filter-label{{font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em}}
.filter-select,.filter-input{{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);padding:8px 12px;font-size:0.8rem;font-family:var(--font-mono);outline:none;transition:border-color 0.2s}}
.filter-select:focus,.filter-input:focus{{border-color:var(--accent-gold)}}
.filter-select option{{background:var(--bg-deep);color:var(--text-primary)}}
.filter-btn{{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;color:var(--text-secondary);padding:8px 16px;font-size:0.8rem;cursor:pointer;font-family:var(--font-body);transition:all 0.2s var(--ease-out-expo)}}
.filter-btn:hover{{border-color:var(--accent-gold);color:var(--accent-gold)}}
.filter-btn.active{{background:var(--accent-gold);color:var(--bg-deep);border-color:var(--accent-gold)}}
.export-btn{{margin-left:auto;display:flex;align-items:center;gap:6px}}

/* ── Heatmap hover (no bounce) ───────────────────── */
#chartHeatmap rect{{transition:stroke 0.15s,stroke-width 0.15s,opacity 0.15s}}
#chartHeatmap rect:hover{{stroke:var(--accent-gold);stroke-width:1.5px;opacity:0.9}}

/* ── Tooltip ─────────────────────────────────────── */
.d3-tooltip{{position:absolute;background:rgba(20,26,40,0.96);border:1px solid rgba(255,255,255,0.12);border-radius:8px;padding:12px 16px;pointer-events:none;font-size:0.8rem;opacity:0;transition:opacity 0.15s;z-index:200;backdrop-filter:blur(8px);box-shadow:0 8px 32px rgba(0,0,0,0.4)}}
.d3-tooltip .tt-date{{color:var(--text-muted);font-size:0.7rem;margin-bottom:4px}}
.d3-tooltip .tt-val{{color:var(--text-primary);font-family:var(--font-display);font-size:1.1rem}}
.d3-tooltip .tt-sub{{color:var(--text-secondary);font-size:0.75rem;margin-top:2px}}

/* ── Responsive ──────────────────────────────────── */
@media(max-width:900px){{
  .grid-2,.grid-3{{grid-template-columns:1fr}}
  .hero-title{{font-size:2rem}}
  .kpi-bar-inner{{flex-wrap:wrap;gap:12px;padding:0 20px}}
  .kpi-item-value{{font-size:1.2rem}}
  .kpi-divider:nth-child(8),.kpi-divider:nth-child(12){{display:none}}
}}

/* ── Reduced Motion ──────────────────────────────── */
@media(prefers-reduced-motion:reduce){{
  .reveal.staggered{{opacity:1;transform:none;transition:none}}
  .hero-eyebrow,.hero-title,.hero-subtitle,.hero-meta{{opacity:1;transform:none;animation:none}}
}}

/* ── Presentation Mode Toggle ────────────────────── */
.pres-toggle{{position:fixed;top:20px;right:20px;z-index:200;background:var(--bg-card);border:1px solid var(--border);border-radius:100px;padding:8px 16px;color:var(--text-secondary);font-size:0.75rem;cursor:pointer;font-family:var(--font-mono);letter-spacing:0.05em;transition:all 0.3s var(--ease-out-expo);backdrop-filter:blur(10px)}}
.pres-toggle:hover{{border-color:var(--accent-gold);color:var(--accent-gold)}}
.pres-toggle.on{{background:var(--accent-gold);color:var(--bg-deep);border-color:var(--accent-gold)}}
body.pres-mode .section{{margin-bottom:100vh}}
body.pres-mode .kpi-bar{{background:rgba(8,12,20,0.98)}}

/* ── Section Nav Dots ────────────────────────────── */
.section-nav{{position:fixed;right:30px;top:50%;transform:translateY(-50%);z-index:150;display:flex;flex-direction:column;gap:14px}}
.section-nav-dot{{width:8px;height:8px;border-radius:50%;background:var(--text-muted);transition:all 0.3s var(--ease-out-expo);cursor:pointer;border:none;padding:0}}
.section-nav-dot.active,.section-nav-dot:hover{{background:var(--accent-gold);transform:scale(1.5)}}
@media(max-width:900px){{.section-nav{{display:none}}}}
</style>
</head>
<body>

<!-- Background Effects -->
<canvas class="bg-canvas" id="bgCanvas"></canvas>
<div class="bg-glow bg-glow-1"></div>
<div class="bg-glow bg-glow-2"></div>
<div class="bg-glow bg-glow-3"></div>

<!-- Presentation Toggle -->
<button class="pres-toggle" id="presToggle" title="切换演示模式 (适合录屏)">REC</button>

<!-- Section Nav Dots -->
<nav class="section-nav" id="sectionNav">
  <button class="section-nav-dot" data-target="sec-overview" title="绩效概览"></button>
  <button class="section-nav-dot" data-target="sec-strategy" title="策略分解"></button>
  <button class="section-nav-dot" data-target="sec-trades" title="交易分析"></button>
  <button class="section-nav-dot" data-target="sec-rolling" title="滚动性能"></button>
  <button class="section-nav-dot" data-target="sec-tradelog" title="交易明细"></button>
</nav>

<!-- Hero -->
<section class="hero">
  <div class="hero-eyebrow">量化择时策略 · 21年回测验证</div>
  <h1 class="hero-title">量价斜率<br><em>择时系统</em></h1>
  <p class="hero-subtitle">
    基于最高价与最低价相对强度关系的市场择时模型。
    右偏标准分 + 价格过滤，沪深300指数21年累计收益 <em style="color:var(--accent-gold);font-style:normal">1674%</em>。
  </p>
  <div class="hero-meta">
    <span><span class="dot"></span>数据来源：baostock</span>
    <span><span class="dot"></span>回测区间：2005.01 — 2026.05</span>
    <span><span class="dot"></span>标的：沪深300</span>
  </div>
  <div style="margin-top:50px;animation:heroFadeIn 0.8s cubic-bezier(0.16,1,0.3,1) 1s both">
    <svg width="32" height="48" viewBox="0 0 32 48" style="opacity:0.4">
      <rect x="14" y="6" width="4" height="28" rx="2" fill="var(--text-secondary)">
        <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2s" repeatCount="indefinite"/>
      </rect>
      <rect x="14" y="36" width="4" height="6" rx="2" fill="var(--accent-gold)" opacity="0.6"/>
    </svg>
  </div>
</section>

<!-- Sticky KPI Bar -->
<div class="kpi-bar" id="kpiBar">
  <div class="kpi-bar-inner" id="kpiInner"></div>
</div>

<div class="container">

  <!-- Section 1: Performance Overview -->
  <section class="section reveal staggered" id="sec-overview">
    <div class="section-header">
      <div class="section-label">01 · 绩效概览</div>
      <div class="section-title">策略 vs 基准</div>
      <div class="section-desc">21年回测，47次交易，累计收益远超买入持有。策略最大回撤仅为基准的一半。</div>
    </div>

    <!-- Strategy Config -->
    <div class="config-row" id="configChips"></div>

    <!-- KPI Stats Grid -->
    <div class="stats-grid" id="statsGrid"></div>

    <!-- Main Equity Curve -->
    <div class="chart-panel">
      <div class="chart-panel-title">净值曲线（对数坐标）· 策略 vs 基准</div>
      <div class="chart-panel-desc">金色区域 = 策略净值 | 灰色虚线 = 买入持有基准 | ▲ 绿三角 = 买入信号 | ▼ 红三角 = 卖出信号</div>
      <div id="chartEquity" style="position:relative"></div>
    </div>
  </section>

  <!-- Section 2: Strategy Decomposition -->
  <section class="section reveal staggered" id="sec-strategy">
    <div class="section-header">
      <div class="section-label">02 · 策略分解</div>
      <div class="section-title">信号指标与市场状态</div>
      <div class="section-desc">滚动线性回归计算的斜率、标准分与择时指标。阈值线之上的区域为做多信号区间。</div>
    </div>

    <div class="grid-2">
      <div class="chart-panel">
        <div class="chart-panel-title">择时指标</div>
        <div id="chartIndicator" style="position:relative"></div>
      </div>
      <div class="chart-panel">
        <div class="chart-panel-title">回撤分析</div>
        <div id="chartDrawdown" style="position:relative"></div>
      </div>
    </div>

    <div class="chart-panel" style="margin-top:24px">
      <div class="chart-panel-title">月度收益热力图</div>
      <div class="chart-panel-desc">每个方块代表一个月的策略收益。绿色 = 盈利，红色 = 亏损，颜色越深幅度越大。</div>
      <div class="filter-bar" id="hmFilterBar">
        <span class="filter-label">年份范围</span>
        <select class="filter-select" id="hmYearStart"></select>
        <span style="color:var(--text-muted);font-size:0.8rem">—</span>
        <select class="filter-select" id="hmYearEnd"></select>
        <button class="filter-btn active" id="hmBtnAll">全部</button>
        <button class="filter-btn" id="hmBtnBull">牛市</button>
        <button class="filter-btn" id="hmBtnBear">熊市</button>
      </div>
      <div id="chartHeatmap" style="position:relative;overflow-x:auto"></div>
    </div>
  </section>

  <!-- Section 3: Trade Analysis -->
  <section class="section reveal staggered" id="sec-trades">
    <div class="section-header">
      <div class="section-label">03 · 交易分析</div>
      <div class="section-title">每笔交易的解剖</div>
      <div class="section-desc">47笔交易中，30笔盈利，17笔亏损。平均持仓46天，盈亏比3.00。</div>
    </div>

    <div class="grid-2">
      <div class="chart-panel">
        <div class="chart-panel-title">交易收益分布</div>
        <div id="chartTradeDist" style="position:relative"></div>
      </div>
      <div class="chart-panel">
        <div class="chart-panel-title">累计交易收益</div>
        <div id="chartTradeCum" style="position:relative"></div>
      </div>
    </div>

    <div class="grid-2" style="margin-top:24px">
      <div class="chart-panel">
        <div class="chart-panel-title">持仓天数分布</div>
        <div id="chartDaysDist" style="position:relative"></div>
      </div>
      <div class="chart-panel">
        <div class="chart-panel-title">收益 vs 持仓天数</div>
        <div id="chartReturnDays" style="position:relative"></div>
      </div>
    </div>
  </section>

  <!-- Section 4: Rolling Performance -->
  <section class="section reveal staggered" id="sec-rolling">
    <div class="section-header">
      <div class="section-label">04 · 滚动性能</div>
      <div class="section-title">策略在不同市场环境下的表现</div>
      <div class="section-desc">252日滚动窗口，观察策略年化收益和最大回撤的时变特征。</div>
    </div>

    <div class="grid-2">
      <div class="chart-panel">
        <div class="chart-panel-title">滚动年化收益率（252日）</div>
        <div id="chartRollingReturn" style="position:relative"></div>
      </div>
      <div class="chart-panel">
        <div class="chart-panel-title">滚动最大回撤（252日）</div>
        <div id="chartRollingDD" style="position:relative"></div>
      </div>
    </div>

    <div class="chart-panel" style="margin-top:24px">
      <div class="chart-panel-title">各年度收益</div>
      <div id="chartAnnualBar" style="position:relative"></div>
    </div>
  </section>

  <!-- Section 5: Trade Log -->
  <section class="section reveal staggered" id="sec-tradelog">
    <div class="section-header">
      <div class="section-label">05 · 交易明细</div>
      <div class="section-title">全部交易记录</div>
    </div>

    <div class="chart-panel" style="padding:24px 0 0 0">
      <div style="display:flex;align-items:center;padding:0 24px 16px">
        <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em">筛选</span>
        <button class="filter-btn active" id="tradeFilterAll" style="margin-left:12px">全部</button>
        <button class="filter-btn" id="tradeFilterWin" style="margin-left:8px">盈利 ({{DATA.tradeStats.winCount}})</button>
        <button class="filter-btn" id="tradeFilterLoss" style="margin-left:8px">亏损 ({{DATA.tradeStats.lossCount}})</button>
        <button class="filter-btn export-btn" id="tradeExport" style="margin-left:auto">⬇ 导出 CSV</button>
      </div>
      <div class="trade-scroll">
        <table class="trade-table">
          <thead><tr><th>#</th><th>开始日期</th><th>结束日期</th><th>持仓天数</th><th>收益率</th></tr></thead>
          <tbody id="tradeTableBody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Footer -->
  <div class="footer reveal">
    <p class="footer-text">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp; 数据源：baostock &nbsp;·&nbsp; 回测标的：沪深300指数</p>
    <div class="disclaimer">
      ⚠️ 以上内容由量化策略模型基于历史数据模拟生成，仅供学习研究参考，不构成任何投资建议或个股推荐。回测收益不代表未来表现。金融市场存在不可预测的风险，投资决策请基于个人风险承受能力和独立判断。
    </div>
  </div>

</div>

<script>
// ── Embedded Data ──────────────────────────────────
const DATA = {embedded_json};

// ── Chart Width Helper ─────────────────────────────
const getW = (el, minW=400) => {{
  const r = el.getBoundingClientRect();
  return Math.max(r.width || el.clientWidth || minW, minW);
}};

// ── Utils ──────────────────────────────────────────
const $$ = (s,p) => (p||document).querySelector(s);
const $$$ = (s,p) => Array.from((p||document).querySelectorAll(s));
const fmt = (n,d) => (n??0).toFixed(d??1);
const fmtPct = (n) => (n>=0?'+':'')+fmt(n,1)+'%';
const fmtNum = (n) => n>=1000 ? (n/1000).toFixed(0)+','+(n%1000).toString().padStart(3,'0') : String(n);
const parseDate = d3.timeParse("%Y-%m-%d");

// ── Background Canvas (cursor-interactive) ────────
(function(){{
  const c=document.getElementById('bgCanvas');
  const ctx=c.getContext('2d');
  let W,H,mx=-999,my=-999,particles=[];
  function resize(){{W=c.width=window.innerWidth;H=c.height=window.innerHeight}}
  resize();window.addEventListener('resize',resize);
  for(let i=0;i<40;i++)particles.push({{
    x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.2+0.3,
    vx:(Math.random()-0.5)*0.3,vy:(Math.random()-0.5)*0.3,
    a:Math.random()*0.15+0.03,ox:0,oy:0
  }});
  document.addEventListener('mousemove',e=>{{mx=e.clientX;my=e.clientY}});
  document.addEventListener('mouseleave',()=>{{mx=-999;my=-999}});
  function draw(){{
    ctx.clearRect(0,0,W,H);
    particles.forEach(p=>{{
      // cursor magnetic — gentle repel
      const dx=p.x-mx,dy=p.y-my;
      const dist=Math.sqrt(dx*dx+dy*dy);
      if(dist<200&&mx>0){{
        const force=(1-dist/200)*0.08;
        p.vx+=dx/dist*force;p.vy+=dy/dist*force;
      }}
      // dampen
      p.vx*=0.995;p.vy*=0.995;
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0)p.x=W;if(p.x>W)p.x=0;if(p.y<0)p.y=H;if(p.y>H)p.y=0;
      ctx.beginPath();ctx.arc(p.x,p.y,dist<200?p.r*1.8:p.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(212,165,98,${{dist<200?Math.min(p.a*3,0.6):p.a}})`;ctx.fill();
    }});
    // connect near particles
    for(let i=0;i<particles.length;i++){{
      for(let j=i+1;j<particles.length;j++){{
        const dx2=particles[i].x-particles[j].x,dy2=particles[i].y-particles[j].y;
        const d2=Math.sqrt(dx2*dx2+dy2*dy2);
        if(d2<150){{
          ctx.beginPath();ctx.moveTo(particles[i].x,particles[i].y);
          ctx.lineTo(particles[j].x,particles[j].y);
          ctx.strokeStyle=`rgba(212,165,98,${{0.02*(1-d2/150)}})`;ctx.stroke();
        }}
      }}
    }}
    requestAnimationFrame(draw);
  }}
  draw();
}})();

// ── Scroll Reveal ──────────────────────────────────
const observer=new IntersectionObserver((entries)=>{{
  entries.forEach(e=>{{if(e.isIntersecting)e.target.classList.add('visible');}})
}},{{threshold:0.15}});
$$$('.reveal.staggered').forEach(el=>observer.observe(el));

// ── KPI Bar ───────────────────────────────────────
(function buildKPI(){{
  const container=document.getElementById('kpiInner');
  const items=[
    {{v:fmt(DATA.perf.totalReturn,0)+'%',l:'累计收益',c:'gold'}},
    {{v:fmtPct(DATA.perf.annualReturn),l:'年化收益',c:'gold'}},
    {{v:fmt(DATA.perf.sharpe,2),l:'夏普比率',c:''}},
    {{v:fmt(DATA.perf.maxDD,1)+'%',l:'最大回撤',c:'red'}},
    {{v:DATA.tradeStats.count,l:'交易次数',c:''}},
    {{v:fmt(DATA.tradeStats.winRate,1)+'%',l:'胜率',c:'green'}},
    {{v:fmt(DATA.tradeStats.plRatio,2),l:'盈亏比',c:'green'}},
    {{v:fmtPct(DATA.excessReturn),l:'超额收益',c:'gold'}},
  ];
  container.innerHTML=items.map((it,i)=>`
    ${{i>0?'<div class="kpi-divider"></div>':''}}
    <div class="kpi-item">
      <div class="kpi-item-value ${{it.c}}${{it.c==='red'?' style="color:var(--accent-red)"':''}}">${{it.v}}</div>
      <div class="kpi-item-label">${{it.l}}</div>
    </div>
  `).join('');
}})();

// ── Config Chips ──────────────────────────────────
(function(){{
  document.getElementById('configChips').innerHTML=DATA.config.map(c=>`
    <div class="config-chip">
      <span class="chip-dot"></span>
      <span class="chip-label">${{c.label}}</span>
      <span class="chip-value">${{c.value}}</span>
    </div>
  `).join('');
}})();

// ── Stats Grid ────────────────────────────────────
(function(){{
  const stats=[
    {{v:fmt(DATA.perf.totalReturn,0)+'%',l:'策略累计收益',n:'基准 '+fmt(DATA.bench.totalReturn,0)+'%',c:'var(--accent-gold)'}},
    {{v:fmtPct(DATA.perf.annualReturn),l:'策略年化收益',n:'基准 '+fmtPct(DATA.bench.annualReturn),c:'var(--accent-gold)'}},
    {{v:fmt(DATA.perf.sharpe,2),l:'夏普比率',n:'越高越好',c:'var(--text-primary)'}},
    {{v:fmt(DATA.perf.maxDD,1)+'%',l:'最大回撤',n:'基准 '+fmt(DATA.bench.maxDD,1)+'%',c:'var(--accent-red)'}},
    {{v:DATA.tradeStats.count,l:'交易次数',n:'胜率 '+fmt(DATA.tradeStats.winRate,1)+'%',c:'var(--text-primary)'}},
    {{v:fmt(DATA.tradeStats.plRatio,2),l:'盈亏比',n:'平均盈利 '+fmt(DATA.tradeStats.avgWin,1)+'%',c:'var(--accent-green)'}},
    {{v:DATA.tradeStats.avgDays+'天',l:'平均持仓天数',n:'最多 '+String(Math.max(...DATA.trades.map(t=>t.days)))+'天',c:'var(--text-primary)'}},
    {{v:fmtPct(DATA.excessReturn),l:'超额收益',n:'vs 买入持有',c:'var(--accent-gold)'}},
  ];
  document.getElementById('statsGrid').innerHTML=stats.map(s=>`
    <div class="stat-card">
      <div class="stat-card-value" style="color:${{s.c}}">${{s.v}}</div>
      <div class="stat-card-label">${{s.l}}</div>
      <div class="stat-card-note">${{s.n}}</div>
    </div>
  `).join('');
}})();

// ── Chart 1: Equity Curve ─────────────────────────
(function(){{
  const container=document.getElementById('chartEquity');
  const data=DATA.equity;
  const margin={{top:20,right:40,bottom:40,left:55}};
  const W=getW(container,900),H=500;
  const width=W+margin.left+margin.right,height=H+margin.top+margin.bottom;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{width}} ${{height}}`).attr('class','chart-svg').style('min-height',height+'px');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const x=d3.scaleTime().range([0,W]);
  const y=d3.scaleLog().range([H,0]).domain([0.8,d3.max(data,d=>Math.max(d.s,d.b))*1.1]);

  // main area clip
  g.append('defs').append('clipPath').attr('id','eqClip').append('rect').attr('width',W).attr('height',H);

  x.domain(d3.extent(data,d=>parseDate(d.d)));

  // Benchmark shadow
  const benchLine=d3.line().x(d=>x(parseDate(d.d))).y(d=>y(d.b));
  g.append('path').datum(data).attr('d',benchLine).attr('fill','none')
    .attr('stroke','rgba(255,255,255,0.12)').attr('stroke-width',1.5).attr('stroke-dasharray','4,4');

  // Strategy area (gradient fill)
  const areaGrad=svg.append('defs').append('linearGradient').attr('id','eqAreaGrad');
  areaGrad.append('stop').attr('offset','0%').attr('stop-color','var(--accent-gold)').attr('stop-opacity',0.25);
  areaGrad.append('stop').attr('offset','100%').attr('stop-color','var(--accent-gold)').attr('stop-opacity',0.02);
  const strategyArea=d3.area().x(d=>x(parseDate(d.d))).y0(y(1)).y1(d=>y(d.s));
  g.append('path').datum(data).attr('d',strategyArea).attr('fill','url(#eqAreaGrad)').attr('clip-path','url(#eqClip)');

  // Strategy line
  const stratLine=d3.line().x(d=>x(parseDate(d.d))).y(d=>y(d.s));
  g.append('path').datum(data).attr('d',stratLine).attr('fill','none')
    .attr('stroke','var(--accent-gold)').attr('stroke-width',2.2);

  // Buy signals (pos 0→1)
  const buyPoints=[];
  for(let i=1;i<data.length;i++)if(data[i].pos===1&&data[i-1].pos===0)buyPoints.push(data[i]);
  g.selectAll('.buy-dot').data(buyPoints).enter().append('polygon')
    .attr('points','0,-7 6,4 -6,4')
    .attr('transform',d=>`translate(${{x(parseDate(d.d))}},${{y(d.s)}})`)
    .attr('fill','var(--accent-green)').attr('opacity',0.9);

  // Sell signals (pos 1→0)
  const sellPoints=[];
  for(let i=1;i<data.length;i++)if(data[i].pos===0&&data[i-1].pos===1)sellPoints.push(data[i]);
  g.selectAll('.sell-dot').data(sellPoints).enter().append('polygon')
    .attr('points','0,7 6,-4 -6,-4')
    .attr('transform',d=>`translate(${{x(parseDate(d.d))}},${{y(d.s)}})`)
    .attr('fill','var(--accent-red)').attr('opacity',0.9);

  // Axes
  const xAxis=d3.axisBottom(x).ticks(10).tickFormat(d3.timeFormat('%Y'));
  const yAxis=d3.axisLeft(y).ticks(6).tickFormat(d=>d>=1?d.toFixed(0)+'x':d.toFixed(2));
  g.append('g').attr('transform',`translate(0,${{H}})`).call(xAxis)
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','11px');
  g.append('g').call(yAxis)
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','11px');

  // Tooltip
  const tip=d3.select(container).append('div').attr('class','d3-tooltip');
  const focusLine=g.append('line').attr('stroke','rgba(255,255,255,0.15)').attr('stroke-width',1).attr('y1',0).attr('y2',H).style('display','none');
  const focusDot=g.append('circle').attr('r',4).attr('fill','var(--accent-gold)').style('display','none');

  const bisect=d3.bisector(d=>parseDate(d.d)).left;
  g.append('rect').attr('width',W).attr('height',H).attr('fill','none').attr('pointer-events','all')
    .on('mousemove',function(e){{
      const mx=d3.pointer(e)[0];
      const dx=x.invert(mx);
      const i=bisect(data,dx,1);
      if(i>=data.length)return;
      const d=data[i];
      focusLine.style('display',null).attr('x1',x(parseDate(d.d))).attr('x2',x(parseDate(d.d)));
      focusDot.style('display',null).attr('cx',x(parseDate(d.d))).attr('cy',y(d.s));
      tip.style('opacity',1).style('left',(x(parseDate(d.d))+margin.left+15)+'px').style('top',(y(d.s)+margin.top-40)+'px')
        .html(`<div class="tt-date">${{d.d}}</div><div class="tt-val">策略净值 ${{fmt(d.s,3)}}</div><div class="tt-sub">基准 ${{fmt(d.b,3)}} | 回撤 ${{fmt(d.dd,1)}}%</div>`);
    }})
    .on('mouseleave',function(){{focusLine.style('display','none');focusDot.style('display','none');tip.style('opacity',0);}});
}})();

// ── Chart 2: Drawdown ─────────────────────────────
(function(){{
  const container=document.getElementById('chartDrawdown');
  const data=DATA.equity.filter((_,i)=>i%3===0);
  const margin={{top:20,right:20,bottom:30,left:45}};
  const W=getW(container,350)-margin.left-margin.right,H=220;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{W+margin.left+margin.right}} ${{H+margin.top+margin.bottom}}`).attr('class','chart-svg').style('min-height',(H+60)+'px');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const x=d3.scaleTime().domain(d3.extent(data,d=>parseDate(d.d))).range([0,W]);
  const y=d3.scaleLinear().domain([d3.min(data,d=>d.dd)*1.1,0]).range([H,0]);

  const area=d3.area().x(d=>x(parseDate(d.d))).y0(y(0)).y1(d=>y(d.dd));
  const grad=svg.append('defs').append('linearGradient').attr('id','ddGrad');
  grad.append('stop').attr('offset','0%').attr('stop-color','var(--accent-red)').attr('stop-opacity',0.3);
  grad.append('stop').attr('offset','100%').attr('stop-color','var(--accent-red)').attr('stop-opacity',0.02);
  g.append('path').datum(data).attr('d',area).attr('fill','url(#ddGrad)');
  const line=d3.line().x(d=>x(parseDate(d.d))).y(d=>y(d.dd));
  g.append('path').datum(data).attr('d',line).attr('fill','none').attr('stroke','var(--accent-red)').attr('stroke-width',1.5);

  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%Y')))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d=>d+'%'))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
}})();

// ── Chart 3: Indicator (sampled) ──────────────────
(function(){{
  const container=document.getElementById('chartIndicator');
  const data=DATA.equity.filter((_,i)=>i%4===0);
  const margin={{top:20,right:20,bottom:30,left:45}};
  const W=getW(container,350)-margin.left-margin.right,H=220;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{W+margin.left+margin.right}} ${{H+margin.top+margin.bottom}}`).attr('class','chart-svg');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const x=d3.scaleTime().domain(d3.extent(data,d=>parseDate(d.d))).range([0,W]);
  const sigVals=data.filter(d=>!isNaN(d.sig));
  const yExtent=d3.extent(sigVals,d=>d.sig);
  const yMax=Math.max(Math.abs(yExtent[0]),Math.abs(yExtent[1]))*1.3;
  const y=d3.scaleLinear().domain([-yMax,yMax]).range([H,0]);

  // fill areas for pos=1
  const posRegions=[];
  let inPos=null;
  data.forEach(d=>{{
    if(d.pos===1&&!inPos)inPos={{start:parseDate(d.d)}};
    else if(d.pos===0&&inPos){{posRegions.push({{start:inPos.start,end:parseDate(d.d)}});inPos=null;}}
  }});
  if(inPos)posRegions.push({{start:inPos.start,end:parseDate(data[data.length-1].d)}});
  posRegions.forEach(r=>{{
    g.append('rect').attr('x',x(r.start)).attr('width',Math.max(x(r.end)-x(r.start),2)).attr('y',0).attr('height',H)
      .attr('fill','rgba(52,211,153,0.04)');
  }});

  const posLine=d3.line().x(d=>x(parseDate(d.d))).y(d=>y(d.sig));
  g.append('path').datum(sigVals).attr('d',posLine).attr('fill','none').attr('stroke','var(--accent-cyan)').attr('stroke-width',1.5);

  // thresholds
  const thr=0.7;
  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(thr)).attr('y2',y(thr)).attr('stroke','rgba(52,211,153,0.3)').attr('stroke-dasharray','4,3');
  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(-thr)).attr('y2',y(-thr)).attr('stroke','rgba(248,113,113,0.3)').attr('stroke-dasharray','4,3');
  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','rgba(255,255,255,0.08)').attr('stroke-width',0.5);

  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%Y')))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
  g.append('g').call(d3.axisLeft(y).ticks(4))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
}})();

// ── Chart 4: Heatmap ──────────────────────────────
(function(){{
  const container=document.getElementById('chartHeatmap');
  const hm=DATA.monthlyHeatmap;
  const data=hm.data;
  const years=hm.years;
  const cellSize=22,gap=3;
  const margin={{left:50,top:10}};
  const W=12*(cellSize+gap),H=data.length*(cellSize+gap);
  const maxAbs=d3.max(data.flat().filter(v=>v!=null),d=>Math.abs(d))||20;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{W+margin.left+20}} ${{H+margin.top+10}}`).attr('class','chart-svg').style('min-height',H+30+'px');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const colorScale=d3.scaleSequential(d3.interpolateRdYlGn).domain([maxAbs,-maxAbs]);

  data.forEach((row,ri)=>{{
    row.forEach((val,ci)=>{{
      if(val===null)return;
      g.append('rect').attr('x',ci*(cellSize+gap)).attr('y',ri*(cellSize+gap))
        .attr('width',cellSize).attr('height',cellSize).attr('rx',2)
        .attr('fill',colorScale(val))
        .append('title').text(`${{years[ri]}}年${{ci+1}}月: ${{fmtPct(val)}}`);
    }});
  }});

  // Year labels every 2 years
  data.forEach((_,ri)=>{{
    if(ri%2===0){{
      g.append('text').attr('x',-8).attr('y',ri*(cellSize+gap)+cellSize/2+4)
        .attr('text-anchor','end').attr('fill','var(--text-muted)').style('font-size','10px').text(years[ri]);
    }}
  }});
  // Month labels
  for(let ci=0;ci<12;ci++){{
    g.append('text').attr('x',ci*(cellSize+gap)+cellSize/2).attr('y',-4)
      .attr('text-anchor','middle').attr('fill','var(--text-muted)').style('font-size','8px').text(ci+1);
  }}
}})();

// ── Chart 5: Trade Distribution ────────────────────
(function(){{
  const container=document.getElementById('chartTradeDist');
  const trades=DATA.trades.map(t=>t.ret);
  const margin={{top:20,right:20,bottom:30,left:45}};
  const W=getW(container,350)-margin.left-margin.right,H=220;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{W+margin.left+margin.right}} ${{H+margin.top+margin.bottom}}`).attr('class','chart-svg');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const bins=d3.histogram().domain(d3.extent(trades)).thresholds(12)(trades);
  const x=d3.scaleLinear().domain(d3.extent(trades)).range([0,W]);
  const y=d3.scaleLinear().domain([0,d3.max(bins,d=>d.length)*1.2]).range([H,0]);

  g.selectAll('rect').data(bins).enter().append('rect')
    .attr('x',d=>x(d.x0)).attr('width',d=>Math.max(x(d.x1)-x(d.x0)-1,0))
    .attr('y',d=>y(d.length)).attr('height',d=>H-y(d.length))
    .attr('fill',d=>(d.x0+d.x1)/2>=0?'var(--accent-green)':'var(--accent-red)')
    .attr('rx',3).attr('opacity',0.8);

  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(6).tickFormat(d=>d+'%'))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
  g.append('g').call(d3.axisLeft(y).ticks(4))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
}})();

// ── Chart 6: Cumulative Trade Return ──────────────
(function(){{
  const container=document.getElementById('chartTradeCum');
  const trades=DATA.tradesCum;
  const margin={{top:20,right:20,bottom:30,left:55}};
  const W=getW(container,350)-margin.left-margin.right,H=220;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{W+margin.left+margin.right}} ${{H+margin.top+margin.bottom}}`).attr('class','chart-svg');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const x=d3.scaleLinear().domain([1,trades.length]).range([0,W]);
  const yMax=d3.max(trades,d=>d.cum)*1.15;
  const yMin=d3.min(trades,d=>d.cum);
  const y=d3.scaleLinear().domain([Math.min(yMin*1.2,-5),yMax]).range([H,0]);

  // Waterfall bars
  g.selectAll('rect').data(trades).enter().append('rect')
    .attr('x',d=>x(d.i)-Math.min((W/trades.length)*0.7,20)/2)
    .attr('width',Math.min((W/trades.length)*0.7,20))
    .attr('y',d=>y(Math.max(0,d.ret)))
    .attr('height',d=>Math.abs(y(d.ret)-y(0)))
    .attr('fill',d=>d.ret>=0?'var(--accent-green)':'var(--accent-red)')
    .attr('rx',2).attr('opacity',0.75);

  // Cumulative line
  const cumLine=d3.line().x(d=>x(d.i)).y(d=>y(d.cum)).curve(d3.curveMonotoneX);
  g.append('path').datum(trades).attr('d',cumLine).attr('fill','none')
    .attr('stroke','var(--accent-gold)').attr('stroke-width',2);

  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(0)).attr('y2',y(0))
    .attr('stroke','rgba(255,255,255,0.1)').attr('stroke-width',0.5);

  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(8).tickFormat(d=>'#'+d))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
  g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d=>d+'%'))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
}})();

// ── Chart 7: Holding Days Distribution ────────────
(function(){{
  const container=document.getElementById('chartDaysDist');
  const days=DATA.trades.map(t=>t.days);
  const margin={{top:20,right:20,bottom:30,left:45}};
  const W=getW(container,350)-margin.left-margin.right,H=220;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{W+margin.left+margin.right}} ${{H+margin.top+margin.bottom}}`).attr('class','chart-svg');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const bins=d3.histogram().domain([0,d3.max(days)*1.05]).thresholds(15)(days);
  const x=d3.scaleLinear().domain([0,d3.max(days)*1.05]).range([0,W]);
  const y=d3.scaleLinear().domain([0,d3.max(bins,d=>d.length)*1.2]).range([H,0]);

  g.selectAll('rect').data(bins).enter().append('rect')
    .attr('x',d=>x(d.x0)).attr('width',d=>Math.max(x(d.x1)-x(d.x0)-1,0))
    .attr('y',d=>y(d.length)).attr('height',d=>H-y(d.length))
    .attr('fill','var(--accent-blue)').attr('rx',3).attr('opacity',0.7);

  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(6).tickFormat(d=>d+'天'))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
  g.append('g').call(d3.axisLeft(y).ticks(4))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
}})();

// ── Chart 8: Return vs Days scatter ───────────────
(function(){{
  const container=document.getElementById('chartReturnDays');
  const trades=DATA.trades;
  const margin={{top:20,right:20,bottom:30,left:50}};
  const W=getW(container,350)-margin.left-margin.right,H=220;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{W+margin.left+margin.right}} ${{H+margin.top+margin.bottom}}`).attr('class','chart-svg');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const x=d3.scaleLinear().domain([0,d3.max(trades,d=>d.days)*1.1]).range([0,W]);
  const yExtent=d3.extent(trades,d=>d.ret);
  const yPad=Math.max(Math.abs(yExtent[0]),yExtent[1])*0.2;
  const y=d3.scaleLinear().domain([yExtent[0]-yPad,yExtent[1]+yPad]).range([H,0]);

  g.selectAll('circle').data(trades).enter().append('circle')
    .attr('cx',d=>x(d.days)).attr('cy',d=>y(d.ret))
    .attr('r',d=>Math.sqrt(Math.abs(d.ret))*2+4)
    .attr('fill',d=>d.ret>=0?'var(--accent-green)':'var(--accent-red)')
    .attr('opacity',0.75);

  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(0)).attr('y2',y(0))
    .attr('stroke','rgba(255,255,255,0.1)').attr('stroke-width',0.5);

  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(5).tickFormat(d=>d+'天'))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
  g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d=>d+'%'))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
}})();

// ── Chart 9: Rolling Return ───────────────────────
(function(){{
  const container=document.getElementById('chartRollingReturn');
  const data=DATA.rollingAnnual.filter((_,i)=>i%3===0);
  const margin={{top:20,right:20,bottom:30,left:50}};
  const W=getW(container,350)-margin.left-margin.right,H=220;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{W+margin.left+margin.right}} ${{H+margin.top+margin.bottom}}`).attr('class','chart-svg');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const x=d3.scaleTime().domain(d3.extent(data,d=>parseDate(d.d))).range([0,W]);
  const yMax=d3.max(data,d=>d.v)*1.2,yMin=d3.min(data,d=>d.v);
  const y=d3.scaleLinear().domain([Math.min(yMin*1.3,-10),Math.max(yMax,10)]).range([H,0]);

  const area=d3.area().x(d=>x(parseDate(d.d))).y0(y(0)).y1(d=>y(d.v));
  g.append('path').datum(data).attr('d',area).attr('fill','rgba(96,165,250,0.1)');
  const line=d3.line().x(d=>x(parseDate(d.d))).y(d=>y(d.v));
  g.append('path').datum(data).attr('d',line).attr('fill','none').attr('stroke','var(--accent-blue)').attr('stroke-width',1.5);

  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','rgba(255,255,255,0.1)');

  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%Y')))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d=>d+'%'))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
}})();

// ── Chart 10: Rolling Drawdown ────────────────────
(function(){{
  const container=document.getElementById('chartRollingDD');
  const data=DATA.rollingDD.filter((_,i)=>i%3===0);
  const margin={{top:20,right:20,bottom:30,left:50}};
  const W=getW(container,350)-margin.left-margin.right,H=220;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{W+margin.left+margin.right}} ${{H+margin.top+margin.bottom}}`).attr('class','chart-svg');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const x=d3.scaleTime().domain(d3.extent(data,d=>parseDate(d.d))).range([0,W]);
  const y=d3.scaleLinear().domain([d3.min(data,d=>d.v)*1.1,0]).range([H,0]);

  const area=d3.area().x(d=>x(parseDate(d.d))).y0(y(0)).y1(d=>y(d.v));
  g.append('path').datum(data).attr('d',area).attr('fill','rgba(248,113,113,0.12)');
  const line=d3.line().x(d=>x(parseDate(d.d))).y(d=>y(d.v));
  g.append('path').datum(data).attr('d',line).attr('fill','none').attr('stroke','var(--accent-red)').attr('stroke-width',1.5);

  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%Y')))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(d=>d+'%'))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
}})();

// ── Chart 11: Annual Bar ──────────────────────────
(function(){{
  const container=document.getElementById('chartAnnualBar');
  const data=DATA.annualReturns;
  const margin={{top:20,right:20,bottom:40,left:55}};
  const W=Math.max(getW(container,600),data.length*50)-margin.left-margin.right;
  const H=260;

  const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{W+margin.left+margin.right}} ${{H+margin.top+margin.bottom}}`).attr('class','chart-svg').style('min-height',H+80+'px');
  const g=svg.append('g').attr('transform',`translate(${{margin.left}},${{margin.top}})`);

  const x=d3.scaleBand().domain(data.map(d=>d.year)).range([0,W]).padding(0.3);
  const yMax=d3.max(data,d=>Math.abs(d.return))*1.2;
  const y=d3.scaleLinear().domain([-yMax,yMax]).range([H,0]);

  g.selectAll('rect').data(data).enter().append('rect')
    .attr('x',d=>x(d.year)).attr('width',x.bandwidth())
    .attr('y',d=>y(Math.max(0,d.return)))
    .attr('height',d=>Math.abs(y(d.return)-y(0)))
    .attr('fill',d=>d.return>=0?'var(--accent-green)':'var(--accent-red)')
    .attr('rx',3).attr('opacity',0.8);

  // Value labels
  g.selectAll('.val-label').data(data).enter().append('text')
    .attr('x',d=>x(d.year)+x.bandwidth()/2)
    .attr('y',d=>d.return>=0?y(d.return)-6:y(d.return)+14)
    .attr('text-anchor','middle').attr('fill','var(--text-secondary)')
    .style('font-size','10px').text(d=>fmtPct(d.return));

  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(0)).attr('y2',y(0))
    .attr('stroke','rgba(255,255,255,0.1)');

  g.append('g').attr('transform',`translate(0,${{H}})`).call(d3.axisBottom(x))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
  g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d=>d+'%'))
    .selectAll('text').attr('fill','var(--text-muted)').style('font-size','10px');
}})();

// ── Trade Table ────────────────────────────────────
(function(){{
  const tbody=document.getElementById('tradeTableBody');
  const allTrades=DATA.trades;
  function renderTradeTable(filter){{
    const filtered=filter?allTrades.filter(filter):allTrades;
    tbody.innerHTML=filtered.map((t,i)=>`
      <tr class="${{t.ret>=0?'win':'loss'}}">
        <td style="color:var(--text-muted)">#${{i+1}}</td>
        <td>${{t.start}}</td><td>${{t.end}}</td>
        <td>${{t.days}}天</td><td>${{fmtPct(t.ret)}}</td>
      </tr>
    `).join('');
  }}
  renderTradeTable();

  // Trade filter buttons
  $$$('.filter-btn').forEach(btn=>{{
    btn.addEventListener('click',function(){{
      const id=this.id;
      if(id==='tradeFilterAll'){{renderTradeTable();activateBtn(this,'#tradeFilterAll,#tradeFilterWin,#tradeFilterLoss')}}
      else if(id==='tradeFilterWin'){{renderTradeTable(t=>t.ret>0);activateBtn(this,'#tradeFilterAll,#tradeFilterWin,#tradeFilterLoss')}}
      else if(id==='tradeFilterLoss'){{renderTradeTable(t=>t.ret<=0);activateBtn(this,'#tradeFilterAll,#tradeFilterWin,#tradeFilterLoss')}}
    }});
  }});
  function activateBtn(active,groupSel){{
    $$$(groupSel).forEach(b=>b.classList.remove('active'));
    active.classList.add('active');
  }}

  // Trade export — CSV download
  document.getElementById('tradeExport').addEventListener('click',()=>{{
    const rows=tbody.querySelectorAll('tr');
    let csv='序号,开始日期,结束日期,持仓天数,收益率\\n';
    rows.forEach((r,i)=>{{
      const cells=r.querySelectorAll('td');
      csv+=`${{i+1}},${{cells[1].textContent}},${{cells[2].textContent}},${{cells[3].textContent}},${{cells[4].textContent}}\\n`;
    }});
    const blob=new Blob(['\\uFEFF'+csv],{{type:'text/csv;charset=utf-8'}});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=url;a.download='交易记录.csv';a.click();
    URL.revokeObjectURL(url);
  }});

  // Heatmap year filter
  (function(){{
    const years=DATA.monthlyHeatmap.years;
    const allYears=years.slice();
    const selStart=document.getElementById('hmYearStart');
    const selEnd=document.getElementById('hmYearEnd');
    if(!selStart||!selEnd)return;

    // Populate selects
    years.forEach(y=>{{
      selStart.innerHTML+=`<option value="${{y}}">${{y}}</option>`;
      selEnd.innerHTML+=`<option value="${{y}}">${{y}}</option>`;
    }});
    selEnd.value=years[years.length-1];

    function filterHeatmap(startY,endY){{
      const rects=$$$('#chartHeatmap rect');
      const textLabels=$$$('#chartHeatmap text');
      rects.forEach((r,i)=>{{
        const row=Math.floor(i/12);
        const yVal=years[row];
        if(yVal>=startY&&yVal<=endY){{r.style.opacity='';r.style.display=''}}
        else{{r.style.opacity='0.08';r.style.display=''}}
      }});
    }}

    function applyFilter(){{
      const sy=+selStart.value,ey=+selEnd.value;
      if(sy>ey)return;
      filterHeatmap(sy,ey);
      $$$('#hmBtnAll,#hmBtnBull,#hmBtnBear').forEach(b=>b.classList.remove('active'));
    }}
    selStart.addEventListener('change',applyFilter);
    selEnd.addEventListener('change',applyFilter);

    document.getElementById('hmBtnAll').addEventListener('click',function(){{
      selStart.value=years[0];selEnd.value=years[years.length-1];
      filterHeatmap(years[0],years[years.length-1]);
      activateBtn(this,'#hmBtnAll,#hmBtnBull,#hmBtnBear');
    }});
    document.getElementById('hmBtnBull').addEventListener('click',function(){{
      filterHeatmap(2006,2007);
      activateBtn(this,'#hmBtnAll,#hmBtnBull,#hmBtnBear');
    }});
    document.getElementById('hmBtnBear').addEventListener('click',function(){{
      filterHeatmap(2008,2008);
      activateBtn(this,'#hmBtnAll,#hmBtnBull,#hmBtnBear');
    }});
  }})();

  // Heatmap cell tooltip (smooth, no bounce)
  (function(){{
    const container=document.getElementById('chartHeatmap');
    if(!container)return;
    const cells=$$$('#chartHeatmap rect');
    const tip=d3.select('#chartHeatmap').append('div').attr('class','d3-tooltip').style('pointer-events','none');
    cells.forEach(cell=>{{
      const titleEl=cell.querySelector('title');
      const titleText=titleEl?titleEl.textContent:'';
      cell.addEventListener('mouseenter',function(e){{
        const r=container.getBoundingClientRect();
        tip.style('opacity',1).html(`<div class="tt-date">${{titleText}}</div>`)
          .style('left',(e.clientX-r.left+15)+'px').style('top',(e.clientY-r.top-40)+'px');
      }});
      cell.addEventListener('mousemove',function(e){{
        const r=container.getBoundingClientRect();
        tip.style('left',(e.clientX-r.left+15)+'px').style('top',(e.clientY-r.top-40)+'px');
      }});
      cell.addEventListener('mouseleave',()=>tip.style('opacity',0));
    }});
  }})();

// ── Handle Resize ──────────────────────────────────
window.addEventListener('resize',()=>{{
  // SVG charts auto-resize via viewBox, no action needed
}});

// ── Presentation Mode Toggle ──────────────────────
const presToggle=document.getElementById('presToggle');
presToggle.addEventListener('click',()=>{{
  document.body.classList.toggle('pres-mode');
  presToggle.classList.toggle('on');
  presToggle.textContent=document.body.classList.contains('pres-mode')?'LIVE':'REC';
}});

// ── Section Nav Dots ──────────────────────────────
const navDots=$$$('.section-nav-dot');
const sections=navDots.map(d=>document.getElementById(d.dataset.target)).filter(Boolean);

// Click to scroll
navDots.forEach(dot=>{{
  dot.addEventListener('click',()=>{{
    const sec=document.getElementById(dot.dataset.target);
    if(sec)sec.scrollIntoView({{behavior:'smooth',block:'start'}});
  }});
}});

// Update active dot on scroll
let scrollTicking=false;
window.addEventListener('scroll',()=>{{
  if(scrollTicking)return;
  scrollTicking=true;
  requestAnimationFrame(()=>{{
    const scrollPos=window.scrollY+window.innerHeight/4;
    let activeIdx=-1;
    sections.forEach((sec,i)=>{{
      if(sec.offsetTop<=scrollPos)activeIdx=i;
    }});
    navDots.forEach((d,i)=>d.classList.toggle('active',i===activeIdx));
    scrollTicking=false;
  }});
}});

// ── Stat Card Mouse Glow Tracking ──────────────────
$$$('.stat-card').forEach(card=>{{
  card.addEventListener('mousemove',e=>{{
    const r=card.getBoundingClientRect();
    const x=((e.clientX-r.left)/r.width*100).toFixed(1);
    const y=((e.clientY-r.top)/r.height*100).toFixed(1);
    card.style.setProperty('--mx',x+'%');
    card.style.setProperty('--my',y+'%');
  }});
  card.addEventListener('mouseleave',()=>{{
    card.style.removeProperty('--mx');
    card.style.removeProperty('--my');
  }});
}});

// ── D3 Chart Enhancements ──────────────────────────
// Helper: create a shared crosshair + tooltip for a chart
function addCrosshair(containerId,opts={{}}){{

  const container=document.getElementById(containerId);
  if(!container)return;
  const svgEl=container.querySelector('svg');
  if(!svgEl)return;
  const svg=d3.select(svgEl);
  const g=svg.select('g'); // first <g> after defs
  const W=+(svg.attr('viewBox')||'').split(' ')[2]||600;
  const H=+(svg.attr('viewBox')||'').split(' ')[3]||300;
  const marginLeft=opts.marginLeft||50;
  const chartW=W-marginLeft;

  // Crosshair line
  const cross=d3.select(container).append('div').style('position','absolute').style('top','0').style('left',marginLeft+'px')
    .style('width','1px').style('height',H+'px').style('background','rgba(255,255,255,0.15)')
    .style('pointer-events','none').style('opacity',0).style('transition','opacity 0.15s');

  const tip=d3.select(container).append('div').attr('class','d3-tooltip').style('pointer-events','none');

  container.addEventListener('mousemove',e=>{{
    const r=container.getBoundingClientRect();
    const mx=e.clientX-r.left-marginLeft;
    if(mx<0||mx>chartW){{cross.style('opacity',0);tip.style('opacity',0);return}}
    cross.style('left',(mx+marginLeft)+'px').style('opacity',1);
    if(opts.onMove)opts.onMove(mx,e.clientX-r.left,{{
      show:(html,l,t)=>{{tip.style('opacity',1).html(html).style('left',l+'px').style('top',t+'px')}},
      hide:()=>tip.style('opacity',0)
    }},container);
  }});
  container.addEventListener('mouseleave',()=>{{cross.style('opacity',0);tip.style('opacity',0)}});
}}

// ── chartIndicator: crosshair + value ─────────────
addCrosshair('chartIndicator',{{
  marginLeft:45,
  onMove(mx,rx,ui,container){{
    const data=DATA.equity.filter((_,i)=>i%4===0);
    const svgEl=container.querySelector('svg');
    const W=+(svgEl.getAttribute('viewBox')||'').split(' ')[2]-45;
    const xScale=d3.scaleTime().domain(d3.extent(data,d=>parseDate(d.d))).range([0,W]);
    const dx=xScale.invert(mx);
    const bisect=d3.bisector(d=>parseDate(d.d)).left;
    let i=bisect(data,dx,1);
    if(i>=data.length)i=data.length-1;
    const d=data[i];
    ui.show(`<div class="tt-date">${{d.d}}</div><div class="tt-val">指标 ${{fmt(d.sig,3)}}</div>`,rx+12,20);
  }}
}});

// ── chartTradeDist: bar hover ────────────────────
(function(){{
  const container=document.getElementById('chartTradeDist');
  if(!container)return;
  const bars=$$$('#chartTradeDist rect');
  const tip=d3.select('#chartTradeDist').append('div').attr('class','d3-tooltip').style('pointer-events','none');
  bars.forEach(bar=>{{
    bar.addEventListener('mouseenter',function(e){{
      const len=+(this.__data__?this.__data__.length:0);
      const x0=this.__data__?this.__data__.x0:0,x1=this.__data__?this.__data__.x1:0;
      d3.select(this).transition().duration(150).attr('opacity',1).attr('filter','brightness(1.3)');
      tip.style('opacity',1).html(`<div class="tt-val">${{len}}笔</div><div class="tt-sub">${{fmtPct(x0)}} ~ ${{fmtPct(x1)}}</div>`)
        .style('left',(e.offsetX+15)+'px').style('top',(e.offsetY-50)+'px');
    }});
    bar.addEventListener('mouseleave',function(){{
      d3.select(this).transition().duration(200).attr('opacity',0.8).attr('filter',null);
      tip.style('opacity',0);
    }});
  }});
}})();

// ── chartTradeCum: bar + line tooltip ────────────
(function(){{
  const container=document.getElementById('chartTradeCum');
  if(!container)return;
  const bars=$$$('#chartTradeCum rect');
  const tip=d3.select('#chartTradeCum').append('div').attr('class','d3-tooltip').style('pointer-events','none');
  bars.forEach(bar=>{{
    bar.addEventListener('mouseenter',function(e){{
      const d=this.__data__||{{}};
      d3.select(this).transition().duration(150).attr('opacity',1).attr('filter','brightness(1.3)');
      tip.style('opacity',1).html(`<div class="tt-date">交易 #${{d.i||''}}</div><div class="tt-val">${{fmtPct(d.ret||0)}}</div><div class="tt-sub">累计 ${{fmt(d.cum||0,1)}}%</div>`)
        .style('left',(e.offsetX+15)+'px').style('top',(e.offsetY-60)+'px');
    }});
    bar.addEventListener('mouseleave',function(){{
      d3.select(this).transition().duration(200).attr('opacity',0.75).attr('filter',null);
      tip.style('opacity',0);
    }});
  }});
}})();

// ── chartDaysDist: bar hover ─────────────────────
(function(){{
  const container=document.getElementById('chartDaysDist');
  if(!container)return;
  const bars=$$$('#chartDaysDist rect');
  const tip=d3.select('#chartDaysDist').append('div').attr('class','d3-tooltip').style('pointer-events','none');
  bars.forEach(bar=>{{
    bar.addEventListener('mouseenter',function(e){{
      const len=+(this.__data__?this.__data__.length:0);
      const x0=+(this.__data__?this.__data__.x0:0),x1=+(this.__data__?this.__data__.x1:0);
      d3.select(this).transition().duration(150).attr('opacity',1).attr('filter','brightness(1.2)');
      tip.style('opacity',1).html(`<div class="tt-val">${{len}}笔</div><div class="tt-sub">${{Math.round(x0)}}~${{Math.round(x1)}}天</div>`)
        .style('left',(e.offsetX+15)+'px').style('top',(e.offsetY-50)+'px');
    }});
    bar.addEventListener('mouseleave',function(){{
      d3.select(this).transition().duration(200).attr('opacity',0.7).attr('filter',null);
      tip.style('opacity',0);
    }});
  }});
}})();

// ── chartReturnDays: enhanced scatter hover ──────
(function(){{
  const circles=$$$('#chartReturnDays circle');
  if(!circles.length)return;
  const scatterTip=d3.select('#chartReturnDays').append('div').attr('class','d3-tooltip').style('pointer-events','none');
  const trades=DATA.trades;
  circles.forEach((c,i)=>{{
    const td=trades[i]||{{}};
    c.addEventListener('mouseenter',function(e){{
      d3.select(this).transition().duration(150).attr('r',+d3.select(this).attr('r')*1.8).attr('opacity',1).attr('stroke','var(--accent-gold)').attr('stroke-width',2);
      scatterTip.style('opacity',1).style('left',(e.offsetX+20)+'px').style('top',(e.offsetY-60)+'px')
        .html(`<div class="tt-date">${{td.start||''}} ~ ${{td.end||''}}</div><div class="tt-val">${{fmtPct(td.ret||0)}}</div><div class="tt-sub">持仓 ${{td.days||0}}天</div>`);
    }});
    c.addEventListener('mousemove',function(e){{
      scatterTip.style('left',(e.offsetX+20)+'px').style('top',(e.offsetY-60)+'px');
    }});
    c.addEventListener('mouseleave',function(){{
      d3.select(this).transition().duration(250).attr('r',+d3.select(this).attr('r')/1.8).attr('opacity',0.75).attr('stroke',null);
      scatterTip.style('opacity',0);
    }});
  }});
}})();

// ── chartRollingReturn: crosshair + value ────────
addCrosshair('chartRollingReturn',{{
  marginLeft:50,
  onMove(mx,rx,ui,container){{
    const data=DATA.rollingAnnual.filter((_,i)=>i%3===0);
    const svgEl=container.querySelector('svg');
    const W=+(svgEl.getAttribute('viewBox')||'').split(' ')[2]-50;
    const xScale=d3.scaleTime().domain(d3.extent(data,d=>parseDate(d.d))).range([0,W]);
    const dx=xScale.invert(mx);
    const bisect=d3.bisector(d=>parseDate(d.d)).left;
    let i=bisect(data,dx,1);if(i>=data.length)i=data.length-1;
    const d=data[i];
    ui.show(`<div class="tt-date">${{d.d}}</div><div class="tt-val">${{fmtPct(d.v)}} 年化</div>`,rx+12,20);
  }}
}});

// ── chartRollingDD: crosshair + value ────────────
addCrosshair('chartRollingDD',{{
  marginLeft:50,
  onMove(mx,rx,ui,container){{
    const data=DATA.rollingDD.filter((_,i)=>i%3===0);
    const svgEl=container.querySelector('svg');
    const W=+(svgEl.getAttribute('viewBox')||'').split(' ')[2]-50;
    const xScale=d3.scaleTime().domain(d3.extent(data,d=>parseDate(d.d))).range([0,W]);
    const dx=xScale.invert(mx);
    const bisect=d3.bisector(d=>parseDate(d.d)).left;
    let i=bisect(data,dx,1);if(i>=data.length)i=data.length-1;
    const d=data[i];
    ui.show(`<div class="tt-date">${{d.d}}</div><div class="tt-val">回撤 ${{fmt(d.v,1)}}%</div>`,rx+12,20);
  }}
}});

// ── chartAnnualBar: enhanced hover + tooltip ─────
(function(){{
  const container=document.getElementById('chartAnnualBar');
  if(!container)return;
  const bars=$$$('#chartAnnualBar rect');
  const labels=$$$('#chartAnnualBar .val-label');
  const tip=d3.select('#chartAnnualBar').append('div').attr('class','d3-tooltip').style('pointer-events','none');
  bars.forEach((bar,i)=>{{
    bar.addEventListener('mouseenter',function(e){{
      bars.forEach(b=>d3.select(b).transition().duration(250).attr('opacity',0.2));
      d3.select(this).transition().duration(200).attr('opacity',1).attr('filter','brightness(1.3)');
      if(labels[i])labels[i].style.opacity='1';
      const d=DATA.annualReturns[i];
      if(d){{
        const r=container.getBoundingClientRect();
        tip.style('opacity',1).html(`<div class="tt-date">${{d.year}}年</div><div class="tt-val">${{fmtPct(d.return)}}</div>`)
          .style('left',(e.clientX-r.left+15)+'px').style('top',(e.clientY-r.top-50)+'px');
      }}
    }});
    bar.addEventListener('mouseleave',function(){{
      bars.forEach(b=>d3.select(b).transition().duration(350).attr('opacity',0.8).attr('filter',null));
      if(labels[i])labels[i].style.opacity='';
      tip.style('opacity',0);
    }});
  }});
}})();

// ── chartDrawdown: also highlight bar at cursor ──
addCrosshair('chartDrawdown',{{
  marginLeft:45,
  onMove(mx,rx,ui,container){{
    const data=DATA.equity.filter((_,i)=>i%3===0);
    const svgEl=container.querySelector('svg');
    const W=+(svgEl.getAttribute('viewBox')||'').split(' ')[2]-45;
    const xScale=d3.scaleTime().domain(d3.extent(data,d=>parseDate(d.d))).range([0,W]);
    const dx=xScale.invert(mx);
    const bisect=d3.bisector(d=>parseDate(d.d)).left;
    let i=bisect(data,dx,1);if(i>=data.length)i=data.length-1;
    const d=data[i];
    const ddVal=d.dd!==undefined?d.dd:0;
    ui.show(`<div class="tt-date">${{d.d}}</div><div class="tt-val" style="color:var(--accent-red)">回撤 ${{fmt(ddVal,1)}}%</div>`,rx+12,20);
  }}
}});


// Navigation dot tooltips
(function(){{
  const navTitles={{'sec-overview':'绩效概览','sec-strategy':'策略分解','sec-trades':'交易分析','sec-rolling':'滚动性能','sec-tradelog':'交易明细'}};
  const navDots=$$$('.section-nav-dot');
  const navTip=d3.select('body').append('div').attr('class','d3-tooltip').style('pointer-events','none').style('position','fixed').style('z-index','300');
  navDots.forEach(dot=>{{
    dot.addEventListener('mouseenter',e=>{{
      const title=navTitles[dot.dataset.target]||'';
      navTip.style('opacity',1).html(`<div class="tt-val" style="font-size:0.8rem">${{title}}</div>`)
        .style('right','50px').style('top',(e.clientY-20)+'px').style('left','auto');
    }});
    dot.addEventListener('mouseleave',()=>navTip.style('opacity',0));
  }});
}})();
</script>
</body>
</html>"""

# ── 写入 ────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(OUTPUT_HTML) / 1024
print(f"✅ 仪表盘已生成: {OUTPUT_HTML}")
print(f"   文件大小: {size_kb:.0f} KB")
print(f"   包含: 11个D3互动图表 + 粒子背景 + 滚动叙事")
