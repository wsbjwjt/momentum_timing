#!/usr/bin/env python3
"""Build cinematic D3 dashboard — clean, brace-verified version"""
import json, pandas as pd, numpy as np, os
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
EQUITY_CSV = os.path.join(BASE, "output", "equity.csv")
SUMMARY_JSON = os.path.join(BASE, "output", "summary.json")
OUTPUT_HTML = os.path.join(BASE, "output", "dashboard_cinema.html")

df = pd.read_csv(EQUITY_CSV, parse_dates=["date"])
with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
    summary = json.load(f)

perf = summary["策略表现"]
bench = summary["基准表现"]
trades_info = summary["交易统计"]
config = summary["策略配置"]
trades_list = trades_info.get("交易明细", [])

# ---- Prep embed data ----
# Equity (every 3rd point for reasonable size)
eq_sample = []
for i in range(0, len(df), 3):
    row = df.iloc[i]
    eq_sample.append({
        "d": row["date"].strftime("%Y-%m-%d"),
        "s": round(float(row["equity_curve_net"]), 3),
        "b": round(float(row["benchmark_curve"]), 3),
        "dd": round(float(row["drawdown"]) * 100, 1),
        "pos": int(row["pos"]),
    })

# Annual returns
temp = df.copy()
temp["year"] = temp["date"].dt.year
temp["month"] = temp["date"].dt.month
monthly = temp.groupby(["year", "month"])["strategy_return"].apply(lambda x: (1 + x).prod() - 1).unstack()
annual_ret = []
for y in monthly.index:
    yr_data = monthly.loc[y].dropna()
    if len(yr_data) > 0:
        annual_ret.append({"year": int(y), "return": round(float((1 + yr_data).prod() - 1) * 100, 1)})

# Monthly heatmap
for m in range(1, 13):
    if m not in monthly.columns:
        monthly[m] = np.nan
monthly = monthly[sorted(monthly.columns)]
hm_data = [[round(float(v) * 100, 1) if not np.isnan(v) else None for v in row] for row in monthly.values]

# Trades
trades_embed = []
for t in trades_list:
    trades_embed.append({
        "start": t["开始日期"], "end": t["结束日期"],
        "days": t["持仓天数"], "ret": t["收益率(%)"],
    })

# Rolling metrics (252d)
window = 252
roll_ann = []
roll_dd_list = []
for i in range(window - 1, len(df)):
    seg = df["strategy_return"].iloc[i - window + 1 : i + 1].values
    total = (1 + seg).prod()
    ann = (total ** (1 / (window / 252)) - 1) * 100
    roll_ann.append({"d": df["date"].iloc[i].strftime("%Y-%m-%d"), "v": round(float(ann), 1)})
    eq = df["equity_curve_net"].iloc[i - window + 1 : i + 1].values
    peak = np.maximum.accumulate(eq)
    dd = np.min((eq - peak) / peak) * 100
    roll_dd_list.append({"d": df["date"].iloc[i].strftime("%Y-%m-%d"), "v": round(float(dd), 1)})

embed = {
    "eq": eq_sample,
    "trades": trades_embed,
    "annual": annual_ret,
    "hm": {"years": monthly.index.astype(int).tolist(), "data": hm_data},
    "rollAnn": roll_ann,
    "rollDD": roll_dd_list,
    "perf": perf, "bench": bench, "tradeStats": trades_info, "config": config,
    "excess": summary.get("超额收益(%)", 0),
    "dateRange": summary["数据日期"],
    "strategyCount": trades_info["交易次数"],
    "strategyLabel": config.get("策略方法", ""),
}

embed_json = json.dumps(embed, ensure_ascii=False)

# ---- HTML ----
html = '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
html += '<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width,initial-scale=1.0">\n'
html += '<title>量化择时策略 · 回测仪表盘</title>\n'
html += '<script src="https://d3js.org/d3.v7.min.js"></script>\n'
html += '<style>\n'
html += open(os.path.join(BASE, "momentum_timing", "dashboard.css")).read() if os.path.exists(os.path.join(BASE, "momentum_timing", "dashboard.css")) else ""
html += '</style>\n</head>\n<body>\n'
html += '<div id="app"></div>\n'
html += '<script>\n'
html += 'var D = ' + embed_json + ';\n'
html += open(os.path.join(BASE, "momentum_timing", "dashboard.js")).read() if os.path.exists(os.path.join(BASE, "momentum_timing", "dashboard.js")) else ""
html += '\n</script>\n</body>\n</html>'

os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Generated: {OUTPUT_HTML} ({os.path.getsize(OUTPUT_HTML)/1024:.0f} KB)")
