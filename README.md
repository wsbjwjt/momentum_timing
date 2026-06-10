# 动量择时策略 · Momentum Timing

基于最高价与最低价相对强度关系的量化择时模型。

---

## 1. 策略概述

通过滚动线性回归计算**高低价斜率(β)**，结合**标准分(Z-score)**生成择时信号，捕捉市场动量变化。

核心特点：
- 右偏标准分 + 价格过滤
- 沪深300指数21年回测验证
- 纯 Python 实现，依赖少

---

## 2. 参数配置

| 参数 | 符号 | 默认值 | 含义 |
|------|------|--------|------|
| 回归周期 | N | 16 | 滚动回归窗口 |
| 标准分周期 | M | 300 | Z-score 计算窗口 |
| 信号阈值 | S | 0.7 | 买卖阈值 |
| 信号方法 | method | `right_biased` | 信号合成方式 |

---

## 3. 价格过滤

仅当 `MA₂₀(t-1) > MA₂₀(t-3)` 时放行买入信号，确保趋势向上确认。

---

## 4. 信号阈值

- signal_t > S → **买入信号**（做多）
- signal_t < -S → **卖出信号**（空仓）

---

## 5. 成交量过滤（可选）

当 `corr(volume, signal)₁₀ > 0` 时放行买入，实现量价共振确认。

---

## 6. 信号生成步骤

**Step 1 — 滚动回归**：`high_t = α_t + β_t · low_t + ε_t`，得到斜率 β_t 和 R²_t

**Step 2 — 标准分**：`Z_t = (β_t - μ_M(β)) / σ_M(β)`

**Step 3 — 信号合成**：支持 slope / zscore / corrected / right_biased 四种方法

**Step 4 — 阈值判断**：signal_t 与 ±S 比较生成买卖信号

**Step 5 — 价格过滤**：均线趋势确认

**Step 6 — 成交量过滤**：量价共振确认（可选）

---

## 7. 安装与快速开始

```bash
# 安装依赖
pip install baostock pandas numpy scikit-learn plotly tqdm tenacity

# 一键回测
python run_backtest.py --mode quick

# 查看结果
open output/backtest_dashboard.html
```

### 运行模式

| 模式 | 命令 | 说明 |
|------|------|------|
| quick | `python run_backtest.py --mode quick` | 快速回测，1-2分钟 |
| standard | `python run_backtest.py --mode standard` | 标准回测，参数扫描 |
| compare | `python run_backtest.py --mode compare` | 多指数对比 |
| custom | `python run_backtest.py --mode custom --n 18 --m 600` | 自定义参数 |

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | right_biased | 信号方法 |
| `--n` | 16 | 回归周期 |
| `--m` | 300 | 标准分周期 |
| `--s` | 0.7 | 信号阈值 |
| `--index` | sh.000300 | 回测标的 |
| `--start` | 2005-01-01 | 开始日期 |

---

## 8. 输出文件与策略表现

### 输出文件

| 文件 | 说明 |
|------|------|
| `backtest_dashboard.html` | 交互式 HTML 仪表盘 |
| `backtest_equity.csv` | 每日净值序列 |
| `backtest_trades.csv` | 交易明细 |
| `backtest_summary.json` | 回测统计摘要 |

### 策略表现（沪深300，2005-2026）

| 指标 | 策略 | 基准 |
|------|------|------|
| 年化收益率 | ~24% | ~8% |
| 夏普比率 | ~1.3 | ~0.4 |
| 最大回撤 | ~-46% | ~-72% |
| 胜率 | ~62% | - |

---

## 9. 项目结构

```
momentum-timing/
├── run_backtest.py          # 主入口
├── momentum_timing/
│   ├── strategy.py          # 策略核心
│   ├── data.py              # 数据获取
│   ├── backtest.py          # 回测引擎
│   ├── dashboard.py         # HTML 生成
│   └── utils.py             # 工具函数
├── output/                  # 输出目录
└── requirements.txt         # 依赖
```

---

## 10. FAQ

**Q: `No module named 'baostock'`**

```bash
pip install baostock pandas numpy scikit-learn plotly tqdm tenacity
```

**Q: 数据下载慢？**

baostock 免费数据源限速，默认启用缓存，首次下载后复用。

**Q: 能否用于实盘？**

仅供学习研究，实盘需考虑滑点、冲击成本、信号滞后等。

**Q: 回测太慢？**

```bash
python run_backtest.py --mode quick --start 2020-01-01
```

**Q: 如何清除缓存？**

```bash
python -c "from momentum_timing.data import clear_cache; clear_cache()"
```

---

## 免责声明

以上内容由量化策略模型基于历史数据模拟生成，仅供学习研究参考，不构成任何投资建议。回测收益不代表未来表现。

## License

MIT
