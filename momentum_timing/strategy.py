"""
核心策略引擎 —— 基于量价斜率指标的择时系统

策略核心逻辑：
1. 用每日最高价与最低价的滚动线性回归，计算斜率（Beta）和拟合优度（R²）
2. 对斜率序列计算滚动标准分（Z-score），衡量当前支撑阻力相对强度
3. 多种指标变体：标准分、修正标准分、右偏标准分
4. 可选的价格趋势过滤（均线）和成交量相关性过滤
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from sklearn.linear_model import LinearRegression


class SlopeTimingStrategy:
    """
    基于量价斜率的择时策略引擎。

    核心思想：
    - 每日最高价 = 阻力位，最低价 = 支撑位
    - 对 N 日内的 (最低价, 最高价) 做线性回归，斜率越大说明支撑越强
    - 将斜率标准化为 Z-score，配合 R² 修正和价格过滤，生成买卖信号

    策略变体：
    1. slope: 直接用斜率值做阈值判断
    2. zscore: 用斜率的标准分（Z-score）
    3. corrected: 用 Z-score × R² （修正标准分）
    4. right_biased: 用 Z-score × R² × Beta （右偏标准分，通常效果最好）

    可选优化：
    - price_filter: 用 MA20 趋势过滤买入信号
    - volume_filter: 用成交量与指标相关性过滤信号
    """

    def __init__(
        self,
        n_regression: int = 16,
        m_standardize: int = 300,
        threshold: float = 0.7,
        method: str = "right_biased",
        use_price_filter: bool = True,
        use_volume_filter: bool = False,
        ma_period: int = 20,
        ma_lookback: int = 3,
        volume_corr_period: int = 10,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        buy_threshold: float = None,
        sell_threshold: float = None,
    ):
        """
        初始化策略参数。

        参数:
            n_regression: 计算线性回归的周期（推荐 16 或 18）
            m_standardize: 计算标准分的滚动周期（推荐 300）
            threshold: 开平仓阈值 — 非 slope 方法用对称阈值（推荐 0.7）
            method: 指标类型 - "slope", "zscore", "corrected", "right_biased"
            use_price_filter: 是否启用均线价格过滤
            use_volume_filter: 是否启用成交量相关性过滤
            ma_period: 均线周期
            ma_lookback: 均线比较的回看天数
            volume_corr_period: 成交量相关性计算周期
            high_col: 最高价列名
            low_col: 最低价列名
            close_col: 收盘价列名
            volume_col: 成交量列名
            buy_threshold: (slope专用) 买入阈值，默认 1.0
            sell_threshold: (slope专用) 卖出阈值，默认 0.8
        """
        self.n_regression = n_regression
        self.m_standardize = m_standardize
        self.threshold = threshold
        self.method = method
        self.use_price_filter = use_price_filter
        self.use_volume_filter = use_volume_filter
        self.ma_period = ma_period
        self.ma_lookback = ma_lookback
        self.volume_corr_period = volume_corr_period
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col

        # slope 方法使用非对称阈值
        self.buy_threshold = buy_threshold if buy_threshold is not None else (1.0 if method == "slope" else threshold)
        self.sell_threshold = sell_threshold if sell_threshold is not None else (0.8 if method == "slope" else -threshold)

        # 验证参数
        valid_methods = ["slope", "zscore", "corrected", "right_biased"]
        if method not in valid_methods:
            raise ValueError(f"method 必须是 {valid_methods} 之一，当前值: {method}")

    def compute(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        对输入的行情数据计算完整的策略信号。

        参数:
            df: 包含 high, low, close, volume 列的 DataFrame
            verbose: 是否输出详细进度

        返回:
            添加了 strategy 相关列的 DataFrame
        """
        df = df.copy()
        df = df.reset_index(drop=True)

        # Step 1: 滚动线性回归，计算 Beta 和 R²
        if verbose:
            print(f"  计算滚动回归 (N={self.n_regression})...")
        self._compute_rolling_regression(df)

        # Step 2: 计算斜率标准分
        if verbose:
            print(f"  计算标准分 (M={self.m_standardize})...")
        self._compute_zscore(df)

        # Step 3: 计算选定方法的指标值
        if verbose:
            print(f"  计算指标值 (method={self.method})...")
        self._compute_indicator(df)

        # Step 4: 计算均线（如需要）
        if self.use_price_filter:
            if verbose:
                print(f"  计算均线过滤 (MA{self.ma_period})...")
            df["ma"] = df[self.close_col].rolling(self.ma_period, min_periods=1).mean()

        # Step 5: 计算成交量相关性（如需要）
        if self.use_volume_filter:
            if verbose:
                print(f"  计算成交量相关性...")
            self._compute_volume_correlation(df)

        # Step 6: 生成买卖信号
        if verbose:
            print("  生成交易信号...")
        df = self._generate_signals(df)

        return df

    def _compute_rolling_regression(self, df: pd.DataFrame):
        """计算滚动窗口的线性回归 Beta 和 R²"""
        n = len(df)
        high_vals = df[self.high_col].values
        low_vals = df[self.low_col].values

        betas = np.full(n, np.nan)
        r2s = np.full(n, np.nan)

        for i in range(self.n_regression - 1, n):
            start = i - self.n_regression + 1
            end = i + 1
            x = low_vals[start:end].reshape(-1, 1)
            y = high_vals[start:end]

            # 跳过包含 NaN 的窗口
            if np.isnan(x).any() or np.isnan(y).any():
                continue

            model = LinearRegression().fit(x, y)
            betas[i] = model.coef_[0]
            r2s[i] = model.score(x, y)

        df["beta"] = betas
        df["r2"] = r2s

    def _compute_zscore(self, df: pd.DataFrame):
        """计算斜率的标准分（Z-score）"""
        roll = df["beta"].rolling(self.m_standardize, min_periods=1)
        df["beta_mean"] = roll.mean()
        df["beta_std"] = roll.std(ddof=0)
        df["beta_std"] = df["beta_std"].replace(0, np.nan)
        df["zscore"] = (df["beta"] - df["beta_mean"]) / df["beta_std"]

    def _compute_indicator(self, df: pd.DataFrame):
        """根据 method 类型计算最终指标值"""
        if self.method == "slope":
            df["indicator"] = df["beta"]
        elif self.method == "zscore":
            df["indicator"] = df["zscore"]
        elif self.method == "corrected":
            df["indicator"] = df["zscore"] * df["r2"]
        elif self.method == "right_biased":
            df["indicator"] = df["zscore"] * df["r2"] * df["beta"]
        else:
            raise ValueError(f"未知方法: {self.method}")

    def _compute_volume_correlation(self, df: pd.DataFrame):
        """计算成交量与指标值的滚动相关性"""
        window = self.volume_corr_period
        # 需要先有指标值
        if "indicator" not in df.columns:
            df["vol_corr"] = 0
            return

        corr_series = np.full(len(df), np.nan)
        vol = df[self.volume_col].values
        ind = df["indicator"].values

        for i in range(window - 1, len(df)):
            start = i - window + 1
            end = i + 1
            v = vol[start:end]
            iv = ind[start:end]
            mask = ~(np.isnan(v) | np.isnan(iv))
            if mask.sum() > 2:
                corr_series[i] = np.corrcoef(v[mask], iv[mask])[0, 1]

        df["vol_corr"] = corr_series

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号。

        - signal=1: 买入开仓
        - signal=0: 卖出平仓
        - pos: 持仓状态（信号滞后一期）
        """
        indicator = df["indicator"].values
        n = len(df)

        signal = np.zeros(n)

        for i in range(n):
            if np.isnan(indicator[i]):
                continue

            # 基础买入条件：指标 > 买入阈值
            base_buy = indicator[i] > self.buy_threshold
            # 基础卖出条件：指标 < 卖出阈值
            base_sell = indicator[i] < self.sell_threshold

            buy_signal = base_buy
            sell_signal = base_sell

            # 价格过滤：均线上行趋势
            if self.use_price_filter and base_buy:
                if i >= self.ma_lookback:
                    ma_now = df["ma"].iloc[i]
                    ma_before = df["ma"].iloc[i - self.ma_lookback]
                    if not (ma_now > ma_before):
                        buy_signal = False

            # 成交量相关性过滤
            if self.use_volume_filter and base_buy:
                if i >= self.volume_corr_period:
                    if not (df["vol_corr"].iloc[i] > 0):
                        buy_signal = False

            if buy_signal:
                signal[i] = 1
            elif sell_signal:
                signal[i] = 0
            else:
                # 延续之前信号（前向填充在后面做）
                signal[i] = np.nan

        df["signal_raw"] = signal.copy()

        # 前向填充 NaN 信号
        signal_filled = signal.copy()
        last_valid = 0
        for i in range(n):
            if np.isnan(signal_filled[i]):
                signal_filled[i] = last_valid
            else:
                last_valid = signal_filled[i]

        df["signal"] = signal_filled

        # 生成仓位（滞后一期，避免未来信息）
        pos = np.concatenate([[0], signal_filled[:-1]])
        df["pos"] = pos

        return df

    def get_signal_counts(self, df: pd.DataFrame) -> Dict:
        """获取买卖信号统计"""
        if "signal" not in df.columns:
            return {}

        signals = df["signal"].values
        # 统计信号变化
        signal_changes = np.diff(np.concatenate([[0], signals]))
        buy_count = np.sum(signal_changes == 1)
        sell_count = np.sum(signal_changes == -1)

        return {
            "买入信号次数": int(buy_count),
            "卖出信号次数": int(sell_count),
            "持仓天数": int(np.sum(df["pos"].values == 1)),
        }

    def get_config_summary(self) -> Dict:
        """获取策略配置摘要"""
        method_names = {
            "slope": "斜率策略",
            "zscore": "标准分策略",
            "corrected": "修正标准分策略",
            "right_biased": "右偏标准分策略",
        }
        return {
            "策略方法": method_names.get(self.method, self.method),
            "回归周期(N)": self.n_regression,
            "标准分周期(M)": self.m_standardize,
            "信号阈值": f"{self.threshold} (买{self.buy_threshold}/卖{self.sell_threshold})" if self.method == "slope" else self.threshold,
            "价格过滤": "启用" if self.use_price_filter else "关闭",
            "成交量过滤": "启用" if self.use_volume_filter else "关闭",
        }
