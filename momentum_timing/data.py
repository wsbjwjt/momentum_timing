"""
数据获取模块
基于 baostock 免费数据源，支持 A 股指数和个股日线数据获取。
包含网络重试机制、本地缓存、进度条显示。
"""

import os
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


# 缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".data_cache")


# 预定义的指数和ETF代码列表
PREDEFINED_INDICES = {
    "沪深300": "sh.000300",
    "中证500": "sh.000905",
    "上证50": "sh.000016",
    "中证1000": "sh.000852",
    "创业板指": "sz.399006",
    "科创50": "sh.000688",
}

PREDEFINED_ETFS = {
    "沪深300ETF": "sh.510300",
    "中证500ETF": "sh.510500",
    "上证50ETF": "sh.510050",
    "创业板ETF": "sz.159915",
    "科创50ETF": "sh.588000",
}


def _get_cache_path(code: str, start: str, end: str) -> str:
    """生成缓存文件路径"""
    key = f"{code}_{start}_{end}"
    hash_key = hashlib.md5(key.encode()).hexdigest()
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{hash_key}.pkl")


def _save_cache(cache_path: str, df: pd.DataFrame):
    """保存数据到本地缓存"""
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass


def _load_cache(cache_path: str) -> Optional[pd.DataFrame]:
    """从本地缓存加载数据"""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True,
)
def _fetch_from_baostock(code: str, start_date: str, end_date: str, adjust: str = "3") -> pd.DataFrame:
    """
    从 baostock 获取日线数据（带自动重试）。

    参数:
        code: 股票/指数代码，格式如 "sh.000300"
        start_date: 开始日期 "YYYY-MM-DD"
        end_date: 结束日期 "YYYY-MM-DD"
        adjust: 复权类型 - "1"后复权, "2"前复权, "3"不复权

    返回:
        DataFrame with columns: date, open, high, low, close, volume, amount
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        bs.logout()
        raise ConnectionError(f"baostock 登录失败: {lg.error_msg}")

    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjust,
    )

    if rs.error_code != "0":
        bs.logout()
        raise RuntimeError(f"数据查询失败: {rs.error_msg}")

    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())

    bs.logout()

    if not data_list:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

    df = pd.DataFrame(data_list, columns=rs.fields)

    # 转换数据类型
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 剔除空数据行
    df = df[df["close"].notna() & (df["close"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)

    return df


def fetch_index_data(
    code: str,
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    获取指数日线数据。

    参数:
        code: 指数代码，如 "sh.000300" 或直接用名称如 "沪深300"
        start_date: 开始日期
        end_date: 结束日期（默认为今天）
        use_cache: 是否使用本地缓存
        verbose: 是否显示进度信息

    返回:
        DataFrame: date, open, high, low, close, volume, amount
    """
    # 解析代码
    if code in PREDEFINED_INDICES:
        code = PREDEFINED_INDICES[code]

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if verbose:
        print(f"📊 获取 {code} 数据 ({start_date} ~ {end_date})")

    # 尝试缓存
    cache_path = _get_cache_path(code, start_date, end_date)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None and len(cached) > 0:
            if verbose:
                print(f"   从缓存加载 ({len(cached)} 条)")
            return cached

    # 从 baostock 获取
    try:
        df = _fetch_from_baostock(code, start_date, end_date)
    except Exception as e:
        raise RuntimeError(f"获取 {code} 数据失败（已重试3次）: {e}")

    if verbose:
        print(f"   获取 {len(df)} 条日线数据")

    # 保存缓存
    if use_cache and len(df) > 0:
        _save_cache(cache_path, df)

    return df


def fetch_multiple_indices(
    codes: List[str],
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    批量获取多个指数的数据。

    参数:
        codes: 指数代码列表
        start_date: 开始日期
        end_date: 结束日期
        use_cache: 是否使用缓存

    返回:
        Dict[code, DataFrame]
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    results = {}
    failed = []

    for code in tqdm(codes, desc="获取指数数据"):
        try:
            df = fetch_index_data(code, start_date, end_date, use_cache, verbose=False)
            if len(df) > 0:
                results[code] = df
            else:
                failed.append(code)
        except Exception as e:
            failed.append(code)
            tqdm.write(f"  ⚠️ {code}: 获取失败 - {e}")

    if failed:
        print(f"\n⚠️ {len(failed)} 个代码获取失败: {failed}")

    return results


def fetch_stock_data(
    code: str,
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    获取个股日线数据（后复权）。

    参数:
        code: 股票代码，如 "sh.600519"（baostock格式）
        start_date: 开始日期
        end_date: 结束日期
        use_cache: 是否使用缓存
        verbose: 是否显示进度

    返回:
        DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    cache_path = _get_cache_path(code, start_date, end_date)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None and len(cached) > 0:
            return cached

    try:
        # 股票数据使用后复权
        df = _fetch_from_baostock(code, start_date, end_date, adjust="1")
    except Exception as e:
        raise RuntimeError(f"获取 {code} 数据失败: {e}")

    if use_cache and len(df) > 0:
        _save_cache(cache_path, df)

    if verbose and len(df) == 0:
        print(f"  ⚠️ {code}: 无数据")

    return df


def fetch_stock_pool(
    codes: List[str],
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    min_days: int = 200,
    use_cache: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    批量获取股票池数据。

    参数:
        codes: baostock 格式代码列表
        start_date: 开始日期
        end_date: 结束日期
        min_days: 最少需要的数据天数，不足则丢弃
        use_cache: 是否使用缓存
        verbose: 是否显示进度

    返回:
        Dict[code, DataFrame]
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    results = {}
    failed = []

    if verbose:
        print(f"📦 开始获取 {len(codes)} 只股票数据...")

    for code in tqdm(codes, desc="获取股票数据", disable=not verbose):
        try:
            df = fetch_stock_data(code, start_date, end_date, use_cache, verbose=False)
            if len(df) >= min_days:
                results[code] = df
            elif len(df) > 0:
                failed.append(f"{code}(仅{len(df)}天)")
            else:
                failed.append(code)
        except Exception as e:
            failed.append(code)

    if verbose:
        print(f"✅ 成功: {len(results)} 只, 失败/不足: {len(failed)} 只")

    return results


def get_hs300_stocks() -> List[str]:
    """
    获取沪深300成分股代码列表（使用 baostock）。
    注意：由于 baostock 不直接提供成分股查询，这里返回一个预定义的核心股票池。
    """
    # 提供一个代表性的A股样本池（优先大盘蓝筹股，数据更完整）
    sample_pool = []
    # 沪市核心蓝筹
    for prefix, nums in [
        ("sh.60", range(600000, 600100, 5)),
        ("sh.60", range(600300, 600400, 5)),
        ("sh.60", range(600500, 600600, 5)),
        ("sh.60", range(601000, 601200, 5)),
        ("sh.60", range(601300, 601500, 5)),
        ("sh.60", range(601600, 601700, 5)),
        ("sh.60", range(601800, 601900, 5)),
        ("sh.60", range(603000, 603100, 5)),
    ]:
        for n in nums:
            sample_pool.append(f"{prefix}{n:04d}")

    # 深市核心
    for prefix, nums in [
        ("sz.00", range(1, 100, 5)),
        ("sz.00", range(100, 200, 5)),
        ("sz.00", range(300, 400, 5)),
        ("sz.00", range(500, 600, 5)),
        ("sz.00", range(700, 900, 5)),
        ("sz.00", range(2000, 2100, 5)),
    ]:
        for n in nums:
            sample_pool.append(f"{prefix}{n:04d}")

    return sample_pool


def get_common_indices() -> List[Dict]:
    """获取常用指数列表（用于展示选择）"""
    return [
        {"name": name, "code": code, "category": "指数"}
        for name, code in PREDEFINED_INDICES.items()
    ]


def get_common_etfs() -> List[Dict]:
    """获取常用ETF列表"""
    return [
        {"name": name, "code": code, "category": "ETF"}
        for name, code in PREDEFINED_ETFS.items()
    ]


def clear_cache():
    """清除所有数据缓存"""
    if os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("✅ 缓存已清除")
