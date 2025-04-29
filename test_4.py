import akshare as ak
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn

def fetch_data():
    data = pd.read_csv('IM9999_1m.csv', encoding='utf-8')
    # 假设 stock_minute_data 是你的分钟级数据
    # 确保数据框有 'datetime', 'open', 'high', 'low', 'close', 'volume' 列
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    # 获取最近 一年 的数据
    data = data[data.index >= pd.Timestamp.now() - pd.Timedelta(days=360)]
    # 重采样为30分钟级别数据
    data_30m = data.resample('30min').agg({
        'open': 'first',  # 30分钟内的第一个开盘价
        'high': 'max',  # 30分钟内的最高价
        'low': 'min',  # 30分钟内的最低价
        'close': 'last',  # 30分钟内的最后一个收盘价
        'volume': 'sum'  # 30分钟内的成交量总和
    })

    # 选择需要的数据列
    minute_data = data[['open', 'high', 'low', 'close', 'volume']]
    data_30m=data_30m[['open', 'high', 'low', 'close', 'volume']]
    return minute_data,data_30m


def calculate_features(minute_data, data_30m):
    """计算多时间尺度特征（带健壮性检查）"""
    # 复制原始数据避免污染
    minute_df = minute_data.copy(deep=True)
    df_30m = data_30m.copy(deep=True)

    # === 分钟级特征 ===
    # 添加容错机制
    try:
        minute_df['ma5'] = minute_df['close'].rolling(5, min_periods=1).mean()
        minute_df['ma20'] = minute_df['close'].rolling(20, min_periods=1).mean()
        minute_df['min_golden_cross'] = ((minute_df['ma5'] > minute_df['ma20']) &
                                         (minute_df['ma5'].shift(1) <= minute_df['ma20'].shift(1))).astype(int)
        minute_df['min_death_cross'] = ((minute_df['ma5'] < minute_df['ma20']) &
                                        (minute_df['ma5'].shift(1) >= minute_df['ma20'].shift(1))).astype(int)
    except Exception as e:
        print(f"分钟特征计算失败: {str(e)}")

    # === 30分钟级特征 ===
    # 确保足够的历史数据
    if len(df_30m) < 24:
        raise ValueError("30分钟数据不足（至少需要24个周期）")

    # 带异常值的滚动计算
    df_30m['30_close'] = df_30m['close']
    df_30m['ma30_8'] = df_30m['close'].rolling(8, min_periods=1).mean()
    df_30m['ma30_24'] = df_30m['close'].rolling(24, min_periods=1).mean()
    df_30m['vol_30m_ma'] = df_30m['volume'].rolling(24, min_periods=1).mean()

    # === 数据合并 ===
    # 使用向前填充合并
    merged_df = minute_df.merge(
        df_30m[['30_close','ma30_8', 'ma30_24', 'vol_30m_ma']],
        left_index=True,
        right_index=True,
        how='left'
    )

    # 分阶段填充
    merged_df['ma30_8'] = merged_df['ma30_8'].fillna(method='ffill').fillna(method='bfill')
    merged_df['ma30_24'] = merged_df['ma30_24'].fillna(method='ffill').fillna(method='bfill')
    merged_df['vol_30m_ma'] = merged_df['vol_30m_ma'].fillna(method='ffill').fillna(0)  # 最后用0填充

    # === 时间特征 ===
    merged_df['intra_minute'] = (merged_df.index - merged_df.index.normalize()).total_seconds() / 60 - 150
    merged_df['trading_phase'] = np.select(
        [merged_df['intra_minute'] < 120, merged_df['intra_minute'] >= 120],
        [0, 1],
        default=-1
    )

    # 安全删除空值
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_df = merged_df.dropna(how='any', subset=['ma5', 'ma20', 'ma30_8'])

    return valid_df
minute_data, data_30m = fetch_data()
merged_data = calculate_features(minute_data, data_30m)
print(merged_data)