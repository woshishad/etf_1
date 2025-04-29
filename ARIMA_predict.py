import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def fetch_minute_data():
    df = pd.read_csv('IM9999_1m.csv', encoding='utf-8')
    # 假设 stock_minute_data 是你的分钟级数据
    # 确保数据框有 'datetime', 'open', 'high', 'low', 'close', 'volume' 列
    df['datetime'] = pd.to_datetime(df['datetime'])

    df.set_index('datetime', inplace=True)

    # 获取最近 一年 的数据
    df = df[df.index >= pd.Timestamp.now() - pd.Timedelta(days=60)]

    ohlc_data = df[['open', 'high', 'low', 'close', 'volume']]

    return ohlc_data


def load_data():
    """加载并合并日线、分钟数据"""
    minute = fetch_minute_data()
    minute = minute.reset_index()
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    minute[numeric_columns] = minute[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return minute.set_index('datetime').dropna()




# 补充缺失的方法定义
def arima_rolling_forecast(series, train_size, order=(3, 1, 2), horizon=30):
    """ARIMA滚动预测函数（已补充完整）

    参数：
    series: 完整时间序列数据
    train_size: 训练集长度
    order: ARIMA模型参数(p,d,q)
    horizon: 预测步长

    返回：
    predictions: 涨跌预测标签列表（1=涨，0=跌）
    """
    predictions = []
    history = list(series[:train_size])  # 初始训练集

    # 遍历测试集进行逐步预测
    for t in range(len(series) - train_size - horizon):
        try:
            # 创建并拟合ARIMA模型
            model = ARIMA(history, order=order)
            model_fit = model.fit()

            # 进行多步预测并取第horizon步的预测值
            forecast = model_fit.forecast(steps=horizon)
            pred_close = forecast.iloc[-1]  # 获取最后一步的预测值

            # 生成涨跌标签
            current_close = history[-1]
            predictions.append(1 if pred_close > current_close else 0)

            # 更新历史数据（添加真实值）
            history.append(series[train_size + t])
        except Exception as e:
            print(f"第{t}次预测出错: {str(e)}")
            predictions.append(0)  # 错误处理默认预测下跌
    return predictions



# 初始化数据
df = load_data()
close_series = df['close'].values  # 使用收盘价序列

# 参数设置
train_size = int(len(close_series) * 0.8)  # 训练集占比
horizon = 30  # 预测步长
order = (3, 1, 2)  # ARIMA参数(p,d,q)

# 生成实际标签（修正索引越界问题）
actual_labels = [
    1 if close_series[i + horizon] > close_series[i] else 0
    for i in range(train_size, len(close_series) - horizon)
]

# 执行滚动预测
predictions = arima_rolling_forecast(close_series, train_size, order, horizon)

# 评估指标计算
print(f"准确率: {accuracy_score(actual_labels, predictions):.2%}")
print(f"精确率: {precision_score(actual_labels, predictions):.2%}")
print(f"召回率: {recall_score(actual_labels, predictions):.2%}")
print(f"F1分数: {f1_score(actual_labels, predictions):.2%}")

# 可视化对比（添加时间戳标签）
plt.figure(figsize=(14, 6))
plt.plot(df.index[train_size + horizon:train_size + horizon + len(actual_labels)],
         actual_labels, label='实际走势', alpha=0.6)
plt.plot(df.index[train_size + horizon:train_size + horizon + len(predictions)],
         predictions, label='预测结果', linestyle='--', alpha=0.8)
plt.title('ARIMA模型30分钟价格涨跌预测表现')
plt.ylabel('涨跌标记（1=涨，0=跌）')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
