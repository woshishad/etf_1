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
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    data = data[data.index >= pd.Timestamp.now() - pd.Timedelta(days=150)]
    data_30m = data.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    minute_data = data[['open', 'high', 'low', 'close', 'volume']]
    data_30m = data_30m[['open', 'high', 'low', 'close', 'volume']]
    return minute_data, data_30m

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
    merged_df['30_close'] = merged_df['ma30_8'].ffill().bfill()
    merged_df['ma30_8'] = merged_df['ma30_8'].ffill().bfill()
    merged_df['ma30_24'] = merged_df['ma30_24'].ffill().bfill()
    merged_df['vol_30m_ma'] = merged_df['vol_30m_ma'].ffill().fillna(0)

    # === 时间特征 ===
    merged_df['intra_minute'] = (merged_df.index - merged_df.index.normalize()).total_seconds() / 60 - 150
    merged_df['trading_phase'] = np.select(
        [merged_df['intra_minute'] < 120, merged_df['intra_minute'] >= 120],
        [0, 1],
        default=-1
    )

    # 安全删除空值
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_df = merged_df.dropna(how='any', subset=['30_close','ma5', 'ma20', 'ma30_8'])

    return valid_df

def create_multi_scale_dataset(data, lookback, horizon, day_window=3):
    """创建多时间尺度数据集"""
    x_minute = []
    x_30m = []
    y = []

    interval = 30  # 每30分钟一个点
    steps_30m = day_window * 24  # 3天 = 72个30分钟点
    window_size_30m = interval * steps_30m  # 实际需要2160分钟长度

    for i in range(window_size_30m, len(data) - lookback - horizon):
        # 分钟级数据窗口
        minute_window = data.iloc[i:i + lookback]

        # 30分钟级数据窗口（以30分钟为间隔向前取72个点）
        thirty_min_indices = range(i - window_size_30m, i, interval)
        thirty_min_window_data = data.iloc[list(thirty_min_indices)][['30_close','ma30_8', 'ma30_24', 'vol_30m_ma']]

        # 特征处理
        minute_features = minute_window[['close', 'volume', 'ma5', 'ma20',
                                         'min_golden_cross', 'min_death_cross']]

        # 标签生成
        current_close = minute_window.iloc[-1]['close']
        future_close = data.iloc[i + lookback + horizon - 1]['close']
        label = 1 if future_close > current_close else 0

        x_minute.append(minute_features.values)
        x_30m.append(thirty_min_window_data.values)
        y.append(label)

    return np.array(x_minute), np.array(x_30m), np.array(y)


class MultiScaleRNN(nn.Module):
    def __init__(self, minute_input_size, thirty_min_input_size, hidden_size, output_size):
        super().__init__()
        self.minute_rnn = nn.RNN(minute_input_size, hidden_size, batch_first=True)
        self.thirty_rnn = nn.RNN(thirty_min_input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x_minute, x_30m):
        _, h_minute = self.minute_rnn(x_minute)
        _, h_30m = self.thirty_rnn(x_30m)
        combined = torch.cat([h_minute[-1], h_30m[-1]], dim=1)
        return self.fc(combined).squeeze(-1)





# 参数设置
lookback = 60
thirty_min_window = 72
horizon = 30
minute_input_size = 6
thirty_min_input_size = 4
output_size = 1

# 数据准备
minute_data, data_30m = fetch_data()
merged_data = calculate_features(minute_data, data_30m)
x_minute, x_30m, y = create_multi_scale_dataset(merged_data, lookback, horizon)

# 数据标准化
minute_scaler = StandardScaler()
thirty_min_scaler = StandardScaler()

x_minute = minute_scaler.fit_transform(x_minute.reshape(-1, x_minute.shape[-1])).reshape(x_minute.shape)
x_30m = thirty_min_scaler.fit_transform(x_30m.reshape(-1, x_30m.shape[-1])).reshape(x_30m.shape)

# 数据分割
x_minute_train, x_minute_test, x_30m_train, x_30m_test, y_train, y_test = train_test_split(x_minute, x_30m, y, test_size=0.2, shuffle=False)

# 转换为Tensor
def to_tensor(x_minute, x_30m, y):
    return (torch.tensor(x_minute, dtype=torch.float32),
            torch.tensor(x_30m, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))

x_minute_train, x_30m_train, y_train = to_tensor(x_minute_train, x_30m_train, y_train)
x_minute_test, x_30m_test, y_test = to_tensor(x_minute_test, x_30m_test, y_test)

# 初始化模型
# 初始化模型
model = MultiScaleRNN(minute_input_size, thirty_min_input_size, hidden_size=64, output_size=1)

# 使用RNN替代CNN部分

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, x_minute, x_30m, y, criterion, optimizer, epochs=100, batch_size=64):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(x_minute), batch_size):
            x_m_batch = x_minute[i:i+batch_size]
            x_30m_batch = x_30m[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(x_m_batch, x_30m_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(x_minute):.4f}')

train(model, x_minute_train, x_30m_train, y_train, criterion, optimizer, epochs=300)


# 评估函数
def evaluate(model, x_minute, x_30m, y):
    model.eval()
    with torch.no_grad():
        outputs = model(x_minute, x_30m)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == y).float().mean()
        precision = precision_score(y.numpy(), preds.numpy(), zero_division=0)
        recall = recall_score(y.numpy(), preds.numpy(), zero_division=0)
        f1 = f1_score(y.numpy(), preds.numpy(), zero_division=0)
    print(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

evaluate(model, x_minute_test, x_30m_test, y_test)

