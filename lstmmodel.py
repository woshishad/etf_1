import akshare as ak
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn


def fetch_minute_data():
    etf_minute_data = ak.stock_zh_a_minute(symbol="sz159919", period="1", adjust="qfq")

    # 假设 stock_minute_data 是你的分钟级数据
    # 确保数据框有 'datetime', 'open', 'high', 'low', 'close', 'volume' 列
    etf_minute_data['datetime'] = pd.to_datetime(etf_minute_data['day'])
    etf_minute_data.set_index('datetime', inplace=True)

    # 获取最近 10 天的数据
    stock_minute_data_last_day = etf_minute_data[etf_minute_data.index >= pd.Timestamp.now() - pd.Timedelta(days=10)]

    # 选择需要的数据列
    ohlc_data = stock_minute_data_last_day[['open', 'high', 'low', 'close', 'volume']]

    return ohlc_data

def fetch_daily_data():
    etf_daily_data= ak.fund_etf_hist_sina(symbol="sz159919")

    # 假设 stock_minute_data 是你的日级数据
    # 确保数据框有 'date', 'open', 'high', 'low', 'close', 'volume' 列
    etf_daily_data['date'] = pd.to_datetime(etf_daily_data['date'])
    etf_daily_data.set_index('date', inplace=True)

    # 获取最近50天的数据
    stock_minute_data_last50_day = etf_daily_data[etf_daily_data.index >= pd.Timestamp.now() - pd.Timedelta(days=50)]

    # 选择需要的数据列
    ohlc_data = stock_minute_data_last50_day[['open', 'high', 'low', 'close', 'volume']]
    return ohlc_data

# ====================
# 1. 数据预处理
# ====================
def load_data():
    """加载并合并日线、分钟数据"""
    daily = fetch_daily_data()
    minute = fetch_minute_data()
    daily = daily.reset_index()
    minute = minute.reset_index()
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    daily[numeric_columns] = daily[numeric_columns].apply(pd.to_numeric, errors='coerce')
    minute[numeric_columns] = minute[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # 将日线数据特征合并到分钟数据
    daily_features = daily[['date', 'open', 'high', 'low','close', 'volume']]
    daily_features.columns = ['date', 'd_open', 'd_high', 'd_low','d_close', 'd_volume']
    merged = pd.merge_asof(minute.sort_values('datetime'),
                           daily_features.sort_values('date'),
                           left_on='datetime', right_on='date',
                           direction='forward')
    return minute.set_index('datetime')

df = load_data()

def calculate_technical_features(df):
    """计算技术指标"""
    # 收益率
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_15'] = df['close'].pct_change(15)

    # 布林带（20分钟窗口）
    df['ma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['upper_band'] = df['ma20'] + 2 * df['std20']
    df['lower_band'] = df['ma20'] - 2 * df['std20']

    # RSI（14分钟窗口）
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))

    # 量价背离（价格新高但成交量下降）
    df['price_high'] = (df['high'] == df['high'].rolling(30).max()).astype(int)
    df['vol_down'] = (df['volume'] < df['volume'].rolling(30).mean()).astype(int)
    df['divergence'] = df['price_high'] & df['vol_down']

    return df.dropna()

df = calculate_technical_features(df)

def create_dataset(data, lookback, horizon):
    """创建时间序列样本"""
    x, y = [], []
    for i in range(len(data)-lookback-horizon):
        # 输入特征：过去60分钟的所有指标
        features = data.iloc[i:i+lookback][['close', 'volume', 'ma20', 'rsi', 'divergence']]
        # 输出目标：未来30分钟的收盘价
        target = data.iloc[i+lookback:i+lookback+horizon]['close'].values
        x.append(features.values)
        y.append(target)
    return np.array(x), np.array(y)

lookback = 60  # 使用过去60分钟数据
horizon = 30   # 预测未来30分钟
(x, y) = create_dataset(df, lookback, horizon)

# 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

# 数据准备函数
def to_tensor(x, y):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return x_tensor, y_tensor


# 数据处理部分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
x_train_tensor, y_train_tensor = to_tensor(x_train, y_train)
x_test_tensor, y_test_tensor = to_tensor(x_test, y_test)

# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# 设置超参数
input_size = x_train.shape[2]
hidden_size = 64
output_size = horizon
num_layers = 2
batch_size = 64
epochs = 100
learning_rate = 0.0005

# 创建模型
model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# 损失函数与优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
def train(model, x_train_tensor, y_train_tensor, criterion, optimizer, epochs, batch_size):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = len(x_train_tensor) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            x_batch = x_train_tensor[start_idx:end_idx]
            y_batch = y_train_tensor[start_idx:end_idx]

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / num_batches:.4f}')


train(model, x_train_tensor, y_train_tensor, criterion, optimizer, epochs, batch_size)

# ====================
# 4. 预测并评估模型
# ====================
def predict(model, data, lookback=60, horizon=30):
    model.eval()
    x_input = data[['close', 'volume', 'ma20', 'rsi', 'divergence']].values[-lookback:]
    x_input = x_input.reshape(1, lookback, -1)  # 输入数据的形状应为 (1, lookback, feature_size)

    # 对输入数据进行标准化
    x_input = scaler.transform(x_input.reshape(-1, x_input.shape[-1])).reshape(x_input.shape)

    # 预测未来30分钟的收盘价
    with torch.no_grad():
        y_pred = model(torch.tensor(x_input, dtype=torch.float32))

    return y_pred.numpy().flatten()

# 获取最近的10天数据用于预测
recent_data = df[df.index >= pd.Timestamp.now() - pd.Timedelta(days=5)]
predictions = []
actual_values = []

# 对每个时间点进行预测
for i in range(len(recent_data) - lookback - horizon):
    test_data = recent_data.iloc[i:i + lookback]
    true_values = recent_data.iloc[i + lookback + horizon]['close']
    predicted_values = predict(model, test_data)[29]

    predictions.append(predicted_values)
    actual_values.append(true_values)

# 转换为numpy数组进行评估
predictions = np.array(predictions)
actual_values = np.array(actual_values)

# 计算均方误差 (MSE)
mse = mean_squared_error(actual_values.flatten(), predictions.flatten())
print(f'Mean Squared Error (MSE): {mse:.4f}')

# ====================
# 5. 绘制预测结果与实际结果的对比图
# ====================
plt.figure(figsize=(14, 7))
plt.plot(np.arange(len(actual_values.flatten())), actual_values.flatten(), label='Actual', color='blue')
plt.plot(np.arange(len(predictions.flatten())), predictions.flatten(), label='Predicted', color='red', linestyle='--')
plt.title('Actual vs Predicted Close Prices for the Next 30 Minutes')
plt.xlabel('Time Step')
plt.ylabel('Close Price')
plt.legend()
plt.show()