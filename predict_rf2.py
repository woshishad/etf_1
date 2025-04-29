import akshare as ak
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn

def fetch_minute_data():
    df = pd.read_csv('IM9999_1m.csv', encoding='utf-8')
    # 假设 stock_minute_data 是你的分钟级数据
    # 确保数据框有 'datetime', 'open', 'high', 'low', 'close', 'volume' 列
    df['datetime'] = pd.to_datetime(df['datetime'])

    df.set_index('datetime', inplace=True)

    # 获取最近 一年 的数据
    df = df[df.index >= pd.Timestamp.now() - pd.Timedelta(days=80)]

    ohlc_data = df[['open', 'high', 'low', 'close', 'volume']]

    return ohlc_data


def load_data():
    """加载并合并日线、分钟数据"""
    minute = fetch_minute_data()
    minute = minute.reset_index()
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    minute[numeric_columns] = minute[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return minute.set_index('datetime').dropna()

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

    df['ma5'] = df['close'].rolling(5).mean()

    # 金叉死叉信号
    df['min_golden_cross'] = ((df['ma5'] > df['ma20']) & (df['ma5'].shift(1) <= df['ma20'].shift(1))).astype(int)
    df['min_death_cross'] = ((df['ma5'] < df['ma20']) & (df['ma5'].shift(1) >= df['ma20'].shift(1))).astype(int)

    # ====================
    # 五大策略信号
    # ====================
    # 1. MACD趋势信号（经典参数）
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_cross'] = np.where(macd > signal, 1, -1)  # 1为金叉，-1为死叉

    # 2. 成交量突变（3倍标准差突破）
    vol_mean = df['volume'].rolling(30).mean()
    vol_std = df['volume'].rolling(30).std()
    df['vol_break'] = ((df['volume'] > vol_mean + 2 * vol_std) | (df['volume'] < vol_mean - 2 * vol_std)).astype(int)

    # 3. 价格通道突破（30分钟高点/低点）
    df['high_30'] = df['high'].rolling(30).max()
    df['low_30'] = df['low'].rolling(30).min()
    df['price_break_high'] = (df['close'] > df['high_30'].shift(1)).astype(int)
    df['price_break_low'] = (df['close'] < df['low_30'].shift(1)).astype(int)

    # 4. 动态RSI超买超卖（自适应阈值）
    rsi_period = 14
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    # 动态阈值：基于过去30分钟RSI的波动
    rsi_std = df['rsi'].rolling(30).std()
    df['rsi_overbought'] = (df['rsi'] > 70 + rsi_std).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30 - rsi_std).astype(int)

    # 5. 订单流不平衡（量价背离增强版）
    price_change = df['close'].pct_change()
    volume_change = df['volume'].pct_change()
    df['order_flow'] = np.where(
        (price_change > 0) & (volume_change > 0), 1,  # 价量齐升
        np.where(
            (price_change < 0) & (volume_change > 0), -1,  # 价跌量增
            0  # 其他情况
        )
    )

    # ====================
    # 时间特征工程
    # ====================
    # 日内分钟数（从9:30开始计算）
    df['intra_minute'] = (df.index - df.index.normalize()) / pd.Timedelta(minutes=1) - 150
    # 交易时段阶段（早盘/午盘）
    df['trading_phase'] = np.where(df['intra_minute'] < 120, 0, 1)  # 0:早盘 1:午盘
    return df.dropna()

df = calculate_technical_features(df)
feature_columns = [
        'close','volume',
        'ma5', 'ma20','min_golden_cross', 'min_death_cross',
        'macd_cross' ,'intra_minute', 'trading_phase'
    ]
used_feature_columns=[
         'vol_break',
        'price_break_high', 'price_break_low',
        'rsi_overbought', 'rsi_oversold',
        'order_flow']

def create_dataset(data, lookback, horizon):
    """创建时间序列样本（分类任务）"""
    x, y = [], []
    for i in range(len(data) - lookback - horizon):
        features = data.iloc[i:i + lookback][feature_columns]
        # 获取当前时间点和未来时间点的收盘价
        current_close = data.iloc[i + lookback - 1]['close']
        future_close = data.iloc[i + lookback + horizon - 1]['close']
        # 生成涨跌标签（1=涨，0=跌）
        label = 1 if future_close > current_close else 0
        x.append(features.values)
        y.append(label)
    return np.array(x), np.array(y)


lookback = 60  # 使用过去60分钟数据
horizon = 30  # 预测30分钟后涨跌
(x, y) = create_dataset(df, lookback, horizon)

# 数据标准化（仅特征）
scaler = StandardScaler()
x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)


# 数据准备函数
def to_tensor(x, y):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)  # BCEWithLogitsLoss需要float类型
    return x_tensor, y_tensor


# 数据拆分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
x_train_tensor, y_train_tensor = to_tensor(x_train, y_train)
x_test_tensor, y_test_tensor = to_tensor(x_test, y_test)


# LSTM模型（分类任务）
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        return out.squeeze(-1)  # 输出形状从(batch_size, 1)变为(batch_size,)


# 模型参数
input_size = x_train.shape[2]
hidden_size = 64
output_size = 1  # 二元分类输出一个值
num_layers = 2
batch_size = 64
epochs = 100
learning_rate = 0.0005

# 初始化模型
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 训练函数（修改后）
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


# 训练模型
train(model, x_train_tensor, y_train_tensor, criterion, optimizer, epochs, batch_size)


# 预测函数（分类任务）
def predict(model, data, lookback=60, horizon=30):
    model.eval()
    x_input = data[feature_columns].values[-lookback:]
    x_input = x_input.reshape(1, lookback, -1)
    # 标准化
    x_input = scaler.transform(x_input.reshape(-1, x_input.shape[-1])).reshape(x_input.shape)
    # 预测
    with torch.no_grad():
        logits = model(torch.tensor(x_input, dtype=torch.float32))
    prob = torch.sigmoid(logits).item()
    return 1 if prob > 0.5 else 0, prob


# 评估模型
recent_data = df[df.index >= pd.Timestamp.now() - pd.Timedelta(days=80)]
predictions = []
actual_labels = []

for i in range(len(recent_data) - lookback - horizon):
    test_data = recent_data.iloc[i:i + lookback]
    # 获取实际标签
    current_close = test_data.iloc[-1]['close']
    future_close = recent_data.iloc[i + lookback + horizon - 1]['close']
    actual_label = 1 if future_close > current_close else 0
    actual_labels.append(actual_label)
    # 获取预测结果
    pred_label, _ = predict(model, test_data)
    predictions.append(pred_label)

# 计算评估指标
accuracy = np.mean(np.array(predictions) == np.array(actual_labels))
precision = precision_score(actual_labels, predictions)
recall = recall_score(actual_labels, predictions)
f1 = f1_score(actual_labels, predictions)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# 绘制预测结果
plt.figure(figsize=(14, 7))
plt.plot(actual_labels, label='Actual', marker='o', linestyle='--', alpha=0.7)
plt.plot(predictions, label='Predicted', marker='x', linestyle='None', alpha=0.7)
plt.title('Actual vs Predicted Price Movement (1=Up, 0=Down)')
plt.xlabel('Time Step')
plt.ylabel('Movement')
plt.legend()
plt.show()