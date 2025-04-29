import akshare as ak
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn


def fetch_30min_data():
    # 读取原始数据（假设原始数据是1分钟级别）
    df = pd.read_csv('IM9999_1m.csv', encoding='utf-8')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # 将1分钟数据聚合为30分钟数据
    df_30min = df.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # 获取最近80天的数据（约3840个30分钟周期）
    df_30min = df_30min[df_30min.index >= pd.Timestamp.now() - pd.Timedelta(days=80)]
    return df_30min


def load_data():
    """加载并处理30分钟级别数据"""
    data_30min = fetch_30min_data()
    return data_30min.dropna()


df = load_data()


def calculate_technical_features(df):
    """计算技术指标（适配30分钟级别）"""
    # 收益率（5个周期=2.5小时，15个周期=7.5小时）
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_15'] = df['close'].pct_change(15)

    # 布林带（20个周期=10小时）
    df['ma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['upper_band'] = df['ma20'] + 2 * df['std20']
    df['lower_band'] = df['ma20'] - 2 * df['std20']

    # RSI（14个周期=7小时）
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))

    # 量价特征（30个周期=15小时）
    df['price_high'] = (df['high'] == df['high'].rolling(30).max()).astype(int)
    df['vol_down'] = (df['volume'] < df['volume'].rolling(30).mean()).astype(int)
    df['divergence'] = df['price_high'] & df['vol_down']

    # 均线系统
    df['ma5'] = df['close'].rolling(5).mean()
    df['min_golden_cross'] = ((df['ma5'] > df['ma20']) & (df['ma5'].shift(1) <= df['ma20'].shift(1))).astype(int)
    df['min_death_cross'] = ((df['ma5'] < df['ma20']) & (df['ma5'].shift(1) >= df['ma20'].shift(1))).astype(int)

    # MACD趋势（保持经典参数）
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_cross'] = np.where(macd > signal, 1, -1)

    # 时间特征（按30分钟周期计算）
    df['intra_period'] = (df.index.hour - 9) * 2 + (df.index.minute // 30)  # 从9:30开始计算
    df['trading_phase'] = np.where(df['intra_period'] < 4, 0, 1)  # 0:早盘 1:午盘

    return df.dropna()


df = calculate_technical_features(df)

feature_columns = [
    'close', 'volume',
    'ma5', 'ma20', 'min_golden_cross', 'min_death_cross',
    'macd_cross', 'intra_period', 'trading_phase'
]

def create_dataset(data, lookback, horizon):
    """创建时间序列样本（适配30分钟级别）"""
    x, y = [], []
    for i in range(len(data) - lookback - horizon):
        features = data.iloc[i:i + lookback][feature_columns]
        current_close = data.iloc[i + lookback - 1]['close']
        future_close = data.iloc[i + lookback + horizon - 1]['close']
        label = 1 if future_close > current_close else 0
        x.append(features.values)
        y.append(label)
    return np.array(x), np.array(y)


# 调整时间窗口参数（20个30分钟周期=10小时，预测1个周期后）
lookback = 20
horizon = 1
x, y = create_dataset(df, lookback, horizon)

# 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

def to_tensor(x, y):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return x_tensor, y_tensor


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
x_train_tensor, y_train_tensor = to_tensor(x_train, y_train)
x_test_tensor, y_test_tensor = to_tensor(x_test, y_test)


# 调整模型参数
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)

input_size = x_train.shape[2]
hidden_size = 64
output_size = 1
num_layers = 2
batch_size = 32  # 减小batch_size以适应数据量
epochs = 100
learning_rate = 0.0005

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 训练函数保持不变
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

    # 确保test_data包含所有的特征列
    test_data = data[feature_columns].values[-lookback:]
    print("Test data shape:", test_data.shape)  # 应为 (lookback, 9)
    x_input = test_data.reshape(1, -1, test_data.shape[1])
    print("Reshaped input shape:", x_input.shape)  # 应为 (1, lookback, 9)

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
    pred_label, _ = predict(model, test_data,lookback=20)
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