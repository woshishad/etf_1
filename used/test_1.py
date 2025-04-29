import akshare as ak
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
import matplotlib.pyplot as plt

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
    return merged.set_index('datetime')

df = load_data()
print(df)
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

    # 日线特征扩展
    df['d_range_pct'] = (df['d_high'] - df['d_low']) / df['d_low']
    return df.dropna()

df = calculate_technical_features(df)
print(df)
# ====================
# 3. 滑动窗口生成
# ====================
def create_dataset(data, lookback=60, horizon=30):
    """创建时间序列样本"""
    X, y = [], []
    for i in range(len(data)-lookback-horizon):
        # 输入特征：过去60分钟的所有指标
        features = data.iloc[i:i+lookback][['close', 'volume', 'ma20', 'rsi', 'divergence', 'd_range_pct']]
        # 输出目标：未来30分钟的收盘价
        target = data.iloc[i+lookback:i+lookback+horizon]['close'].values
        X.append(features.values)
        y.append(target)
    return np.array(X), np.array(y)

lookback = 60  # 使用过去60分钟数据
horizon = 30   # 预测未来30分钟
X, y = create_dataset(df, lookback, horizon)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# ====================
# 4. LSTM模型构建
# ====================
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(lookback, 6)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(horizon)  # 直接输出30个时间点的预测值
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
model.compile(loss='mse', optimizer=optimizer)

# ====================
# 5. 时间序列交叉验证
# ====================
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train,
              epochs=50,
              batch_size=64,
              validation_data=(X_test, y_test),
              verbose=1)


# ====================
# 6. 预测与可视化
# ====================
def plot_predictions(model, X_test, y_test, scaler):
    """可视化预测结果"""
    preds = model.predict(X_test)

    # 反标准化收盘价
    close_scaler = scaler.__dict__['scale_'][0]
    close_mean = scaler.__dict__['mean_'][0]
    y_test = y_test * close_scaler + close_mean
    preds = preds * close_scaler + close_mean

    plt.figure(figsize=(15, 6))
    plt.plot(y_test[-1], label='Actual')
    plt.plot(preds[-1], label='Predicted')
    plt.title('30-Minute Price Prediction')
    plt.legend()
    plt.show()


plot_predictions(model, X_test, y_test, scaler)

# ====================
# 7. 预测评估指标
# ====================
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    direction_acc = np.mean(np.sign(y_true[:, -1] - y_true[:, 0]) ==
                            np.sign(y_pred[:, -1] - y_true[:, 0]))
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | Direction Acc: {direction_acc:.2%}")


evaluate_predictions(y_test, model.predict(X_test))