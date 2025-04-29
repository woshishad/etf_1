import akshare as ak
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import random
import copy
from tqdm import tqdm

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
# 设置随机种子保证可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 遗传算法参数
POP_SIZE = 20  # 种群大小
N_GENERATIONS = 30  # 进化代数
CROSS_RATE = 0.8  # 交叉概率
MUTATION_RATE = 0.1  # 变异概率
ELITE_SIZE = 2  # 精英个体数量

# 特征列表（你当前的）
FEATURES = [
    'close', 'volume', 'ma5', 'ma20', 'min_golden_cross',
    'min_death_cross', 'macd_cross', 'intra_minute', 'trading_phase','vol_break',
        'price_break_high', 'price_break_low',
        'rsi_overbought', 'rsi_oversold',
        'order_flow'
]
N_FEATURES = len(FEATURES)


# 生成初始种群
def generate_population(pop_size, n_features):
    return [np.random.choice([0, 1], size=n_features).tolist() for _ in range(pop_size)]


# 特征子集选择
def select_features(data, feature_mask):
    selected = [f for f, m in zip(FEATURES, feature_mask) if m == 1]
    if not selected:  # 防止无特征的个体
        selected = [random.choice(FEATURES)]
    return selected


# 重定义 create_dataset 支持动态特征列
def create_dataset_dynamic(data, lookback, horizon, feature_columns):
    x, y = [], []
    for i in range(len(data) - lookback - horizon):
        features = data.iloc[i:i + lookback][feature_columns]
        current_close = data.iloc[i + lookback - 1]['close']
        future_close = data.iloc[i + lookback + horizon - 1]['close']
        label = 1 if future_close > current_close else 0
        x.append(features.values)
        y.append(label)
    return np.array(x), np.array(y)

lookback = 60  # 使用过去60分钟数据
horizon = 30  # 预测30分钟后涨跌

# 数据准备函数
def to_tensor(x, y):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)  # BCEWithLogitsLoss需要float类型
    return x_tensor, y_tensor
# 个体适应度评估
def fitness(individual):
    feature_subset = select_features(df, individual)
    x, y = create_dataset_dynamic(df, lookback, horizon, feature_subset)
    if len(x) == 0:
        return 0  # 无法训练
    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    # 划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
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

    # 初始化小型模型（防止太慢）
    model = LSTMModel(input_size=x_train.shape[2], hidden_size=32, output_size=1, num_layers=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 快速小批量训练
    model.train()
    for epoch in range(5):  # 少量epoch快速评估
        optimizer.zero_grad()
        output = model(x_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(x_test_tensor)
        preds = torch.sigmoid(output) > 0.5
        preds = preds.int().numpy()
    # 计算F1
    try:
        f1 = f1_score(y_test, preds)
    except:
        f1 = 0.0
    return f1


# 选择操作（精英保留 + 轮盘赌）
def select(pop, fitnesses):
    # 精英直接保留
    elite_idx = np.argsort(fitnesses)[-ELITE_SIZE:]
    elites = [copy.deepcopy(pop[i]) for i in elite_idx]

    # 轮盘赌选择
    total_fit = sum(fitnesses)
    probs = [f / total_fit for f in fitnesses]
    selected = []
    for _ in range(len(pop) - ELITE_SIZE):
        idx = np.random.choice(len(pop), p=probs)
        selected.append(copy.deepcopy(pop[idx]))
    return elites + selected


# 两点交叉
def crossover(parent1, parent2):
    if random.random() < CROSS_RATE:
        idx1, idx2 = sorted(random.sample(range(len(parent1)), 2))
        child1 = parent1[:idx1] + parent2[idx1:idx2] + parent1[idx2:]
        child2 = parent2[:idx1] + parent1[idx1:idx2] + parent2[idx2:]
        return child1, child2
    else:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)


# 变异
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]  # 翻转
    return individual


# 遗传算法主程序
def genetic_algorithm():
    population = generate_population(POP_SIZE, N_FEATURES)
    best_f1 = 0
    best_features = None

    for gen in range(N_GENERATIONS):
        fitnesses = [fitness(individual) for individual in tqdm(population, desc=f'Gen {gen + 1}')]

        # 保存最优
        max_f1 = max(fitnesses)
        if max_f1 > best_f1:
            best_f1 = max_f1
            best_features = population[np.argmax(fitnesses)]

        print(f'Generation {gen + 1} Best F1: {max_f1:.4f}')

        # 进化
        selected = select(population, fitnesses)
        children = []
        for i in range(0, len(selected) - 1, 2):
            child1, child2 = crossover(selected[i], selected[i + 1])
            children.append(mutate(child1))
            children.append(mutate(child2))
        population = children

    print('\n=== Evolution Complete ===')
    print('Best F1 Score:', best_f1)
    selected_features = select_features(df, best_features)
    print('Best Feature Subset:', selected_features)
    return best_features, best_f1


# 运行
best_features, best_score = genetic_algorithm()
