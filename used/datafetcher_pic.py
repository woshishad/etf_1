import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
# 获取上证指数的日内分钟级数据（比如获取最近的几分钟数据）
stock_minute_data = ak.stock_zh_a_minute(symbol="sz159919", period="5", adjust="qfq")
# 确保数据列的类型是 float 或 int
stock_minute_data['open'] = pd.to_numeric(stock_minute_data['open'], errors='coerce')
stock_minute_data['high'] = pd.to_numeric(stock_minute_data['high'], errors='coerce')
stock_minute_data['low'] = pd.to_numeric(stock_minute_data['low'], errors='coerce')
stock_minute_data['close'] = pd.to_numeric(stock_minute_data['close'], errors='coerce')
stock_minute_data['volume'] = pd.to_numeric(stock_minute_data['volume'], errors='coerce')

# 查看是否有缺失值
print(stock_minute_data.isnull().sum())

# 如果有缺失值，可以选择填充或者删除缺失数据
stock_minute_data.dropna(inplace=True)  # 删除缺失值

# 假设 stock_minute_data 是你的分钟级数据
# 确保数据框有 'datetime', 'open', 'high', 'low', 'close', 'volume' 列
stock_minute_data['datetime'] = pd.to_datetime(stock_minute_data['day'])
stock_minute_data.set_index('datetime', inplace=True)
# 获取最近 1 天的数据
stock_minute_data_last_day = stock_minute_data[stock_minute_data.index >= pd.Timestamp.now() - pd.Timedelta(days=1)]

# 选择需要的数据列
ohlc_data = stock_minute_data_last_day[['open', 'high', 'low', 'close', 'volume']]

# 绘制K线图
mpf.plot(ohlc_data, type='candle', style='charles', title='Last Day Stock Minute-level Candlestick Chart', volume=True)

# 选择需要的数据列
ohlc_data = stock_minute_data[['open', 'high', 'low', 'close', 'volume']]

# 绘制K线图
mpf.plot(ohlc_data, type='candle', style='charles', title='Stock Minute-level Candlestick Chart', volume=True)
