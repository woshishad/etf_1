import akshare as ak
import pandas as pd
# 获取上证指数的日内分钟级数据（比如获取最近的几分钟数据）
stock_minute_data = ak.stock_zh_a_minute(symbol="sz159919", period="1", adjust="qfq")

# 输出数据
print(stock_minute_data)
file_path = "stock_data_sz159919_minute.csv"
stock_minute_data.to_csv(file_path, index=False)
print(f"数据已保存到: {file_path}")