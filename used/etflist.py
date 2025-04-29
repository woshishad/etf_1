import akshare as ak
import pandas as pd

#etf = ak.fund_etf_category_sina(symbol="ETF基金")
#etf.to_csv("sina_etf_list.csv", encoding='utf-8-sig')

etf_daily_data = ak.fund_etf_hist_sina(symbol="sz159919")
etf_daily_data['open'] = pd.to_numeric(etf_daily_data['open'], errors='coerce')
etf_daily_data['high'] = pd.to_numeric(etf_daily_data['high'], errors='coerce')
etf_daily_data['low'] = pd.to_numeric(etf_daily_data['low'], errors='coerce')
etf_daily_data['close'] = pd.to_numeric(etf_daily_data['close'], errors='coerce')
etf_daily_data['volume'] = pd.to_numeric(etf_daily_data['volume'], errors='coerce')

    # 查看是否有缺失值
print(etf_daily_data.isnull().sum())

print(etf_daily_data)