# -*- coding: utf-8 -*-
"""
ETF多周期量化策略系统
作者：复旦大学数学系
"""

# region 环境配置
import numpy as np
import pandas as pd
import akshare as ak
import talib
import backtrader as bt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pyfolio as pf
import warnings
import tushare as ts
warnings.filterwarnings('ignore')

# 初始化配置
pd.set_option('display.max_columns', None)
#bt.blines().setdefaults(style='candlestick')  # 设置K线模式


# endregion

# region 数据获取模块
class DataFetcher:
    @staticmethod
    def fetch_etf_data(symbol: str, freq: str):
        """
        获取ETF历史数据
        :param symbol: 标的代码（510300/518880）
        :param freq: 周期（daily/5min）
        :return: 增强后的DataFrame
        """
        # 设置Tushare token
        ts.set_token('f7814886dad9ca0aed700fe320699750de5306e9aafb337bcc530adc')  # 你需要在Tushare官网申请一个token
        pro = ts.pro_api('f7814886dad9ca0aed700fe320699750de5306e9aafb337bcc530adc')

        # 获取日线数据
        if freq == 'daily':
            if symbol == '510300':
                df = pro.fund_nav(ts_code='510300.SH', start_date='20180101', end_date='20231231')
            else:
                df = pro.fund_nav(ts_code=symbol, start_date='20230101', end_date='20231231')

        # 转换为标准OHLC格式
        df = df.rename(columns={
            'trade_date': 'date', 'open': 'open', 'high': 'high',
            'low': 'low', 'close': 'close', 'vol': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        return DataProcessor.enrich_data(df)
# endregion

# region 数据处理模块
class DataProcessor:
    @staticmethod
    def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        数据增强处理
        """
        # 基础特征
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(20).std() * np.sqrt(252)

        # 技术指标
        df['EMA5'] = talib.EMA(df['close'], timeperiod=5)
        df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
        df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # 滞后特征
        for lag in [1, 3, 5]:
            df[f'return_lag{lag}'] = df['log_return'].shift(lag)

        return df.dropna()


# endregion

# region 策略实现模块
class EnhancedMACDStrategy(bt.Strategy):
    params = (
        ('fast', 12),  # 快速EMA周期
        ('slow', 26),  # 慢速EMA周期
        ('signal', 9),  # 信号线周期
        ('atr_mult', 2),  # ATR乘数
        ('risk_pct', 0.01)  # 风险敞口比例
    )

    def __init__(self):
        # 指标计算
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast,
            period_me2=self.p.slow,
            period_signal=self.p.signal
        )
        self.atr = bt.indicators.ATR(
            self.data.high,
            self.data.low,
            self.data.close,
            period=14
        )

    def next(self):
        if self.position.size == 0:  # 无持仓时
            # 金叉信号
            if self.macd.macd[0] > self.macd.signal[0] and \
                    self.macd.macd[-1] < self.macd.signal[-1]:
                # 动态仓位计算
                risk_capital = self.broker.getvalue() * self.p.risk_pct
                size = risk_capital / (self.atr[0] * self.p.atr_mult)
                self.buy(size=size)  # 做多

        elif self.position.size > 0:  # 持有多头时
            # 死叉信号
            if self.macd.macd[0] < self.macd.signal[0] and \
                    self.macd.macd[-1] > self.macd.signal[-1]:
                self.close()  # 平仓


# endregion

# region 回测引擎模块
class BacktestEngine:
    def __init__(self, symbol, start, end):
        self.cerebro = bt.Cerebro()
        self.symbol = symbol
        self.start = start
        self.end = end

        # 添加数据
        data = DataFetcher.fetch_etf_data(symbol, 'daily')
        data_feed = bt.feeds.PandasData(dataname=data)
        self.cerebro.adddata(data_feed)

        # 配置引擎
        self._configure_engine()

    def _configure_(self):
    # 添加策略
        self.cerebro.addstrategy(EnhancedMACDStrategy)

    # 设置初始资金
        self.cerebro.broker.setcash(100000.0)

    # 交易费用
        comm_info = bt.commissions.CommInfo_Stocks(
            commission = 0.0003,  # 万三佣金
            stamp_duty = 0.001,  # 千一印花税
            commission_short = 0.0003
        )
        self.cerebro.broker.addcommissioninfo(comm_info)

# 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')


def run_backtest(self):
    return self.cerebro.run()


# endregion

# region 可视化模块
class Visualizer:
    @staticmethod
    def plot_strategy(df, results):
        # 创建子图
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02,
                            row_heights=[0.6, 0.2, 0.2])

        # K线主图
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Price'), row=1, col=1)

        # MACD指标
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
                                 line=dict(color='blue', width=1),
                                 name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'],
                                 line=dict(color='orange', width=1),
                                 name='Signal'), row=2, col=1)

        # 交易信号标注
        transactions = [t for t in results[0].transactions if t.status == 'closed']
        buy_dates = [t.datetime.date() for t in transactions if t.direction == 'long']
        sell_dates = [t.datetime.date() for t in transactions if t.direction == 'short']

        fig.add_trace(go.Scatter(x=buy_dates, y=[df.loc[d, 'low'] * 0.95 for d in buy_dates],
                                 mode='markers',
                                 marker=dict(symbol='triangle-up', size=10, color='green'),
                                 name='Buy Signal'), row=1, col=1)

        # 绩效分析
        returns = results[0].analyzers.pyfolio.get_analysis().returns
        pf.create_returns_tear_sheet(returns, benchmark_rets=df['log_return'])

        return fig.show()


# endregion

# region 主程序
if __name__ == '__main__':
    # 初始化回测引擎
    engine = BacktestEngine(symbol='510300', start='2018-01-01', end='2023-12-31')

    # 执行回测
    results = engine.run_backtest()

    # 获取回测数据
    strat = results[0]
    portfolio_stats = strat.analyzers.getbyname('pyfolio').get_analysis()

    # 可视化展示
    df = DataFetcher.fetch_etf_data('510300', 'daily')
    Visualizer.plot_strategy(df, results)

    # 打印关键指标
    print(f"最终资产价值: {strat.broker.getvalue():.2f}元")
    print(f"最大回撤: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2%}")
    print(f"年化收益率: {pf.timeseries.annual_return(portfolio_stats.returns):.2%}")
# endregion