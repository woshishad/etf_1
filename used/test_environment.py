import numpy as np
import talib
import platform
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices('Metal')) 
print(f"✅ 系统架构: {platform.machine()}")
print(f"✅ NumPy版本: {np.__version__}")

# 正确用法：将输入转换为NumPy数组
close_prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

# 检查TA-Lib安装
try:
    macd, signal, hist = talib.MACD(close_prices)
    print(f"✅ TA-Lib测试成功:\nMACD={macd}\nSignal={signal}")
except Exception as e:
    print(f"❌ TA-Lib错误: {str(e)}")

# 补充其他指标测试
try:
    print(f"✅ SMA测试: {talib.SMA(close_prices, timeperiod=3)}")
    print(f"✅ RSI测试: {talib.RSI(close_prices)}")
except Exception as e:
    print(f"❌ 其他指标错误: {str(e)}")