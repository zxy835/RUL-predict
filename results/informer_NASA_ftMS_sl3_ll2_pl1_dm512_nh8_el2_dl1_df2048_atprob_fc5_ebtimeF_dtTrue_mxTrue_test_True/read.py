import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

# MAPE 计算
def mape(y_pred, y_true):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

# 相关系数计算
def correlation_coefficient(y_pred, y_true):
    return np.corrcoef(y_pred, y_true)[0, 1]
# 加载数据
df1 = np.load('pred.npy')
df2 = np.load('true.npy')

# 使用squeeze方法将形状为2496*1*1的数组转换为2496
a = df1.squeeze()  # 去除多余的维度
b = df2.squeeze()  # 去除多余的维度

for i in range(len(a)):
    a[i] = a[i]*29.36370 + 110.09300
    b[i] = b[i] * 29.36370 + 110.09300

# 转换为DataFrame，便于操作
a = pd.DataFrame(a)
b = pd.DataFrame(b)
a = a[0:100]
b = b[0:100]
rmse_value = rmse(a, b)
mape_value = mape(a, b)
correlation_value = correlation_coefficient(a, b)
print(f"RMSE: {rmse_value}")
print(f"MAPE: {mape_value}%")
print(f"相关系数: {correlation_value}")
# 绘制数据
plt.plot(a, label='Predicted')
plt.plot(b, label='True')

plt.legend()  # 添加图例
plt.show()
