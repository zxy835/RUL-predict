import numpy as np
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 加载数据
df1 = np.load('pred.npy')
df2 = np.load('true.npy')

# 展平为 1D
y_pred = df1.squeeze()
y_true = df2.squeeze()

# 画图
plt.plot(y_pred, label='Predicted')
plt.plot(y_true, label='True')
plt.legend()
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Predicted vs True (1D Flattened)")
plt.show()

r2 = r2_score(y_true, y_pred)

print("R2:", r2)
