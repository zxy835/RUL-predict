import numpy as np
import matplotlib.pyplot as plt

# 加载数据
df1 = np.load('pred.npy')
df2 = np.load('true.npy')
df3 = np.load('real_prediction.npy')
# 绘制数据
plt.plot(df3[0], label='predicted')
plt.show()
for i in range(len(df1)):

    plt.plot(df1[i], label='Predicted')
    plt.plot(df2[i], label='True')

    plt.legend()  # 添加图例
    plt.show()
