import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('train_FD001.csv',header=None) # 读取数据
df = df.drop(columns=[1,2,3,4,5,9,10,14,20,22,23]) # 删除冗余数据
#归一化
x_min = MinMaxScaler().fit_transform(df)
x_min = pd.DataFrame(x_min)

x_min.to_csv('processed_FD001.csv', index=False)
print(x_min)