import numpy as np
import pandas as pd
import csv
import warnings
from scipy.spatial.distance import cdist
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ====== 定义计算 MMD 的函数 ======
def mmd_dtw(X, Y):
    K_XX = np.exp(-cdist(X, X, 'sqeuclidean') / 2)
    K_XY = np.exp(-cdist(X, Y, 'sqeuclidean') / 2)
    K_YX = np.exp(-cdist(Y, X, 'sqeuclidean') / 2)
    K_YY = np.exp(-cdist(Y, Y, 'sqeuclidean') / 2)
    return K_XX.mean() + K_YY.mean() - K_XY.mean() - K_YX.mean()

# ====== 主程序部分 ======
all_mmd = []
df = pd.read_csv('processed_FD001.csv')
interval = 5
groups = df.groupby(df.iloc[:, 0])  # 按第一列分组

for k in range(0, 99):
    print('第', k + 1, '组')
    in_mmd = []

    # ---- 获取第k组数据 ----
    group1 = groups.get_group(list(groups.groups.keys())[k]).iloc[:, 1:]
    group1_segments = [group1[i: i + interval].astype(float) for i in range(0, len(group1) - interval)]

    if len(group1) % interval != 0:
        last_segment = group1[len(group1) - len(group1) % interval:].astype(float)
        last_segment = np.concatenate((group1[-interval:], last_segment), axis=0)
        group1_segments.append(last_segment)

    # ---- 与其他组比较 ----
    for m in range(k + 1, 100):
        group2 = groups.get_group(list(groups.groups.keys())[m]).iloc[:, 1:]
        group2_segments = [group2[i: i + interval].astype(float) for i in range(0, len(group2) - interval)]

        if len(group2) % interval != 0:
            last_segment = group2[len(group2) - len(group2) % interval:].astype(float)
            last_segment = np.concatenate((group2[-interval:], last_segment), axis=0)
            group2_segments.append(last_segment)

        # ---- 使用 DTW 核计算 MMD 值 ----
        print("使用 DTW 核函数计算 MMD ...")
        mmd_values = []
        start_time = time.time()

        for i, group1_matrix in enumerate(group1_segments):
            for j, group2_matrix in enumerate(group2_segments):
                mmd_value = mmd_dtw(group1_matrix, group2_matrix) * 100
                mmd_values.append((i, j, mmd_value, k + 1, m + 1))

        end_time = time.time()
        print(f"计算耗时: {end_time - start_time:.3f} 秒")

        # ---- 找到最小 MMD ----
        mmd_values.sort(key=lambda x: x[2])
        min_mmd = {}
        for mmd_value in mmd_values:
            if mmd_value[0] not in min_mmd and mmd_value[1] not in min_mmd.values():
                min_mmd[mmd_value[0]] = (mmd_value[1], mmd_value[2])
                if len(min_mmd) == 1:
                    break

        key = list(min_mmd.keys())[0]
        value = min_mmd[key]
        print(f"最小的MMD值为：G1:x{key + 1}--G2:x{value[0] + 1} == {value[1]}")

        in_mmd.append(mmd_values[0])

    in_mmd.sort(key=lambda x: x[2])
    num = len(in_mmd)

    if num >= 8:
        all_mmd.extend(in_mmd[:8])
    else:
        all_mmd.extend(in_mmd[:num])

    print(f"当前累计结果数量：{len(all_mmd)}")

# ====== 保存结果到 CSV ======
csv_file_path = 'similarity.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(all_mmd)

print(f'Data has been written to {csv_file_path}')
