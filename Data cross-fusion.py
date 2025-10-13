import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
index = pd.read_csv('similarity.csv', header=None)
df = pd.read_csv('processed_FD001.csv.csv')
interval = 5
groups = df.groupby(df.iloc[:, 0])  # 分组

def grouped(a):

    group = groups.get_group(list(groups.groups.keys())[a])
    group = group.iloc[:, 1:]
    return group

def get_fragment(group, b):
    data_fragment = group.iloc[b:b + interval, :]
    return data_fragment

def deleteX(group1, group2, c, d):
    group1.reset_index(drop=True, inplace=True)
    group1 = group1.drop(list(range(c, c + interval)))
    group2.reset_index(drop=True, inplace=True)
    group2 = group2.drop(list(range(d, d + interval)))
    return group1, group2

def exchange_fragement(a, b, c, d):
    data1 = grouped(a)
    data2 = grouped(b)

    fragement1 = get_fragment(data1, c)
    fragement2 = get_fragment(data2, d)

    group1, group2 = deleteX(data1, data2, c, d)
    group1 = pd.concat([group1.iloc[:c], fragement2, group1.iloc[c:]], ignore_index=True)
    group2 = pd.concat([group2.iloc[:d], fragement1, group2.iloc[d:]], ignore_index=True)
    return group1, group2

for i in range(0, len(index)):
    a = index[1][i] - 1
    b = index[2][i] - 1
    c = index[3][i] - 1
    d = index[4][i] - 1
    group1, group2 = exchange_fragement(a, b, c, d)
    search_value1 = a + 1
    indices1 = df[df.iloc[:, 0] == search_value1].index
    search_value2 = b + 1
    indices2 = df[df.iloc[:, 0] == search_value2].index
    df.loc[indices1, df.columns[1:]] = group1.values
    df.loc[indices2, df.columns[1:]] = group2.values

print(df)
df.to_csv('augmentation data.csv', index=False)