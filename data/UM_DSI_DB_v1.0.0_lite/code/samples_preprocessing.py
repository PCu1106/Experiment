import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 讀取 2019-06-11 的 rssis.csv 文件
with open('D:\\Experiment\\data\\UM_DSI_DB_v1.0.0_lite\\data\\site_surveys\\2019-06-11\\rssis.csv', 'r') as file:
    lines = file.readlines()

data0611 = []
for line in lines:
    ap_rssi_pairs = [pair.split(':') for pair in line.strip().split(',')]
    row_data = {int(ap): int(rssi) for ap, rssi in ap_rssi_pairs}
    data0611.append(row_data)

# 將 2019-06-11 的資料轉換為 DataFrame
df0611 = pd.DataFrame(data0611)

# 讀取 2019-12-11 的 rssis.csv 文件
with open('D:\\Experiment\\data\\UM_DSI_DB_v1.0.0_lite\\data\\site_surveys\\2019-12-11\\rssis.csv', 'r') as file:
    lines = file.readlines()

data1211 = []
for line in lines:
    ap_rssi_pairs = [pair.split(':') for pair in line.strip().split(',')]
    row_data = {int(ap): int(rssi) for ap, rssi in ap_rssi_pairs}
    data1211.append(row_data)

# 將 2019-12-11 的資料轉換為 DataFrame
df1211 = pd.DataFrame(data1211)

# 找出兩者的 AP 交集
common_aps = set(df0611.columns).intersection(set(df1211.columns))
print(common_aps)

# 保留交集部分資料
df0611 = df0611[list(common_aps)]
df1211 = df1211[list(common_aps)]

# 使用插值處理缺失值
df0611 = df0611.interpolate(method='linear', limit_direction='both')
df1211 = df1211.interpolate(method='linear', limit_direction='both')

# 如果還有缺失值，則填補為-100
df0611 = df0611.fillna(-100)
df1211 = df1211.fillna(-100)

# 處理 label0611.csv
label_df0611 = pd.read_csv('labels0611.csv', header=0)
df0611.insert(0, 'label', label_df0611['label'])

# 處理 label1211.csv
label_df1211 = pd.read_csv('labels1211.csv', header=0)
df1211.insert(0, 'label', label_df1211['label'])

# 計算每個AP的最小值和最大值
min_values = df0611.iloc[:, 1:].min()
max_values = df0611.iloc[:, 1:].max()

# 計算每個AP的平均值和標準差
mean_values = df0611.iloc[:, 1:].mean()
std_values = df0611.iloc[:, 1:].std()

# 對每個AP的RSSI做標準化
for ap in mean_values.index:
    if std_values[ap] == 0:
        df0611[ap] = 0  # 或者使用其他值代替 0
    else:
        df0611[ap] = (df0611[ap] - mean_values[ap]) / std_values[ap]

mean_values = df1211.iloc[:, 1:].mean()
std_values = df1211.iloc[:, 1:].std()

for ap in mean_values.index:
    if std_values[ap] == 0:
        df1211[ap] = 0  # 或者使用其他值代替 0
    else:
        df1211[ap] = (df1211[ap] - mean_values[ap]) / std_values[ap]

# # 對每個AP的RSSI做min_max normalization，處理除以0的情況
# for ap in min_values.index:
#     if max_values[ap] - min_values[ap] == 0:
#         df0611[ap] = 0  # 或者使用其他值代替 0
#     else:
#         df0611[ap] = (df0611[ap] - min_values[ap]) / (max_values[ap] - min_values[ap])

# min_values = df1211.iloc[:, 1:].min()
# max_values = df1211.iloc[:, 1:].max()

# for ap in min_values.index:
#     if max_values[ap] - min_values[ap] == 0:
#         df1211[ap] = 0  # 或者使用其他值代替 0
#     else:
#         df1211[ap] = (df1211[ap] - min_values[ap]) / (max_values[ap] - min_values[ap])

# 將處理後的資料存為 csv 檔案
df0611.to_csv('data0611.csv', index=False)
df1211.to_csv('data1211.csv', index=False)

for df, name in zip([df0611, df1211], ['0611', '1211']):
    testing_set = pd.DataFrame(columns=df.columns)
    for label in df['label'].unique():
        sample = df[df['label'] == label].sample(n=1)
        testing_set = pd.concat([testing_set, sample], ignore_index=True)
        df = df.drop(sample.index)
        print(f"  Label{label} samples: {len(df[df['label'] == label])}")

    testing_set.to_csv(f'wireless_testing_{name}.csv', index=False)
    df.to_csv(f'wireless_training_{name}.csv', index=False)

