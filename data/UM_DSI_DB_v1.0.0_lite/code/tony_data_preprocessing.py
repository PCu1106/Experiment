'''
This code output three dataset. The feature of all dataset depends on the APs of first dataset (source domain)
The AP will disappear through times going, in other words, there will be some features become 0 for all samples
'''
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

# 讀取 2019-10-09 的 rssis.csv 文件
with open('D:\\Experiment\\data\\UM_DSI_DB_v1.0.0_lite\\data\\site_surveys\\2019-10-09\\rssis.csv', 'r') as file:
    lines = file.readlines()

data1009 = []
for line in lines:
    ap_rssi_pairs = [pair.split(':') for pair in line.strip().split(',')]
    row_data = {int(ap): int(rssi) for ap, rssi in ap_rssi_pairs}
    data1009.append(row_data)

# 將 2019-10-09 的資料轉換為 DataFrame
df1009 = pd.DataFrame(data1009)

# 讀取 2020-02-19 的 rssis.csv 文件
with open('D:\\Experiment\\data\\UM_DSI_DB_v1.0.0_lite\\data\\site_surveys\\2020-02-19\\rssis.csv', 'r') as file:
    lines = file.readlines()

data0219 = []
for line in lines:
    ap_rssi_pairs = [pair.split(':') for pair in line.strip().split(',')]
    row_data = {int(ap): int(rssi) for ap, rssi in ap_rssi_pairs}
    data0219.append(row_data)

# 將 2020-02-19 的資料轉換為 DataFrame
df0219 = pd.DataFrame(data0219)
print(df0219.shape)
# 找出 df0611 中存在但 df1009 中不存在的 AP
missing_aps1009 = [ap for ap in df0611.columns if ap not in df1009.columns]
missing_aps0219 = [ap for ap in df0611.columns if ap not in df0219.columns]

# 将缺失的 AP 添加到 df1009 中，并填充缺失值为 -100
print(f'1009 miss {len(missing_aps1009)} aps')
print(missing_aps1009)
print(f'0219 miss {len(missing_aps0219)} aps')
print(missing_aps0219)

for ap in missing_aps1009:
    df1009[ap] = -100

for ap in missing_aps0219:
    df0219[ap] = -100

# 保留source domain資料
print(df0611.shape)

# 确保列的顺序和 df0611 一致
df1009 = df1009[df0611.columns]
df0219 = df0219[df0611.columns]
# 使用插值處理缺失值
df0611 = df0611.interpolate(method='linear', limit_direction='both')
df1009 = df1009.interpolate(method='linear', limit_direction='both')
df0219 = df0219.interpolate(method='linear', limit_direction='both')

# 如果還有缺失值，則填補為-100
df0611 = df0611.fillna(-100)
df1009 = df1009.fillna(-100)
df0219 = df0219.fillna(-100)

# 處理 label0611.csv
label_df0611 = pd.read_csv('labels190611.csv', header=0)
df0611.insert(0, 'label', label_df0611['label'])

# 處理 label1009.csv
label_df1009 = pd.read_csv('labels191009.csv', header=0)
df1009.insert(0, 'label', label_df1009['label'])

# 處理 label0219.csv
label_df0219 = pd.read_csv('labels200219.csv', header=0)
df0219.insert(0, 'label', label_df0219['label'])

# 計算每個AP的平均值和標準差
mean_values = df0611.iloc[:, 1:].mean()
std_values = df0611.iloc[:, 1:].std()

# 對每個AP的RSSI做標準化
for ap in mean_values.index:
    if std_values[ap] == 0:
        df0611[ap] = 0  # 或者使用其他值代替 0
    else:
        df0611[ap] = (df0611[ap] - mean_values[ap]) / std_values[ap]

mean_values = df1009.iloc[:, 1:].mean()
std_values = df1009.iloc[:, 1:].std()

for ap in mean_values.index:
    if std_values[ap] == 0:
        df1009[ap] = 0  # 或者使用其他值代替 0
    else:
        df1009[ap] = (df1009[ap] - mean_values[ap]) / std_values[ap]

mean_values = df0219.iloc[:, 1:].mean()
std_values = df0219.iloc[:, 1:].std()

for ap in mean_values.index:
    if std_values[ap] == 0:
        df0219[ap] = 0  # 或者使用其他值代替 0
    else:
        df0219[ap] = (df0219[ap] - mean_values[ap]) / std_values[ap]

# 將處理後的資料存為 csv 檔案
df0611.to_csv('data190611.csv', index=False)
df1009.to_csv('data191009.csv', index=False)
df0219.to_csv('data200219.csv', index=False)

for df, name in zip([df0611, df1009, df0219], ['190611', '191009', '200219']):
    testing_set = pd.DataFrame(columns=df.columns)
    for label in df['label'].unique():
        sample = df[df['label'] == label].sample(n=1)
        testing_set = pd.concat([testing_set, sample], ignore_index=True)
        df = df.drop(sample.index)
        print(f"  Label{label} samples: {len(df[df['label'] == label])}")

    testing_set.to_csv(f'wireless_testing_{name}.csv', index=False)
    df.to_csv(f'wireless_training_{name}.csv', index=False)

