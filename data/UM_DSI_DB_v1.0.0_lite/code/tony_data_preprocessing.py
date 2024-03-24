'''
This code output three dataset. The feature of all dataset depends on the APs of first dataset (source domain)
The AP will disappear through times going, in other words, there will be some features become 0 for all samples
'''
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import parser

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
df0611 = df0611.interpolate(method='linear', limit_direction='both')
df0611 = df0611.fillna(-100)
# 處理 label0611.csv
label_df0611 = pd.read_csv('labels_2019-06-11.csv', header=0)
df0611.insert(0, 'label', label_df0611['label'])
# 計算每個AP的平均值和標準差
mean_values = df0611.iloc[:, 1:].mean()
std_values = df0611.iloc[:, 1:].std()


# 對每個AP的RSSI做標準化
for ap in mean_values.index:
    if std_values[ap] == 0:
        df0611[ap] = 0  # 或者使用其他值代替 0
    else:
        df0611[ap] = (df0611[ap] - mean_values[ap]) / std_values[ap]

df0611.to_csv('data_2019-06-11.csv', index=False)


dataset_folder = Path('..')
ssv_folder = dataset_folder/'data'/'site_surveys'
site_surveys = [f.name for f in os.scandir(ssv_folder) if f.is_dir()]
site_surveys = parser.sorted_nicely(site_surveys)

for site_survey in site_surveys:
    if(site_survey == '2019-02-19' or site_survey == '2019-03-25' or site_survey == '2019-06-11'): # number of RP different, don't use them
        continue
    # 讀取 2019-10-09 的 rssis.csv 文件
    with open(f'D:\\Experiment\\data\\UM_DSI_DB_v1.0.0_lite\\data\\site_surveys\\{site_survey}\\rssis.csv', 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        ap_rssi_pairs = [pair.split(':') for pair in line.strip().split(',')]
        row_data = {int(ap): int(rssi) for ap, rssi in ap_rssi_pairs}
        data.append(row_data)

    # 將 2019-10-09 的資料轉換為 DataFrame
    df = pd.DataFrame(data)
    missing_aps = [ap for ap in df0611.columns if ap not in df.columns]
    # 将缺失的 AP 添加到 df 中，并填充缺失值为 -100
    print(f'{site_survey} miss {len(missing_aps)} aps')
    print(missing_aps)
    for ap in missing_aps:
        df[ap] = -100
        
    # 确保列的顺序和 df0611 一致
    df = df[df0611.columns]
    # 使用插值處理缺失值
    df = df.interpolate(method='linear', limit_direction='both')
    # 如果還有缺失值，則填補為-100
    df = df.fillna(-100)
    
    # 處理 label.csv
    label_df = pd.read_csv(f'labels_{site_survey}.csv', header=0)
    df['label'] = label_df['label']

    mean_values = df.iloc[:, 1:].mean()
    std_values = df.iloc[:, 1:].std()
    
    for ap in mean_values.index:
        if std_values[ap] == 0:
            df[ap] = 0  # 或者使用其他值代替 0
        else:
            df[ap] = (df[ap] - mean_values[ap]) / std_values[ap]
            
    # 將處理後的資料存為 csv 檔案
    df.to_csv(f'data_{site_survey}.csv', index=False)


# for df, name in zip([df0611, df1009, df0219], ['190611', '191009', '200219']):
#     testing_set = pd.DataFrame(columns=df.columns)
#     for label in df['label'].unique():
#         sample = df[df['label'] == label].sample(n=1)
#         testing_set = pd.concat([testing_set, sample], ignore_index=True)
#         df = df.drop(sample.index)
#         print(f"  Label{label} samples: {len(df[df['label'] == label])}")

#     testing_set.to_csv(f'wireless_testing_{name}.csv', index=False)
#     df.to_csv(f'wireless_training_{name}.csv', index=False)

