import csv
import pandas as pd
import numpy as np

with open('D:\\Experiment\\data\\UM_DSI_DB_v1.0.0_lite\\data\\site_surveys\\2019-06-11\\rssis.csv', 'r') as file:
    lines = file.readlines()

data = []
for line in lines:
    ap_rssi_pairs = [pair.split(':') for pair in line.strip().split(',')]
    row_data = {int(ap): int(rssi) for ap, rssi in ap_rssi_pairs}
    data.append(row_data)

# 將資料轉換為 DataFrame
df = pd.DataFrame(data)

# 將缺失值補為 -100
df = df.fillna(-100)

# 將處理後的資料存為 csv 檔案
df.to_csv('data0611.csv', index=False)
header_order = df.columns

# 讀取 2019-12-11 的 rssis.csv 文件
with open('D:\\Experiment\\data\\UM_DSI_DB_v1.0.0_lite\\data\\site_surveys\\2019-12-11\\rssis.csv', 'r') as file:
    lines = file.readlines()

data = []
for line in lines:
    ap_rssi_pairs = [pair.split(':') for pair in line.strip().split(',')]
    row_data = {int(ap): int(rssi) for ap, rssi in ap_rssi_pairs if int(ap) in header_order}
    # 補充缺失的 AP，值為 -100
    missing_ap = set(header_order) - set(row_data.keys())
    for ap in missing_ap:
        row_data[ap] = -100
    data.append(row_data)

# 將資料轉換為 DataFrame
df_new = pd.DataFrame(data)

# 將 header 順序調整為與 data0611.csv 相同
df_new = df_new[header_order]

# 將處理後的資料存為 csv 檔案
df_new.to_csv('data1211.csv', index=False)