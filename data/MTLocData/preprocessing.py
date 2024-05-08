import os
import h5py
import pandas as pd
import numpy as np
import pickle

date_list = ['2021-11-20', '2021-12-20', '2022-07-21', '2022-06-21', '2022-10-21', '2022-11-21', '2022-12-21']
# date_list = [f'week{i}' for i in range(1, 38)]

def convert_h5_to_csv(h5_file, output_folder, reference_bssids, label_map):
    # 讀取HDF5文件
    with h5py.File(h5_file, 'r') as hdf5_data:
        # 提取bssids和rssis數據集
        bssids = hdf5_data['bssids'][:]
        cdns = hdf5_data['cdns'][:]
        rssis = hdf5_data['rssis'][:]

        # 將rssis轉換為DataFrame
        rssis_df = pd.DataFrame(rssis, columns=[bssid.decode('utf-8') for bssid in bssids])

        # 對於缺失的bssid，補上缺失的col（補0）
        missing_cols = set(reference_bssids) - set(rssis_df.columns)
        for col in missing_cols:
            rssis_df[col] = 0

        # 重新排列DataFrame的col順序
        rssis_df = rssis_df[reference_bssids]

        # 將cdns轉換為label，使用label_map
        cdn_labels = [label_map[tuple(cdn)] for cdn in cdns]
        cdn_labels_df = pd.DataFrame({'label': cdn_labels})

        # 將label和rssis組合成DataFrame
        df = pd.concat([cdn_labels_df, rssis_df], axis=1)

        mean_values = df.iloc[:, 1:].mean()
        std_values = df.iloc[:, 1:].std()
        
        for ap in mean_values.index:
            if std_values[ap] == 0:
                df[ap] = 0  # 或者使用其他值代替 0
            else:
                df[ap] = (df[ap] - mean_values[ap]) / std_values[ap]

        testing_set = pd.DataFrame(columns=df.columns)
        for label in df['label'].unique():
            sample_num = int(len(df[df['label'] == label])/10)
            sample = df[df['label'] == label].sample(n=sample_num if sample_num != 0 else 1)
            testing_set = pd.concat([testing_set, sample], ignore_index=True)
            df = df.drop(sample.index)
            print(f"  Label{label} samples: {len(df[df['label'] == label])}")

        testing_set.to_csv(os.path.join(output_folder, f'wireless_testing.csv'), index=False)

        # 將DataFrame保存為CSV文件
        output_file = os.path.join(output_folder, f'wireless_training.csv')
        df.to_csv(output_file, index=False)

# 資料夾路徑
folder_path = 'Mall'
# folder_path = 'OfficeP1+P2'

# 讀取Mall_1_training.h5，獲取參考bssids和座標對應的label_map
reference_file = os.path.join(folder_path, 'Mall_1_training.h5')
# reference_file = os.path.join(folder_path, 'OfficeP1+P2_1_training.h5')
with h5py.File(reference_file, 'r') as hdf5_data:
    reference_bssids = [bssid.decode('utf-8') for bssid in hdf5_data['bssids'][:]]
    cdns = hdf5_data['cdns'][:]
    unique_cdns = np.unique(cdns, axis=0)
    label_map = {tuple(cdn): i + 1 for i, cdn in enumerate(unique_cdns)}
    label_map_file = os.path.join(folder_path, 'label_map.pkl')
    # 將label_map保存為文件
    with open(label_map_file, 'wb') as f:
        pickle.dump(label_map, f)

# 將每個.h5檔案轉換成csv檔案
for i, date in enumerate(date_list):
    training_file = os.path.join(folder_path, f'Mall_{i+1}_training.h5')
    # training_file = os.path.join(folder_path, f'OfficeP1+P2_{i+1}_training.h5')
    output_folder = os.path.join(folder_path, date)

    # 建立輸出資料夾
    os.makedirs(output_folder, exist_ok=True)

    # 轉換訓練檔案
    if os.path.exists(training_file):
        convert_h5_to_csv(training_file, output_folder, reference_bssids, label_map)

