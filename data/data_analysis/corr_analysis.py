import cv2
import pandas as pd
import numpy as np

# dir_list = ['231116', '231218', '240117_troy', '240217_troy', '240319']
# feature_selection = ['Beacon_1', 'Beacon_2', 'Beacon_3', 'Beacon_4', 'Beacon_5', 'Beacon_6', 'Beacon_7']
# dir_list = ['231116/GalaxyA51', '231218/GalaxyA51', '240117/GalaxyA51', '240217/GalaxyA51', '240319/GalaxyA51']
# feature_selection = ['Beacon_5', 'Beacon_6', 'Beacon_7']
dir_list = ['UM_DSI_DB_v1.0.0_lite/data/tony_data/2019-06-11', 'UM_DSI_DB_v1.0.0_lite/data/tony_data/2019-10-09', 'UM_DSI_DB_v1.0.0_lite/data/tony_data/2020-02-19']
feature_selection = []
hist_list = []

# 定义每个样本的数量
sample_size = 900

for dir_name in dir_list:
    file_path = f"../{dir_name}/wireless_training.csv"
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取所有特征值（不包括标签列）
        if feature_selection:
            selected_features = df[feature_selection].values
        else:
            selected_features = df.iloc[:, 1:].values
        
        # 从所有特征值中随机抽取样本
        sample_indices = np.random.choice(len(selected_features), sample_size, replace=False)
        sampled_values = selected_features[sample_indices]
        sampled_values = sampled_values.flatten()

        # 获取样本的最小值和最大值
        min_val = np.min(sampled_values)
        max_val = np.max(sampled_values)

        # 计算直方图
        hist = cv2.calcHist([sampled_values.astype(np.float32)], [0], None, [100], [min_val, max_val])
        
        # 打印或存储直方图等操作
        hist_list.append(hist)
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")

# 计算直方图之间的相关性
for target_hist in hist_list:
    correlation = cv2.compareHist(hist_list[0], target_hist, cv2.HISTCMP_CORREL)
    print(correlation)