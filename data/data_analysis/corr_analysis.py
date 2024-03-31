import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyzing_dataset(datasets, sample_size):
    sample_size = sample_size
    for key in datasets:
        file_list, feature_selection = datasets[key]
        hist_list = []
        for file_path in file_list:
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
        correlations = []
        for target_hist in hist_list:
            correlation = cv2.compareHist(hist_list[0], target_hist, cv2.HISTCMP_CORREL)
            correlations.append(correlation)
        print(correlations)
        # 绘制折线图
        plt.plot(correlations, label=key)
        # Adding dataset index numbers
        for i, correlation in enumerate(correlations):
            plt.text(i, correlation, f"{correlation:.2f}", ha='right')
        



file_list1 = ['../231116/GalaxyA51/wireless_training.csv', '../231218/GalaxyA51/wireless_training.csv', '../240117_troy/GalaxyA51/wireless_training.csv', '../240217_troy/GalaxyA51/wireless_training.csv', '../240319/GalaxyA51/wireless_training.csv']
feature_selection1 = ['Beacon_1', 'Beacon_2', 'Beacon_3', 'Beacon_4', 'Beacon_5', 'Beacon_6', 'Beacon_7']
file_list2 = ['../231116/GalaxyA51/wireless_training.csv', '../231218/GalaxyA51/wireless_training.csv', '../240117/GalaxyA51/wireless_training.csv', '../240217/GalaxyA51/wireless_training.csv', '../240319/GalaxyA51/wireless_training.csv']
feature_selection2 = ['Beacon_5', 'Beacon_6', 'Beacon_7']
dir_list3 = ['UM_DSI_DB_v1.0.0_lite/data/tony_data/2019-06-11', 'UM_DSI_DB_v1.0.0_lite/data/tony_data/2019-10-09', 'UM_DSI_DB_v1.0.0_lite/data/tony_data/2020-02-19']
feature_selection3 = [] # use all feature
datasets1 = {'Time variation': (file_list1, feature_selection1), 'Spatial variaton': (file_list1, feature_selection2)}
folder_path = 'UM_DSI_DB'
file_names = os.listdir(folder_path)
file_names_with_prefix = ['UM_DSI_DB/' + file_name for file_name in file_names]
datasets2 = {'UM_DSI_DB': (file_names_with_prefix, [])}
dates_only = [file_name[-14:-4] for file_name in file_names_with_prefix]

folder_path2 = 'UM_DSI_DB_reversal/'
file_names2 = os.listdir(folder_path2)
file_names2.reverse()
file_names_with_prefix2 = [folder_path2 + file_name2 for file_name2 in file_names2]
datasets3 = {'UM_DSI_DB_reversal': (file_names_with_prefix2, [])}
dates_only2 = [file_name2[-14:-4] for file_name2 in file_names_with_prefix2]

print(file_names_with_prefix)
print(dates_only)
# 定义每个样本的数量
analyzing_dataset(datasets1, 500)
# 保存折线图到文件
plt.title(f"The correlation with the first day")
plt.xlabel("Dataset Index")
plt.ylabel("correlation")
plt.xticks(range(5), ['2023-11-16', '2023-12-18', '2024-01-17', '2024-02-17', '2024-03-19'])
plt.legend()
plt.grid(True)
plt.savefig(f"correlations_plot.png")
plt.clf()


analyzing_dataset(datasets2, 500)
plt.title(f"The correlation with the first day")
plt.xlabel("Dataset Index")
plt.ylabel("correlation")
plt.xticks(range(len(dates_only)), dates_only, rotation = 45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"correlations_UM_DSI.png")
plt.clf()

analyzing_dataset(datasets3, 500)
plt.title(f"The correlation with the first day")
plt.xlabel("Dataset Index")
plt.ylabel("correlation")
plt.xticks(range(len(dates_only2)), dates_only2, rotation = 45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"correlations_UM_DSI_reversal.png")
plt.clf()