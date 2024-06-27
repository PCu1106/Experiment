import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import argparse
import os
import csv

class DistributionComparator:
    def __init__(self, file1, file2, bins):
        self.data1 = pd.read_csv(file1)
        self.data2 = pd.read_csv(file2)
        self.rssi1 = self.data1["RSSI"].values
        self.rssi2 = self.data2["RSSI"].values
        self.bins = bins
        parts = file1.split('\\')
        self.label1 = parts[3]+' '+parts[4]
        parts = file2.split('\\')
        self.label2 = parts[3]+' '+parts[4]


    def plot_histograms(self, fig_name):
        self.n1, self.bins1, _ = plt.hist(self.rssi1, bins=self.bins, alpha=0.5, label=self.label1)
        self.n2, self.bins2, _ = plt.hist(self.rssi2, bins=self.bins, alpha=0.5, label=self.label2)
        plt.legend(loc='upper right')
        plt.xlabel('RSSI')
        plt.ylabel('Frequency')
        plt.title('RSSI Histogram')
        plt.savefig(fig_name)
        plt.clf()
        # plt.show()

    def save_histo_csv(self, name=""):
        # 合併 bin 邊界，確保它們一致
        bins = self.bins1 if len(self.bins1) > len(self.bins2) else self.bins2
        with open(f'{name}_histogram_data.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Bin Start', 'Bin End', 'Count Data 1', 'Count Data 2'])
            for i in range(len(self.n1)):
                count2 = self.n2[i] if i < len(self.n2) else 0  # 確保兩個數據集的 bin 數量一致
                csvwriter.writerow([bins[i], bins[i+1], self.n1[i], count2])

    def compare_histograms(self):
        hist1, _ = np.histogram(self.rssi1, bins=self.bins, range=(-100, 0))
        hist2, _ = np.histogram(self.rssi2, bins=self.bins, range=(-100, 0))
        
        hist1 = hist1.astype(np.float32)  # 转换为32位浮点数
        hist2 = hist2.astype(np.float32)  # 转换为32位浮点数

        hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return hist_similarity
    
    def plot_avg_rssi(self, fig_name, skip=[]):
        ap_values1 = {}  # 存储每个AP的RSSI平均值
        ap_values2 = {}
        mae = 0
        for ap_id in range(1, 8):  # 7个AP，根据需要调整范围
            if ap_id in skip:
                continue
            ap_data1 = self.data1[self.data1['UUID'].str.endswith(str(ap_id))]
            ap_data2 = self.data2[self.data2['UUID'].str.endswith(str(ap_id))]
            
            avg_rssi1 = ap_data1['RSSI'].mean() if not ap_data1.empty else 0
            avg_rssi2 = ap_data2['RSSI'].mean() if not ap_data2.empty else 0
            
            ap_values1[f'AP{ap_id}'] = avg_rssi1
            ap_values2[f'AP{ap_id}'] = avg_rssi2
            mae += abs(avg_rssi1 - avg_rssi2)
        
        # 绘制每个AP的平均值
        plt.bar(ap_values1.keys(), ap_values1.values(), alpha=0.5, label=self.label1)
        plt.bar(ap_values2.keys(), ap_values2.values(), alpha=0.5, label=self.label2)
        plt.legend(loc='upper right')
        plt.xlabel('AP')
        plt.ylabel('Average RSSI')
        plt.title('Average RSSI Comparison')
        plt.savefig(fig_name)
        plt.clf()

        return mae/(7-len(skip))
    
    def print_analysis(self):
        print()
        print(f'mean1: {np.mean(self.rssi1)}')
        print(f'std1: {np.std(self.rssi1)}')
        print(f'mean2: {np.mean(self.rssi2)}')
        print(f'std2: {np.std(self.rssi2)}')
    
    

if __name__ == '__main__':
    if not os.path.exists('result'):
        os.makedirs('result')
    os.chdir('result')

    parser = argparse.ArgumentParser(description='DNN Indoor Localization')

    # 添加参数选项
    parser.add_argument('--histogram', action='store_true', help='compare histograms')
    parser.add_argument('--APmae', action='store_true', help='compare APmae')

    # 解析命令行参数
    args = parser.parse_args()

    if not args.histogram and not args.APmae:
        print('please enter argument --histogram or --APmae')
        sys.exit(0)

    # all pairs of same condition
    compare_file_pair1 = [
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\original2 11011125\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\original2 11011125\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\shading objects1 11011155\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\shading objects2 11011200\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\shading objects1 11011155\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\shading objects2 11011200\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\floor1 11011217\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\floor2 11011222\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\floor1 11011217\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\floor2 11011222\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\people1 11011414\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\people2 11011418\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\people1 11011414\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\people2 11011418\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\night1 11052050\\sharp4025\\20231105_data_BLE.csv", "D:\\Experiment\\data_change_env\\night2 11052055\\sharp4025\\20231105_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\night1 11052050\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\night2 11052055\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\AP34broken1 11052104\\sharp4025\\20231105_data_BLE.csv", "D:\\Experiment\\data_change_env\\AP34broken2 11052108\\sharp4025\\20231105_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\AP34broken1 11052104\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\AP34broken2 11052108\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\AP67broken_repositioned1 11052118\\sharp4025\\20231105_data_BLE.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned2 11052122\\sharp4025\\20231105_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\AP67broken_repositioned1 11052118\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned2 11052122\\U11\\BLE_data.csv")
        ]
    
    # all pairs of different condition
    compare_file_pair2 = [
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\original1 11011120\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\shading objects1 11011155\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\people1 11011414\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\floor1 11011217\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\0318\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\0318\\A51\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\night1 11052050\\sharp4025\\20231105_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\AP34broken1 11052104\\sharp4025\\20231105_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned1 11052118\\sharp4025\\20231105_data_BLE.csv"),
    ]

    # all pair of changing AP position
    compare_file_pair3 = [
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\floor1 11011217\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\floor2 11011222\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\floor1 11011217\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\floor2 11011222\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\floor1 11011217\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\floor2 11011222\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\floor1 11011217\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\floor2 11011222\\U11\\BLE_data.csv")
    ]

    # all pair of shading object
    compare_file_pair4 = [
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\shading objects1 11011155\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\shading objects2 11011200\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\shading objects1 11011155\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\shading objects2 11011200\\sharp4025\\20231031_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\shading objects1 11011155\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\shading objects2 11011200\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\shading objects1 11011155\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\shading objects2 11011200\\U11\\BLE_data.csv")
    ]

    # all pair of AP6, 7 broken & repositioned
    compare_file_pair5 = [
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned1 11052118\\sharp4025\\20231105_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned2 11052122\\sharp4025\\20231105_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned1 11052118\\sharp4025\\20231105_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\sharp4025\\20231031_data_BLE.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned2 11052122\\sharp4025\\20231105_data_BLE.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned1 11052118\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original1 11011120\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned2 11052122\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned1 11052118\\U11\\BLE_data.csv"),
        ("D:\\Experiment\\data_change_env\\original2 11011125\\U11\\BLE_data.csv", "D:\\Experiment\\data_change_env\\AP67broken_repositioned2 11052122\\U11\\BLE_data.csv")
    ]

    compare_file_pair6 = [
        ("D:\\Experiment\\data\\231116\\GalaxyA51\\BLE_data.csv", "D:\\Experiment\\data\\220318\\GalaxyA51\\BLE_data.csv"),
        ("D:\\Experiment\\data\\231116\\GalaxyA51\\BLE_data.csv", "D:\\Experiment\\data\\231117\\GalaxyA51\\BLE_data.csv")
    ]
    
    bins = 60

    print("the same environments")
    similarity_list = []
    mae_list = []
    for i, (file1, file2) in enumerate(compare_file_pair1):
        comparator = DistributionComparator(file1, file2, bins)
        if args.histogram:
            comparator.plot_histograms(f"same_env_{i}")
            similarity = comparator.compare_histograms()
            similarity_list.append(similarity)
            print(f"Histogram Similarity: {similarity}")
        if args.APmae:
            mae = comparator.plot_avg_rssi(f'AP_avg_rssi same_env_{i}')
            mae_list.append(mae)
            print(f"AP MAE: {mae}")
    print(f"Average: {np.array(similarity_list).mean()}")
    print(f"Average: {np.array(mae_list).mean()}")

    env_list = ["change device", "shading objects", "crowded", "change AP position", "after 19 months", "night", "AP34broken", "AP67broken_repositioned"]

    for i, (file1, file2) in enumerate(compare_file_pair2):
        comparator = DistributionComparator(file1, file2, bins)
        print(env_list[i])
        if args.histogram:
            comparator.plot_histograms(env_list[i])
            similarity = comparator.compare_histograms()
            print(f"Histogram Similarity: {similarity}")
        if args.APmae:
            if env_list[i] == "AP34broken":
                mae = comparator.plot_avg_rssi(f'AP_avg_rssi {env_list[i]}', skip = [3, 4])
            elif env_list[i] == "AP67broken_repositioned":
                mae = comparator.plot_avg_rssi(f'AP_avg_rssi {env_list[i]}', skip = [6, 7])
            else:
                mae = comparator.plot_avg_rssi(f'AP_avg_rssi {env_list[i]}')
            print(f"AP MAE: {mae}")

    for i, (file1, file2) in enumerate(compare_file_pair3):
        comparator = DistributionComparator(file1, file2, bins)
        print(f"change AP position {i}", end=' ')
        if args.histogram:
            comparator.plot_histograms(f"change AP position {i}")
            comparator.save_histo_csv(f"change_AP_position_{i}")
            comparator.print_analysis()
            similarity = comparator.compare_histograms()
            print(f"Histogram Similarity: {similarity}")
        if args.APmae:
            mae = comparator.plot_avg_rssi(f'AP_avg_rssi change AP position {i}')
            print(f"AP MAE: {mae}")

    for i, (file1, file2) in enumerate(compare_file_pair4):
        comparator = DistributionComparator(file1, file2, bins)
        print(f"shading object {i}", end=' ')
        if args.histogram:
            comparator.plot_histograms(f"shading object {i}")
            similarity = comparator.compare_histograms()
            print(f"Histogram Similarity: {similarity}")
        if args.APmae:
            mae = comparator.plot_avg_rssi(f'AP_avg_rssi shading object {i}')
            print(f"AP MAE: {mae}")

    for i, (file1, file2) in enumerate(compare_file_pair5):
        comparator = DistributionComparator(file1, file2, bins)
        print(f"AP6, 7 broken & repositioned {i}", end=' ')
        if args.histogram:
            comparator.plot_histograms(f"AP6, 7 broken & repositioned {i}")
            similarity = comparator.compare_histograms()
            print(f"Histogram Similarity: {similarity}")
        if args.APmae:
            mae = comparator.plot_avg_rssi(f'AP_avg_rssi AP6, 7 broken & repositioned {i}', skip = [6, 7])
            print(f"AP MAE: {mae}")

    for i, (file1, file2) in enumerate(compare_file_pair6):
        comparator = DistributionComparator(file1, file2, bins)
        print(f"target domain{i}", end=' ')
        if args.histogram:
            comparator.plot_histograms(f"target domain{i}")
            similarity = comparator.compare_histograms()
            print(f"Histogram Similarity: {similarity}")
        if args.APmae:
            mae = comparator.plot_avg_rssi(f"AP_avg_rssi target domain{i}")
            print(f"AP MAE: {mae}")