import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes
import pickle

with open('D:\Experiment\data\MTLocData\Mall\label_map.pkl', 'rb') as f:
    label_map = pickle.load(f) # coordinate to label

label_to_coordinate = {value: key for key, value in label_map.items()}

def count_mdes(dir_list, model_name_list, domain_name):

    mdes = {domain_name["source"]:[], domain_name["target"]:[]}
    for dir, model_name in zip(dir_list, model_name_list):
        # 讀取結果
        for domain in [domain_name["source"], domain_name["target"]]:
            results = pd.read_csv(f'{dir}/predictions/{domain}_results.csv')

            # 計算每個預測點的距離誤差
            errors = []
            for idx, row in results.iterrows():
                pred_label = row['pred']
                pred_coord = label_to_coordinate[pred_label]
                actual_coord = label_to_coordinate[row['label']]
                distance_error = np.linalg.norm(np.array(pred_coord) - np.array(actual_coord))
                errors.append(distance_error)

            # 計算平均距離誤差
            mean_distance_error = np.mean(errors)
            print(f'{model_name} {domain} MDE: {mean_distance_error}')
            mdes[domain].append(mean_distance_error)
    return mdes

def calculate_mde(prediction_data_path):
    errors = []
    mde = 0
    predict_file = pd.read_csv(f'{prediction_data_path}')
    for i in range(len(predict_file)):
        pred_coord = label_to_coordinate[predict_file['pred'].iloc[i]]
        actual_coord = label_to_coordinate[predict_file['label'].iloc[i]]
        de = np.linalg.norm(np.array(pred_coord) - np.array(actual_coord))
        errors.append(de)
        mde += de
    mde = mde / len(predict_file)
    return mde, errors

def plot_cdf(errors, label, color, max_error=20.0, bin_width=0.4):
    # 设置CDF图的范围和分辨率
    min_error = 0.0
    max_error = max_error
    bin_width = bin_width

    # 创建直方图
    hist, bin_edges = np.histogram(errors, bins=np.arange(min_error, max_error + bin_width, bin_width), density=True)

    # 计算CDF
    cdf = np.cumsum(hist) * bin_width

    # 绘制CDF图
    plt.plot(bin_edges[:-1], cdf, label=label, color=color)
    return cdf, bin_edges

def find_y_lim(mdes):
    all_values = []
    for key in mdes:
        all_values.extend(mdes[key])
    
    all_values.sort()
    
    max_gap = 0
    max_gap_start = None
    max_gap_end = None
    
    for i in range(len(all_values) - 1):
        gap = all_values[i+1] - all_values[i]
        if gap > max_gap:
            max_gap = gap
            max_gap_start = all_values[i]
            max_gap_end = all_values[i+1]
            
    return max_gap_start + 0.1, max_gap_end - 0.1

def plot_bar(model_name_list, mdes, title, domain_name):
    num_models = len(model_name_list)
    # 設定長條圖的寬度
    bar_width = 0.35
    index = np.arange(num_models)
    plt.figure(figsize=(12, 6))

    # 创建一个包含中断的坐标轴
    max_val = max(max(mdes[domain_name["source"]]), max(mdes[domain_name["target"]]))
    gap_start, gap_end = find_y_lim(mdes)
    bax = brokenaxes(ylims=((0, gap_start), (gap_end, max_val+1)))

    # 繪製0611error的長條圖
    bax.bar(index - bar_width/2, mdes[domain_name["source"]], bar_width, label=domain_name["source"])

    # 繪製1211error的長條圖
    bax.bar(index + bar_width/2, mdes[domain_name["target"]], bar_width, label=domain_name["target"])

    # 添加標籤、標題和圖例
    bax.set_xlabel('Model')
    bax.set_ylabel('Mean Distance Error')
    bax.set_title(title)
    bax.set_xticks(index)
    bax.set_xticklabels([' ']+model_name_list)
    bax.legend(loc='upper left')

    # 在長條圖上標註數字
    for i, v in enumerate(mdes[domain_name["source"]]):
        bax.text(i - bar_width, v + 0.01, f'{v:.2f}', color='black', va='center')

    for i, v in enumerate(mdes[domain_name["target"]]):
        bax.text(i, v + 0.01, f'{v:.2f}', color='black', va='center')

    # # 顯示圖表
    # plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.clf()


if __name__ == '__main__':
    # dir_list = ['DANN_pytorch/1_0_0.9', 'DANN_pytorch/DANN/1_1_0.9','DANN_AE/1_2_2_0.9', 
    #             'DANN_CORR/0.1_10_0.9', 'DANN_CORR_AE/0.1_2_2_0.9', 'AdapLoc/1_0.01_0.9', 'DANN_baseline/0_1_10_0.9']# , 'DANN_1DCAE/0.1_0.1_10_0.9'
    domain_name = {"source": '211120', "target":'221221'}
    domain_dir = f'{domain_name["source"]}_{domain_name["target"]}'
    dir_list = [f'DANN_pytorch/{domain_dir}/1_0_0.9', f'DANN_pytorch/{domain_dir}/1_1_0.9', f'DANN_AE/{domain_dir}/1_2_2_0.9', f'DANN_1DCAE/{domain_dir}/0.1_0.1_10_0.9', 
                f'DANN_CORR/{domain_dir}/0.1_10_0.9', f'AdapLoc/{domain_dir}/1_0.01_0.9', f'DANN_baseline/{domain_dir}/0.1_0.1_10_0.9'] # , f'DANN_CORR_AE/{domain_dir}/0.1_2_2_0.9'
    model_name_list = ['DNN', 'DANN', 'DANN_AE', 'DANN_1DCAE', 
                       'DANN_CORR', 'AdapLoc', 'K. Long et al.'] # , 'DANN_CORR_AE'
    
    mdes = count_mdes(dir_list, model_name_list, domain_name)
    plot_bar(model_name_list, mdes, 'MDE for Different Models', domain_name)

    unlabeled_dir_list = [f'DANN_pytorch/{domain_dir}/unlabeled/1_0_0.0', f'DANN_pytorch/{domain_dir}/unlabeled/1_1_0.0', f'DANN_AE/{domain_dir}/unlabeled/1_2_2_0.0', 
                          f'DANN_1DCAE/{domain_dir}/unlabeled/0.1_0.1_10_0.0', f'DANN_CORR/{domain_dir}/unlabeled/0.1_10_0.0', 
                          f'AdapLoc/{domain_dir}/unlabeled/1_0.01_0.0', f'DANN_baseline/{domain_dir}/unlabeled/0.1_0.1_10_0.0']# , f'DANN_CORR_AE/{domain_dir}/unlabeled/0.1_2_2_0.0'
    unlabeled_model_name_list = ['DNN', 'DANN', 'DANN_AE', 'DANN_1DCAE', 
                                 'DANN_CORR', 
                                 'AdapLoc', 'K. Long et al.'] # , 'DANN_CORR_AE'
    
    unabeled_mdes = count_mdes(unlabeled_dir_list, unlabeled_model_name_list, domain_name)
    plot_bar(unlabeled_model_name_list, unabeled_mdes, 'MDE for Different Models with Unlabeled data', domain_name)
