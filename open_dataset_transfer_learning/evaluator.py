import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes

label_to_coordinate = {1: (-53.56836, 5.83747), 2: (-50.051947, 5.855995), 3: (-46.452556, 5.869534), 
                       4: (-42.853167, 5.883073), 5: (-44.659589, 7.011051), 6: (-44.751032, 11.879306), 
                       7: (-40.626278, 11.865147), 8: (-37.313205, 14.650224), 9: (-40.672748, 7.050528), 
                       10: (-39.253777, 5.896612), 11: (-35.654387, 5.91015), 12: (-32.054999, 5.923687), 
                       13: (-29.658016, 7.136601), 14: (-29.715037, 10.176074), 15: (-28.455609, 5.937224), 
                       16: (-24.856221, 5.950761), 17: (-21.256833, 5.964297), 18: (-21.06986, 12.146254), 
                       19: (-17.657445, 5.977833), 20: (-14.058057, 5.991368), 21: (-14.001059, 12.211117), 
                       22: (-10.458671, 6.004903), 23: (-6.859283, 6.018437), 24: (-6.616741, 8.258015), 
                       25: (-3.259896, 6.031971), 26: (0.33949, 6.045505), 27: (0.297446, 12.26954), 
                       28: (3.938876, 6.059038), 29: (7.538262, 6.07257), 30: (7.525253, 12.321256), 
                       31: (11.137647, 6.086102), 32: (14.737032, 6.099633), 33: (14.705246, 2.374095), 
                       34: (14.717918, 12.321068), 35: (18.336417, 6.113164), 36: (21.935801, 6.126695), 
                       37: (21.899795, 12.339099), 38: (21.921602, 2.423358), 39: (36.238672, 6.108184), 
                       40: (32.733952, 6.167284), 41: (31.779903, 2.442016), 42: (29.134569, 6.153754), 
                       43: (25.535185, 6.140225), 44: (38.088066, 7.394376), 45: (38.040971, 10.951591), 
                       46: (37.993873, 14.508804), 47: (29.037591, 12.318132), 48: (44.93136, 6.314889), 
                       49: (44.816113, 13.54513)}

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
    domain_name = {"source": '190611', "target":'200219'}
    domain_dir = f'{domain_name["source"]}_{domain_name["target"]}'
    dir_list = [f'DANN_pytorch/{domain_dir}/1_0_0.9', f'DANN_pytorch/{domain_dir}/1_1_0.9', f'DANN_AE/{domain_dir}/1_2_2_0.9', f'DANN_1DCAE/{domain_dir}/0.1_0.1_10_0.9', 
                f'DANN_CORR/{domain_dir}/0.1_10_0.9', f'DANN_CORR_AE/{domain_dir}/0.1_2_2_0.9', f'AdapLoc/{domain_dir}/1_0.01_0.9', f'DANN_baseline/{domain_dir}/0.1_0.1_10_0.9']
    model_name_list = ['DNN', 'DANN', 'DANN_AE', 'DANN_1DCAE', 
                       'DANN_CORR', 'DANN_CORR_AE', 'AdapLoc', 'K. Long et al.']
    
    mdes = count_mdes(dir_list, model_name_list, domain_name)
    plot_bar(model_name_list, mdes, 'MDE for Different Models', domain_name)

    unlabeled_dir_list = [f'DANN_pytorch/{domain_dir}/unlabeled/1_0_0.0', f'DANN_pytorch/{domain_dir}/unlabeled/1_1_0.0', f'DANN_AE/{domain_dir}/unlabeled/1_2_2_0.0', 
                          f'DANN_1DCAE/{domain_dir}/unlabeled/0.1_0.1_10_0.0', f'DANN_CORR/{domain_dir}/unlabeled/0.1_10_0.0', f'DANN_CORR_AE/{domain_dir}/unlabeled/0.1_2_2_0.0', 
                          f'AdapLoc/{domain_dir}/unlabeled/1_0.01_0.0', f'DANN_baseline/{domain_dir}/unlabeled/0.1_0.1_10_0.0']# 'DANN_1DCAE/unlabeled/0.1_0.1_10_0.0', 
    unlabeled_model_name_list = ['DNN', 'DANN', 'DANN_AE', 'DANN_1DCAE', 
                                 'DANN_CORR', 'DANN_CORR_AE', 
                                 'AdapLoc', 'K. Long et al.']
    
    unabeled_mdes = count_mdes(unlabeled_dir_list, unlabeled_model_name_list, domain_name)
    plot_bar(unlabeled_model_name_list, unabeled_mdes, 'MDE for Different Models with Unlabeled data', domain_name)
