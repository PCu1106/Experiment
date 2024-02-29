import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def count_mdes(dir_list, model_name_list):
    mdes = {'0611':[], '1211':[]}
    for dir, model_name in zip(dir_list, model_name_list):
        # 讀取結果
        for domain in ['0611', '1211']:
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

def plot_bar(model_name_list, mdes, title):
    num_models = len(model_name_list)
    # 設定長條圖的寬度
    bar_width = 0.35
    index = np.arange(num_models)

    # 繪製0611error的長條圖
    plt.bar(index, mdes['0611'], bar_width, label='0611')

    # 繪製1211error的長條圖
    plt.bar(index + bar_width, mdes['1211'], bar_width, label='1211')

    # 添加標籤、標題和圖例
    plt.xlabel('Model')
    plt.ylabel('Mean Distance Error')
    plt.title(title)
    plt.xticks(index + bar_width / 2, model_name_list, rotation=45)
    plt.legend()

    # 在長條圖上標註數字
    for i, v in enumerate(mdes['0611']):
        plt.text(i - 0.1, v + 0.01, f'{v:.2f}', color='black', va='center')

    for i, v in enumerate(mdes['1211']):
        plt.text(i + bar_width - 0.1, v + 0.01, f'{v:.2f}', color='black', va='center')

    # 顯示圖表
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.clf()


if __name__ == '__main__':
    dir_list = ['DNN', 'DANN_pytorch/DANN/1_0_0.9', 'DANN/DANN_0.9', 'DANN_pytorch/DANN/1_1_0.9', 'DANN_CORR/0.1_10_0.9', 
                'DANN_1DCAE/DANN_CORR_0.9', 'DANN_baseline/0_1_10_0.9']
    model_name_list = ['DNN tensorflow', 'DNN pytoch', 'DANN tensorflow', 'DANN pytorch', 'DANN_CORR', 
                       'DANN_1DCAE', 'K. Long et al.']
    
    mdes = count_mdes(dir_list, model_name_list)
    plot_bar(model_name_list, mdes, 'MDE for Different Models')

    unlabeled_dir_list = ['DANN_pytorch/unlabeled/1_0_0.9', 'DANN_pytorch/unlabeled/1_1_0.9', 'DANN_pytorch/unlabeled/1_1_0.0', 
                          'DANN_CORR/unlabeled/0.1_10_0.0', 'DANN_1DCAE/unlabeled/0.1_0.1_10_0.0', 
                          'DANN_baseline/unlabeled/0_1_10_0.0']
    unlabeled_model_name_list = ['unlabeled DNN pytorch', 'unlabeled DANN0.9 pytorch', 'unlabeled DANN0.0 pytorch', 
                                 'unlabeled DANN_CORR', 'unlabeled DANN_1DCAE', 
                                 'unlabeled K. Long et al.']
    
    unabeled_mdes = count_mdes(unlabeled_dir_list, unlabeled_model_name_list)
    plot_bar(unlabeled_model_name_list, unabeled_mdes, 'MDE for Different Models with Unlabeled data')
