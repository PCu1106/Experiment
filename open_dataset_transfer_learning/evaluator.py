import pandas as pd
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

if __name__ == '__main__':
    # 讀取結果
    for domain in ['0611', '1211']:
        results = pd.read_csv(f'DNN/predictions/{domain}_results.csv')

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
        print(f'DNN tensorflow {domain} MDE: {mean_distance_error}')

    # 讀取結果
    for domain in ['0611', '1211']:
        results = pd.read_csv(f'DANN_pytorch/DANN/1_0_0.9/predictions/{domain}_results.csv')

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
        print(f'DNN pytorch {domain} MDE: {mean_distance_error}')

    # 讀取結果
    for domain in ['0611', '1211']:
        results = pd.read_csv(f'DANN/DANN_0.9/predictions/{domain}_results.csv')

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
        print(f'DANN tensorflow {domain} MDE: {mean_distance_error}')


    # 讀取結果
    for domain in ['0611', '1211']:
        results = pd.read_csv(f'DANN_pytorch/DANN/1_1_0.9/predictions/{domain}_results.csv')

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
        print(f'DANN pytorch {domain} MDE: {mean_distance_error}')

    # 讀取結果
    for domain in ['0611', '1211']:
        results = pd.read_csv(f'DANN_CORR/0.1_10_0.9/predictions/{domain}_results.csv')

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
        print(f'DANN_CORR {domain} MDE: {mean_distance_error}')

    # 讀取結果
    for domain in ['0611', '1211']:
        results = pd.read_csv(f'DANN_pytorch/unlabeled/1_0_0.9/predictions/{domain}_results.csv')

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
        print(f'unlabeled DNN pytorch {domain} MDE: {mean_distance_error}')

    # 讀取結果
    for domain in ['0611', '1211']:
        results = pd.read_csv(f'DANN_pytorch/unlabeled/1_1_0.9/predictions/{domain}_results.csv')

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
        print(f'unlabeled DANN0.9 pytorch {domain} MDE: {mean_distance_error}')

            # 讀取結果
    for domain in ['0611', '1211']:
        results = pd.read_csv(f'DANN_pytorch/unlabeled/1_1_0.0/predictions/{domain}_results.csv')

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
        print(f'unlabeled DANN0.0 pytorch {domain} MDE: {mean_distance_error}')

    # 讀取結果
    for domain in ['0611', '1211']:
        results = pd.read_csv(f'DANN_CORR/unlabeled/0.1_10_0.0/predictions/{domain}_results.csv')

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
        print(f'unlabeled DANN_CORR {domain} MDE: {mean_distance_error}')