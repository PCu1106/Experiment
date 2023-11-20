'''
python .\KNN.py \
--training_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
--model_path 231116.pkl

python .\KNN.py \
--testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                    D:\Experiment\data\220318\GalaxyA51\routes \
                    D:\Experiment\data\231117\GalaxyA51\routes \
--model_path 231116.pkl
'''

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import matplotlib.pyplot as plt
import argparse

class KNN:
    def __init__(self, k=5):
        if not os.path.exists(f'KNN_{k}'):
            os.makedirs(f'KNN_{k}')
        os.chdir(f'KNN_{k}')
        self.k = k
        self.model = None
        self.scripted_walk = [
            'one_way_walk_1','one_way_walk_2','one_way_walk_3','one_way_walk_4','one_way_walk_5','one_way_walk_6','one_way_walk_7','one_way_walk_8',
            'round_trip_walk_1', 'round_trip_walk_2','round_trip_walk_3','round_trip_walk_4'
        ]       
        self.stationary = ['stationary_1']
        self.freewalk = [
            'freewalk_1','freewalk_2','freewalk_3','freewalk_4','freewalk_5','freewalk_6','freewalk_7','freewalk_8','freewalk_9'
        ]
        self.walk_class = [('scripted_walk', self.scripted_walk), ('stationary', self.stationary), ('freewalk', self.freewalk)]

    def load_data(self, training_data_path):
        data = pd.read_csv(training_data_path)
        self.X = data.iloc[:, 1:]
        self.y = data['label']
        
    def train_model(self, model_path):
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(self.X, self.y)
        joblib.dump(self.model, model_path)

    def predict(self, input_data):
        # Assuming input_data is a list of RSSI values for Beacon_1 to Beacon_7
        # input_data = [input_data]  # Scikit-Learn's predict method expects a 2D array
        return self.model.predict(input_data)
    
    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def generate_predictions(self, testing_data_path):
        for walk_str, walk_list in self.walk_class:
            prediction_results = {
                'label': [],
                'pred': []
            }
            for walk in walk_list:
                data_path = f"{testing_data_path}\\{walk}.csv"

                # 加載數據
                self.load_data(data_path)

                # 進行預測
                predicted_labels = self.predict(self.X)
                # 將預測結果保存到 prediction_results 中
                prediction_results['label'].extend(self.y.tolist())
                prediction_results['pred'].extend(predicted_labels.tolist())
            df = pd.DataFrame(prediction_results)
            
            split_path = testing_data_path.split('\\')
            predictions_dir = f'predictions/{split_path[3]}'
            os.makedirs(predictions_dir, exist_ok=True)
            df.to_csv(os.path.join(predictions_dir, f'{walk_str}_predictions.csv'), index=False)

    def test(self, testing_data_path_list, model_path):
        self.load_model(model_path)
        for testing_data_path in testing_data_path_list:
            self.generate_predictions(testing_data_path)

if __name__ == '__main__':

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='KNN Indoor Localization')

    # 添加参数选项
    parser.add_argument('--training_data', type=str, help='csv file of the training data')
    parser.add_argument('--testing_data_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='path of .h5 file of model')

    # 解析命令行参数
    args = parser.parse_args()

    model_path = args.model_path
    # 在根据参数来执行相应的操作
    for i in range(1, 10):
        knn_model = KNN(k = i)
        if args.training_data:
                knn_model.load_data(args.training_data)
                knn_model.train_model(model_path)
        elif args.testing_data_list:
            testing_data_path_list = args.testing_data_list
            knn_model.test(testing_data_path_list, model_path)
        else:
            print('Please specify --training_data or --test option.')

        os.chdir('..')
