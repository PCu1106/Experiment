'''
python .\RandomForest.py \
    --training_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
    --model_path 231116.pkl 

python .\RandomForest.py \
    --testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                        D:\Experiment\data\220318\GalaxyA51\routes \
                        D:\Experiment\data\231117\GalaxyA51\routes \
    --model_path 231116.pkl 
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import argparse
from walk_definitions import walk_class

class RandomForest:
    def __init__(self, max_depth=2):
        if not os.path.exists(f'RF_{max_depth}'):
            os.makedirs(f'RF_{max_depth}')
        os.chdir(f'RF_{max_depth}')
        self.max_depth = max_depth
        self.model = None
        

    def load_data(self, training_data_path):
        data = pd.read_csv(training_data_path)
        self.X = data.iloc[:, 1:]
        self.y = data['label']
        
    def train_model(self, model_path):
        self.model = RandomForestClassifier(max_depth=self.max_depth, random_state=0)
        self.model.fit(self.X, self.y)
        joblib.dump(self.model, model_path)

    def predict(self, input_data):
        # Assuming input_data is a list of RSSI values for Beacon_1 to Beacon_7
        # input_data = [input_data]  # Scikit-Learn's predict method expects a 2D array
        return self.model.predict(input_data)
    
    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def generate_predictions(self, model_path):
        self.load_model(model_path)
        prediction_results = {
            'label': [],
            'pred': []
        }
        # 進行預測
        predicted_labels = self.predict(self.X)
        # 將預測結果保存到 prediction_results 中
        prediction_results['label'].extend(self.y.tolist())
        prediction_results['pred'].extend(predicted_labels.tolist())
        return pd.DataFrame(prediction_results)


if __name__ == '__main__':

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='RandomForest Indoor Localization')

    # 添加参数选项
    parser.add_argument('--training_data', type=str, help='csv file of the training data')
    parser.add_argument('--testing_data_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='path of .h5 file of model')

    # 解析命令行参数
    args = parser.parse_args()

    model_path = args.model_path
    # 在根据参数来执行相应的操作
    for i in range(10, 20):
        rf_model = RandomForest(max_depth=i)
        if args.training_data:
            rf_model.load_data(args.training_data)
            rf_model.train_model(model_path)
        elif args.testing_data_list:
            testing_data_path_list = args.testing_data_list
            for testing_data_path in testing_data_path_list:
                for walk_str, walk_list in walk_class:
                    prediction_results = pd.DataFrame()
                    for walk in walk_list:
                        # 加載數據
                        rf_model.load_data(f"{testing_data_path}\\{walk}.csv")
                        results = rf_model.generate_predictions(model_path)
                        prediction_results = pd.concat([prediction_results, results], ignore_index=True)
                    split_path = testing_data_path.split('\\')
                    predictions_dir = f'predictions/{split_path[3]}'
                    os.makedirs(predictions_dir, exist_ok=True)
                    prediction_results.to_csv(os.path.join(predictions_dir, f'{walk_str}_predictions.csv'), index=False)
        else:
            print('Please specify --training_data or --test option.')

        os.chdir('..')
