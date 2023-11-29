'''
python .\AE_Integration.py \
    --training_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
    --AE_model_path 231116.h5 \
    --model KNN \
    --model_path 231116.pkl

python .\AE_Integration.py \
    --testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                    D:\Experiment\data\220318\GalaxyA51\routes \
                    D:\Experiment\data\231117\GalaxyA51\routes \
    --AE_model_path 231116.h5 \
    --model KNN \
    --model_path 231116.pkl
'''

from AutoEncoder import AutoEncoder
from KNN import KNN
from DNN import DNN
from RandomForest import RandomForest
import argparse
import os
import pandas as pd
from walk_definitions import walk_class


if __name__ == '__main__':

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Indoor Localization')

    # 添加参数选项
    parser.add_argument('--training_data', type=str, help='csv file of the training data')
    parser.add_argument('--testing_data_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--AE_model_path', type=str, default='my_model.h5', help='path of .h5 file of AE model')
    parser.add_argument('--model', type=str, help='the model contating after AE')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='path of contating model')
    parser.add_argument('--work_dir', type=str, help='create new directory to save result')
    

    # 解析命令行参数
    args = parser.parse_args()

    AE_model_path = args.AE_model_path
    model_path = args.model_path

    input_size = 7
    code_size = 4
    ae_model = AutoEncoder(input_size, code_size)
    ae_model.load_model(AE_model_path)
    encoder = ae_model.getEncoder()

    if not args.model:
        print('please specify the model contating after AE')

    if args.model == 'KNN':
        for i in range(1, 10):
            knn_model = KNN(k = i)
            if args.training_data:
                knn_model.load_data(args.training_data)
                knn_model.X = encoder.predict(knn_model.X)
                knn_model.train_model(model_path)
            elif args.testing_data_list:
                testing_data_path_list = args.testing_data_list
                for testing_data_path in testing_data_path_list:
                    for walk_str, walk_list in walk_class:
                        prediction_results = pd.DataFrame()
                        for walk in walk_list:
                            # 加載數據
                            knn_model.load_data(f"{testing_data_path}\\{walk}.csv")
                            knn_model.X = encoder.predict(knn_model.X)
                            results = knn_model.generate_predictions(model_path)
                            prediction_results = pd.concat([prediction_results, results], ignore_index=True)
                        split_path = testing_data_path.split('\\')
                        predictions_dir = f'predictions/{split_path[3]}'
                        os.makedirs(predictions_dir, exist_ok=True)
                        prediction_results.to_csv(os.path.join(predictions_dir, f'{walk_str}_predictions.csv'), index=False)
            else:
                print('Please specify --training_data or --test option.')

            os.chdir('..')

    
    if args.model == 'DNN':
        input_size = 8
        output_size = 41
        hidden_sizes = [8, 16, 32]
        dnn_model = DNN(input_size, output_size, hidden_sizes, args.work_dir)
        if args.training_data:
            dnn_model.load_data(args.training_data)
            dnn_model.X = encoder.predict(dnn_model.X)
            dnn_model.train_model(model_path, epochs=500)
        elif args.testing_data_list:
            testing_data_path_list = args.testing_data_list
            for testing_data_path in testing_data_path_list:
                for walk_str, walk_list in walk_class:
                    prediction_results = pd.DataFrame()
                    for walk in walk_list:
                        # 加載數據
                        dnn_model.load_data(f"{testing_data_path}\\{walk}.csv", shuffle=False)
                        dnn_model.X = encoder.predict(dnn_model.X)
                        results = dnn_model.generate_predictions(model_path)
                        prediction_results = pd.concat([prediction_results, results], ignore_index=True)
                    split_path = testing_data_path.split('\\')
                    predictions_dir = f'predictions/{split_path[3]}'
                    os.makedirs(predictions_dir, exist_ok=True)
                    prediction_results.to_csv(os.path.join(predictions_dir, f'{walk_str}_predictions.csv'), index=False)
        else:
            print('Please specify --training_data or --test option.')

    if args.model == 'RF':
        for i in range(1, 10):
            rf_model = RandomForest(max_depth=i)
            if args.training_data:
                rf_model.load_data(args.training_data)
                rf_model.X = encoder.predict(rf_model.X)
                rf_model.train_model(model_path)
            elif args.testing_data_list:
                testing_data_path_list = args.testing_data_list
                for testing_data_path in testing_data_path_list:
                    for walk_str, walk_list in walk_class:
                        prediction_results = pd.DataFrame()
                        for walk in walk_list:
                            # 加載數據
                            rf_model.load_data(f"{testing_data_path}\\{walk}.csv")
                            rf_model.X = encoder.predict(rf_model.X)
                            results = rf_model.generate_predictions(model_path)
                            prediction_results = pd.concat([prediction_results, results], ignore_index=True)
                        split_path = testing_data_path.split('\\')
                        predictions_dir = f'predictions/{split_path[3]}'
                        os.makedirs(predictions_dir, exist_ok=True)
                        prediction_results.to_csv(os.path.join(predictions_dir, f'{walk_str}_predictions.csv'), index=False)
            else:
                print('Please specify --training_data or --test option.')

            os.chdir('..')