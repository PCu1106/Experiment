'''
python .\TL_DNN.py \
    --training_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv \
    --base_model_path D:\Experiment\model_comparison\DNN\231116.h5 \
    --freeze_layers 1 \
    --model_path 231116_220318.h5 \
    --work_dir 231116_220318\TL_DNN_1

python .\TL_DNN.py \
    --testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                        D:\Experiment\data\220318\GalaxyA51\routes \
                        D:\Experiment\data\231117\GalaxyA51\routes \
    --model_path 231116_220318.h5 \
    --work_dir 231116_220318\TL_DNN_1
'''

import sys
sys.path.append('..\\model_comparison')
from DNN import DNN
from walk_definitions import walk_class
import pandas as pd
import argparse
import os


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Indoor Localization')

    # 添加参数选项
    parser.add_argument('--training_data', type=str, help='csv file of the training data')
    parser.add_argument('--testing_data_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--base_model_path', type=str, help='the path of base model to be transfered')
    parser.add_argument('--freeze_layers', type=int, help='how many layer want to freeze')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='path of contating model')
    parser.add_argument('--work_dir', type=str, help='create new directory to save result')
    

    # 解析命令行参数
    args = parser.parse_args()

    base_model_path = args.base_model_path
    model_path = args.model_path

    dnn_model = DNN(work_dir=args.work_dir)

    if args.training_data:
        dnn_model.load_model(base_model_path)
        dnn_model.build_transfer_model(freeze_layers = args.freeze_layers)
        dnn_model.load_data(args.training_data)
        dnn_model.train_model(model_path, epochs=500)
    elif args.testing_data_list:
        testing_data_path_list = args.testing_data_list
        for testing_data_path in testing_data_path_list:
            for walk_str, walk_list in walk_class:
                prediction_results = pd.DataFrame()
                for walk in walk_list:
                    # 加載數據
                    dnn_model.load_data(f"{testing_data_path}\\{walk}.csv", shuffle=False)
                    results = dnn_model.generate_predictions(model_path)
                    prediction_results = pd.concat([prediction_results, results], ignore_index=True)
                split_path = testing_data_path.split('\\')
                predictions_dir = f'predictions/{split_path[3]}'
                os.makedirs(predictions_dir, exist_ok=True)
                prediction_results.to_csv(os.path.join(predictions_dir, f'{walk_str}_predictions.csv'), index=False)
    else:
        print('Please specify --training_data or --test option.')