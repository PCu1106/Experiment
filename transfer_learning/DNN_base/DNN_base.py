'''
python .\DNN_base.py \
    --training_source_domain_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv \
    --training_target_domain_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
    --model_path 220318_231116.h5 \
    --work_dir 220318_231116
python .\DNN_base.py \
    --testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                        D:\Experiment\data\220318\GalaxyA51\routes \
                        D:\Experiment\data\231117\GalaxyA51\routes \
    --model_path 220318_231116.h5 \
    --work_dir 220318_231116
'''
import sys
sys.path.append('..\\..\\model_comparison')
from DNN import DNN

import argparse
from walk_definitions import walk_class
from evaluator import Evaluator
import os
import pandas as pd

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='DNN Indoor Localization')

    # 添加参数选项
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
    parser.add_argument('--testing_data_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='path of .h5 file of model')
    parser.add_argument('--work_dir', type=str, default='DNN', help='create new directory to save result')


    # 解析命令行参数
    args = parser.parse_args()

    model_path = args.model_path
    # 在根据参数来执行相应的操作
    input_size = 7
    output_size = 41
    hidden_sizes = [8, 16, 32]

    data_drop_out = 0.9
    dnn_model = DNN(input_size, output_size, hidden_sizes, args.work_dir)
    if args.training_source_domain_data:
        dnn_model.load_two_data(args.training_source_domain_data, args.training_target_domain_data, data_drop_out=data_drop_out)
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