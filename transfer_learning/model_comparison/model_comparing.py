'''
Time Experiment:
python .\model_comparing.py \
    --model1_prediction_list D:\Experiment\transfer_learning\DANN\220318_231116\12_0.0\predictions\220318 \
                             D:\Experiment\transfer_learning\DANN\220318_231116\12_0.0\predictions\231116 \
    --model2_prediction_list D:\Experiment\transfer_learning\DANN_AE\220318_231116\122_0.0\predictions\220318 \
                             D:\Experiment\transfer_learning\DANN_AE\220318_231116\122_0.0\predictions\231116 \
    --model3_prediction_list D:\Experiment\transfer_learning\DANN_CORR\220318_231116\0.1_10_0.0\predictions\220318 \
                             D:\Experiment\transfer_learning\DANN_CORR\220318_231116\0.1_10_0.0\predictions\231116 \
    --model4_prediction_list D:\Experiment\transfer_learning\DANN_CORR_AE\220318_231116\0.1_2_2_0.0\predictions\220318 \
                             D:\Experiment\transfer_learning\DANN_CORR_AE\220318_231116\0.1_2_2_0.0\predictions\231116 \
    --experiment_name time_variation
Space Experiment:
python .\model_comparing.py \
    --model1_prediction_list D:\Experiment\transfer_learning\DANN\231116_231117\12_0.0\predictions\231116 \
                             D:\Experiment\transfer_learning\DANN\231116_231117\12_0.0\predictions\231117 \
    --model2_prediction_list D:\Experiment\transfer_learning\DANN_AE\231116_231117\122_0.0\predictions\231116 \
                             D:\Experiment\transfer_learning\DANN_AE\231116_231117\122_0.0\predictions\231117 \
    --model3_prediction_list D:\Experiment\transfer_learning\DANN_CORR\231116_231117\0.1_10_0.0\predictions\231116 \
                             D:\Experiment\transfer_learning\DANN_CORR\231116_231117\0.1_10_0.0\predictions\231117 \
    --model4_prediction_list D:\Experiment\transfer_learning\DANN_CORR_AE\231116_231117\0.1_2_2_0.0\predictions\231116 \
                             D:\Experiment\transfer_learning\DANN_CORR_AE\231116_231117\0.1_2_2_0.0\predictions\231117 \
    --experiment_name spatail_variation
'''

import numpy as np
import sys
import argparse
sys.path.append('..\\..\\model_comparison')
from walk_definitions import walk_class
from evaluator import Evaluator
import matplotlib.pyplot as plt


# 使用示例
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Indoor Localization')

    parser.add_argument('--model1_prediction_list', nargs='+', type=str, required = True, help='List of model1 prediction paths')
    parser.add_argument('--model2_prediction_list', nargs='+', type=str, required = True, help='List of model2 prediction paths')
    parser.add_argument('--model3_prediction_list', nargs='+', type=str, required = True, help='List of model3 prediction paths')
    parser.add_argument('--model4_prediction_list', nargs='+', type=str, required = True, help='List of model4 prediction paths')
    parser.add_argument('--experiment_name', type=str, default='', required = True, help='Would show on the pigure name')


    # 解析命令行参数
    args = parser.parse_args()
    evaluator = Evaluator()
    model1_error = []
    model2_error = []
    model3_error = []
    model4_error = []
    for model1_prediction in args.model1_prediction_list:
        total_mde, total_errors = evaluator.calculate_mde(model1_prediction)
        model1_error.extend(total_errors[1])
    for model2_prediction in args.model2_prediction_list:
        total_mde, total_errors = evaluator.calculate_mde(model2_prediction)
        model2_error.extend(total_errors[1])
    for model3_prediction in args.model3_prediction_list:
        total_mde, total_errors = evaluator.calculate_mde(model3_prediction)
        model3_error.extend(total_errors[1])
    for model4_prediction in args.model4_prediction_list:
        total_mde, total_errors = evaluator.calculate_mde(model4_prediction)
        model4_error.extend(total_errors[1])

    # print(model1_error)
    print(len(model1_error))
    print(np.std(model1_error))
    # print(model2_error)
    print(len(model2_error))
    print(np.std(model2_error))
    # print(model3_error)
    print(len(model3_error))
    print(np.std(model3_error))
    # print(model4_error)
    print(len(model4_error))
    print(np.std(model4_error))
    color_list = ['red', 'black', 'purple', 'brown']
    evaluator.plot_cdf(model1_error, 'DANN', color_list[0])
    evaluator.plot_cdf(model2_error, 'DANN_AE', color_list[1])
    evaluator.plot_cdf(model3_error, 'DANN_CORR', color_list[2])
    evaluator.plot_cdf(model4_error, 'DANN_CORR_AE', color_list[3])
    plt.title(f'{args.experiment_name} CDF of Errors')
    plt.xlabel('Error')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.savefig(f"{args.experiment_name}_CDF.png")
    plt.clf()