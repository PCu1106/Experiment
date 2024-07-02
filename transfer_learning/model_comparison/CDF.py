'''
Time Experiment:
python .\CDF.py ^
    --model_prediction_list ^
    D:\Experiment\transfer_learning\DNN_base\220318_231116_0.9\predictions\231116 ^
    D:\Experiment\transfer_learning\DANN\220318_231116\12_0.9\predictions\231116 ^
    D:\Experiment\transfer_learning\DANN_AE\220318_231116\122_0.9\predictions\231116 ^
    D:\Experiment\transfer_learning\DANN_1DCAE\220318_231116\0.1_0.1_10_0.9\predictions\231116 ^
    D:\Experiment\transfer_learning\AdapLoc\220318_231116\1_0.01_0.9\predictions\231116 ^
    D:\Experiment\transfer_learning\DANN_baseline\220318_231116\0.1_0.1_10_0.9\predictions\231116 ^
    D:\Experiment\transfer_learning\DANN_CORR\220318_231116\0.1_10_0.9\predictions\231116 ^
    --experiment_name 1_time_variation_labeled
python .\CDF.py ^
    --model_prediction_list ^
    D:\Experiment\transfer_learning\DNN_base\220318_231116_1.0\predictions\231116 ^
    D:\Experiment\transfer_learning\DANN\\unlabeled\220318_231116\12_0.0\predictions\231116 ^
    D:\Experiment\transfer_learning\DANN_AE\\unlabeled\220318_231116\122_0.0\predictions\231116 ^
    D:\Experiment\transfer_learning\DANN_1DCAE\\unlabeled\220318_231116\0.1_0.1_10_0.0\predictions\231116 ^
    D:\Experiment\transfer_learning\AdapLoc\\unlabeled\220318_231116\1_0.01_0.0\predictions\231116 ^
    D:\Experiment\transfer_learning\DANN_baseline\\unlabeled\220318_231116\0.1_0.1_10_0.0\predictions\231116 ^
    D:\Experiment\transfer_learning\DANN_CORR\\unlabeled\220318_231116\0.1_10_0.0\predictions\231116 ^
    --experiment_name 1_time_variation_unlabeled
Space Experiment:
python .\CDF.py ^
    --model_prediction_list ^
    D:\Experiment\transfer_learning\DNN_base\231116_231117_0.9\predictions\231117 ^
    D:\Experiment\transfer_learning\DANN\231116_231117\12_0.9\predictions\231117 ^
    D:\Experiment\transfer_learning\DANN_AE\231116_231117\122_0.9\predictions\231117 ^
    D:\Experiment\transfer_learning\DANN_1DCAE\231116_231117\0.1_0.1_10_0.9\predictions\231117 ^
    D:\Experiment\transfer_learning\AdapLoc\231116_231117\1_0.01_0.9\predictions\231117 ^
    D:\Experiment\transfer_learning\DANN_baseline\231116_231117\0.1_0.1_10_0.9\predictions\231117 ^
    D:\Experiment\transfer_learning\DANN_CORR\231116_231117\0.1_10_0.9\predictions\231117 ^
    --experiment_name 2_spatial_variation_labeled
python .\CDF.py ^
    --model_prediction_list ^
    D:\Experiment\transfer_learning\DNN_base\231116_231117_1.0\predictions\231117 ^
    D:\Experiment\transfer_learning\DANN\\unlabeled\231116_231117\12_0.0\predictions\231117 ^
    D:\Experiment\transfer_learning\DANN_AE\\unlabeled\231116_231117\122_0.0\predictions\231117 ^
    D:\Experiment\transfer_learning\DANN_1DCAE\\unlabeled\231116_231117\0.1_0.1_10_0.0\predictions\231117 ^
    D:\Experiment\transfer_learning\AdapLoc\\unlabeled\231116_231117\1_0.01_0.0\predictions\231117 ^
    D:\Experiment\transfer_learning\DANN_baseline\\unlabeled\231116_231117\0.1_0.1_10_0.0\predictions\231117 ^
    D:\Experiment\transfer_learning\DANN_CORR\\unlabeled\231116_231117\0.1_10_0.0\predictions\231117 ^
    --experiment_name 2_spatial_variation_unlabeled
'''


import numpy as np
import sys
import argparse
sys.path.append('..\\..\model_comparison')
from walk_definitions import walk_class
from walk_definitions import date2color
from evaluator import Evaluator
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd

# 使用示例
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Indoor Localization')

    parser.add_argument('--model_prediction_list', nargs='+', type=str, required = True, help='List of model prediction paths')
    parser.add_argument('--experiment_name', type=str, default='', required = True, help='Would show on the pigure name')


    # 解析命令行参数
    args = parser.parse_args()
    evaluator = Evaluator()
    model_mdes, model_errors = [], []
    cdfs = []
    for model_prediction in args.model_prediction_list:
        _, total_errors = evaluator.calculate_mde(model_prediction) # [scripted_walk mde, stationary mde, freewalk mde]
        model_errors.append(total_errors[1])

    model_names = ['DNN', 'DANN', 'DANN_AE', 'DANN_1DCAE', 'AdapLoc', 'FusionDANN', 'HistLoc']
    color_list = ['red', 'black', 'purple', 'brown', 'gray', 'pink', 'yellow', 'steelblue']
    for j in range(len(model_errors)):
        cdf, bin_edges = evaluator.plot_cdf(model_errors[j], model_names[j], color_list[j])
        cdfs.append(cdf)
    plt.title(f'{args.experiment_name} CDF of Errors of Target Domain')
    plt.xlabel('Error')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.savefig(f"CDF/{args.experiment_name}.png")
    plt.clf()
    # Write losses to CSV
    print(cdfs)
    with open(f"CDF/{args.experiment_name}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Error'] + list(bin_edges))
        for i, label in enumerate(model_names):
            writer.writerow([label] + list(cdfs[i]))