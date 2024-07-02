'''
Mall:
python .\CDF.py ^
    --model_prediction_list ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_pytorch/211120_221221/1_0_0.9/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_pytorch/211120_221221/1_1_0.9/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_AE/211120_221221/1_2_2_0.9/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_1DCAE/211120_221221/0.1_0.1_10_0.9/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/AdapLoc/211120_221221/1_0.01_0.9/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_baseline/211120_221221/0.1_0.1_10_0.9/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_CORR/211120_221221/0.1_10_0.9/predictions/221221_results.csv ^
    --experiment_name 7_labeled
python .\CDF.py ^
    --model_prediction_list ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_pytorch/211120_221221/unlabeled/1_0_0.0/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_pytorch/211120_221221/unlabeled/1_1_0.0/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_AE/211120_221221/unlabeled/1_2_2_0.0/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_1DCAE/211120_221221/unlabeled/0.1_0.1_10_0.0/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/AdapLoc/211120_221221/unlabeled/1_0.01_0.0/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_baseline/211120_221221/unlabeled/0.1_0.1_10_0.0/predictions/221221_results.csv ^
    D:/Experiment/Mall_dataset_transfer_learning/DANN_CORR/211120_221221/unlabeled/0.1_10_0.0/predictions/221221_results.csv ^
    --experiment_name 7_unlabeled
'''


import numpy as np
import sys
import argparse
import evaluator
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
    model_mdes, model_errors = [], []
    cdfs = []
    for model_prediction in args.model_prediction_list:
        _, errors = evaluator.calculate_mde(model_prediction)
        model_errors.append(errors)

    model_names = ['DNN', 'DANN', 'DANN_AE', 'DANN_1DCAE', 'AdapLoc', 'FusionDANN', 'HistLoc']
    color_list = ['red', 'black', 'purple', 'brown', 'gray', 'pink', 'yellow', 'steelblue']
    for j in range(len(model_errors)):
        cdf, bin_edges = evaluator.plot_cdf(model_errors[j], model_names[j], color_list[j], 60.0, 2.0)
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