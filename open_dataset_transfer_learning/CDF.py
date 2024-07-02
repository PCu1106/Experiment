'''
Opendataset1:
python .\CDF.py ^
    --model_prediction_list ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/190611_191009/1_0_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/190611_191009/1_1_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_AE/190611_191009/1_2_2_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_1DCAE/190611_191009/0.1_0.1_10_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/AdapLoc/190611_191009/1_0.01_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_baseline/190611_191009/0.1_0.1_10_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_CORR/190611_191009/0.1_10_0.9/predictions/191009_results.csv ^
    --experiment_name 3_labeled
python .\CDF.py ^
    --model_prediction_list ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/190611_191009/unlabeled/1_0_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/190611_191009/unlabeled/1_1_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_AE/190611_191009/unlabeled/1_2_2_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_1DCAE/190611_191009/unlabeled/0.1_0.1_10_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/AdapLoc/190611_191009/unlabeled/1_0.01_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_baseline/190611_191009/unlabeled/0.1_0.1_10_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_CORR/190611_191009/unlabeled/0.1_10_0.0/predictions/191009_results.csv ^
    --experiment_name 3_unlabeled
Opendataset2:
python .\CDF.py ^
    --model_prediction_list ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/190611_200219/1_0_0.9/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/190611_200219/1_1_0.9/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_AE/190611_200219/1_2_2_0.9/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_1DCAE/190611_200219/0.1_0.1_10_0.9/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/AdapLoc/190611_200219/1_0.01_0.9/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_baseline/190611_200219/0.1_0.1_10_0.9/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_CORR/190611_200219/0.1_10_0.9/predictions/200219_results.csv ^
    --experiment_name 4_labeled
python .\CDF.py ^
    --model_prediction_list ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/190611_200219/unlabeled/1_0_0.0/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/190611_200219/unlabeled/1_1_0.0/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_AE/190611_200219/unlabeled/1_2_2_0.0/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_1DCAE/190611_200219/unlabeled/0.1_0.1_10_0.0/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/AdapLoc/190611_200219/unlabeled/1_0.01_0.0/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_baseline/190611_200219/unlabeled/0.1_0.1_10_0.0/predictions/200219_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_CORR/190611_200219/unlabeled/0.1_10_0.0/predictions/200219_results.csv ^
    --experiment_name 4_unlabeled
Time reversal1:
python .\CDF.py ^
    --model_prediction_list ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/200219_190611/1_0_0.9/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/200219_190611/1_1_0.9/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_AE/200219_190611/1_2_2_0.9/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_1DCAE/200219_190611/0.1_0.1_10_0.9/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/AdapLoc/200219_190611/1_0.01_0.9/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_baseline/200219_190611/0.1_0.1_10_0.9/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_CORR/200219_190611/0.1_10_0.9/predictions/190611_results.csv ^
    --experiment_name 5_labeled
python .\CDF.py ^
    --model_prediction_list ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/200219_190611/unlabeled/1_0_0.0/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/200219_190611/unlabeled/1_1_0.0/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_AE/200219_190611/unlabeled/1_2_2_0.0/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_1DCAE/200219_190611/unlabeled/0.1_0.1_10_0.0/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/AdapLoc/200219_190611/unlabeled/1_0.01_0.0/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_baseline/200219_190611/unlabeled/0.1_0.1_10_0.0/predictions/190611_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_CORR/200219_190611/unlabeled/0.1_10_0.0/predictions/190611_results.csv ^
    --experiment_name 5_unlabeled
Time reversal2:
python .\CDF.py ^
    --model_prediction_list ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/200219_191009/1_0_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/200219_191009/1_1_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_AE/200219_191009/1_2_2_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_1DCAE/200219_191009/0.1_0.1_10_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/AdapLoc/200219_191009/1_0.01_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_baseline/200219_191009/0.1_0.1_10_0.9/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_CORR/200219_191009/0.1_10_0.9/predictions/191009_results.csv ^
    --experiment_name 6_labeled
python .\CDF.py ^
    --model_prediction_list ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/200219_191009/unlabeled/1_0_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_pytorch/200219_191009/unlabeled/1_1_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_AE/200219_191009/unlabeled/1_2_2_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_1DCAE/200219_191009/unlabeled/0.1_0.1_10_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/AdapLoc/200219_191009/unlabeled/1_0.01_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_baseline/200219_191009/unlabeled/0.1_0.1_10_0.0/predictions/191009_results.csv ^
    D:/Experiment/open_dataset_transfer_learning/DANN_CORR/200219_191009/unlabeled/0.1_10_0.0/predictions/191009_results.csv ^
    --experiment_name 6_unlabeled
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
        cdf, bin_edges = evaluator.plot_cdf(model_errors[j], model_names[j], color_list[j], 16.0, 0.2)
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