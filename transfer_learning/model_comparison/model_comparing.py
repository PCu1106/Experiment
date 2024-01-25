'''
Time Experiment:
python .\model_comparing.py \
    --model1_prediction_list D:\Experiment\transfer_learning\DANN\220318_231116\12_0.9\predictions\220318 \
                             D:\Experiment\transfer_learning\DANN\220318_231116\12_0.9\predictions\231116 \
    --model2_prediction_list D:\Experiment\transfer_learning\DANN_AE\220318_231116\122_0.9\predictions\220318 \
                             D:\Experiment\transfer_learning\DANN_AE\220318_231116\122_0.9\predictions\231116 \
    --model3_prediction_list D:\Experiment\transfer_learning\DANN_CORR\220318_231116\0.1_10_0.9\predictions\220318 \
                             D:\Experiment\transfer_learning\DANN_CORR\220318_231116\0.1_10_0.9\predictions\231116 \
    --model4_prediction_list D:\Experiment\transfer_learning\DANN_CORR_AE\220318_231116\0.1_2_2_0.9\predictions\220318 \
                             D:\Experiment\transfer_learning\DANN_CORR_AE\220318_231116\0.1_2_2_0.9\predictions\231116 \
    --model5_prediction_list D:\Experiment\transfer_learning\DNN_base\220318_231116\predictions\220318 \
                             D:\Experiment\transfer_learning\DNN_base\220318_231116\predictions\231116 \
    --experiment_name time_variation
Space Experiment:
python .\model_comparing.py \
    --model1_prediction_list D:\Experiment\transfer_learning\DANN\231116_231117\12_0.9\predictions\231116 \
                             D:\Experiment\transfer_learning\DANN\231116_231117\12_0.9\predictions\231117 \
    --model2_prediction_list D:\Experiment\transfer_learning\DANN_AE\231116_231117\122_0.9\predictions\231116 \
                             D:\Experiment\transfer_learning\DANN_AE\231116_231117\122_0.9\predictions\231117 \
    --model3_prediction_list D:\Experiment\transfer_learning\DANN_CORR\231116_231117\0.1_10_0.9\predictions\231116 \
                             D:\Experiment\transfer_learning\DANN_CORR\231116_231117\0.1_10_0.9\predictions\231117 \
    --model4_prediction_list D:\Experiment\transfer_learning\DANN_CORR_AE\231116_231117\0.1_2_2_0.9\predictions\231116 \
                             D:\Experiment\transfer_learning\DANN_CORR_AE\231116_231117\0.1_2_2_0.9\predictions\231117 \
    --model5_prediction_list D:\Experiment\transfer_learning\DNN_base\231116_231117\predictions\231116 \
                             D:\Experiment\transfer_learning\DNN_base\231116_231117\predictions\231117 \
    --experiment_name spatail_variation
'''

import numpy as np
import sys
import argparse
sys.path.append('..\\..\\model_comparison')
from walk_definitions import walk_class
from walk_definitions import date2color
from evaluator import Evaluator
import matplotlib.pyplot as plt


# 使用示例
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Indoor Localization')

    parser.add_argument('--model1_prediction_list', nargs='+', type=str, required = True, help='List of model1 prediction paths')
    parser.add_argument('--model2_prediction_list', nargs='+', type=str, required = True, help='List of model2 prediction paths')
    parser.add_argument('--model3_prediction_list', nargs='+', type=str, required = True, help='List of model3 prediction paths')
    parser.add_argument('--model4_prediction_list', nargs='+', type=str, required = True, help='List of model4 prediction paths')
    parser.add_argument('--model5_prediction_list', nargs='+', type=str, required = True, help='List of model5 prediction paths')
    parser.add_argument('--experiment_name', type=str, default='', required = True, help='Would show on the pigure name')


    # 解析命令行参数
    args = parser.parse_args()
    evaluator = Evaluator()
    model1_mde, model1_error = [], []
    model2_mde, model2_error = [], []
    model3_mde, model3_error = [], []
    model4_mde, model4_error = [], []
    model5_mde, model5_error = [], []
    for model1_prediction in args.model1_prediction_list:
        total_mde, total_errors = evaluator.calculate_mde(model1_prediction) # [scripted_walk mde, stationary mde, freewalk mde]
        model1_error.append(total_errors[1])
        model1_mde.append(total_mde[1])
    for model2_prediction in args.model2_prediction_list:
        total_mde, total_errors = evaluator.calculate_mde(model2_prediction)
        model2_error.append(total_errors[1])
        model2_mde.append(total_mde[1])
    for model3_prediction in args.model3_prediction_list:
        total_mde, total_errors = evaluator.calculate_mde(model3_prediction)
        model3_error.append(total_errors[1])
        model3_mde.append(total_mde[1])
    for model4_prediction in args.model4_prediction_list:
        total_mde, total_errors = evaluator.calculate_mde(model4_prediction)
        model4_mde.append(total_mde[1])
        model4_error.append(total_errors[1])
    for model5_prediction in args.model5_prediction_list:
        total_mde, total_errors = evaluator.calculate_mde(model5_prediction)
        model5_mde.append(total_mde[1])
        model5_error.append(total_errors[1])

    model_names = ['DANN', 'DANN_AE', 'DANN_CORR', 'DANN_CORR_AE', 'DNN']
    color_list = ['red', 'black', 'purple', 'brown', 'gray']
    for i, domain in enumerate(['source domain', 'target domain']):
        evaluator.plot_cdf(model1_error[i], model_names[0], color_list[0])
        evaluator.plot_cdf(model2_error[i], model_names[1], color_list[1])
        evaluator.plot_cdf(model3_error[i], model_names[2], color_list[2])
        evaluator.plot_cdf(model4_error[i], model_names[3], color_list[3])
        evaluator.plot_cdf(model5_error[i], model_names[4], color_list[4])
        plt.title(f'{args.experiment_name} {domain} CDF of Errors')
        plt.xlabel('Error')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.savefig(f"{args.experiment_name} {domain}_CDF.png")
        plt.clf()

    if args.experiment_name == 'time_variation':
        color_list = [date2color['220318'], date2color['231116']]
    elif args.experiment_name == 'spatail_variation':
        color_list = [date2color['231116'], date2color['231117']]
    model_mde = [model1_mde, model2_mde, model3_mde, model4_mde, model5_mde]
    bar_width = 0.35

    # 設定x軸的位置
    index = range(len(model_names))  # 每組 model 只有一個長條

    # 繪製長條圖
    plt.figure(figsize=(8, 6))
    for i, model_name in enumerate(model_names):
        source_index = i
        target_index = i + bar_width
        plt.bar(source_index, model_mde[i][0], width=bar_width, label=f"{model_name}_source", color=color_list[0])
        plt.bar(target_index, model_mde[i][1], width=bar_width, label=f"{model_name}_target", color=color_list[1])
        # 在長條頂部顯示數字
        plt.text(source_index, model_mde[i][0], f'{model_mde[i][0]:.2f}', ha='center', va='bottom', color='black')
        plt.text(target_index, model_mde[i][1], f'{model_mde[i][1]:.2f}', ha='center', va='bottom', color='black')


    # 設定x軸標籤
    plt.xlabel('Models')
    # 設定y軸標籤
    plt.ylabel('MDE')
    # 设置Y轴范围
    plt.ylim(0, 2)
    # 添加單一的 Source Domain 和 Target Domain 圖例
    plt.legend(['Source Domain', 'Target Domain'])
    # 設定x軸刻度
    plt.xticks([i + bar_width/2 for i in index], model_names)
    plt.title(f'{args.experiment_name} MDE')

    plt.savefig(f"{args.experiment_name}_bar.png")
    plt.clf()

    errors = [model1_error, model2_error, model3_error, model4_error, model5_error]
    # 設定 source domain 和 target domain 的位置
    positions = np.array(range(len(model_names))) * 2.0  # 每隔2的位置
    # 將 source domain 跟 target domain 分別放入一個列表
    source_errors = [error[0] for error in errors]
    target_errors = [error[1] for error in errors] 

    # 設定中位數線的顏色
    medianprops = dict(linestyle='-', linewidth=2, color='red')
    # 設定箱型圖的顏色
    boxprops = dict(linestyle='-', linewidth=2, color=color_list[0])
    # 繪製 source domain 的箱型圖
    plt.boxplot(source_errors, positions=positions - 0.3, labels=model_names, widths=0.3, boxprops=boxprops, medianprops=medianprops)
    # 繪製 target domain 的箱型圖
    # 設定箱型圖的顏色
    boxprops = dict(linestyle='-', linewidth=2, color=color_list[1])
    plt.boxplot(target_errors, positions=positions + 0.3, labels=model_names, widths=0.3, boxprops=boxprops, medianprops=medianprops)

    # 設定標題和標籤
    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.title(f'{args.experiment_name} Box of Errors')
    # 設定x軸刻度
    plt.legend(['Source Domain', 'Target Domain'])

    plt.xticks(positions, model_names)
    # 顯示圖形
    plt.savefig(f"{args.experiment_name}_box.png")
    plt.clf()