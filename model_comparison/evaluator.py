'''
python .\evaluator.py --model 231116 --directory DNN
'''

import argparse
import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from walk_definitions import date2domain

class Evaluator:
    def __init__(self):
        self.walk_class = ['scripted_walk', 'stationary', 'freewalk']

    def class_to_coordinate(self, a):
        table = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
            [ 0, 1, 0, 0, 0, 0,11, 0, 0, 0, 0,21],\
            [ 0, 2, 0, 0, 0, 0,12, 0, 0, 0, 0,22],\
            [ 0, 3, 0, 0, 0, 0,13, 0, 0, 0, 0,23],\
            [ 0, 4, 0, 0, 0, 0,14, 0, 0, 0, 0,24],\
            [ 0, 5, 0, 0, 0, 0,15, 0, 0, 0, 0,25],\
            [ 0, 6, 0, 0, 0, 0,16, 0, 0, 0, 0,26],\
            [ 0, 7, 0, 0, 0, 0,17, 0, 0, 0, 0,27],\
            [ 0, 8, 0, 0, 0, 0,18, 0, 0, 0, 0,28],\
            [ 0, 9, 0, 0, 0, 0,19, 0, 0, 0, 0,29],\
            [ 0,10, 0, 0, 0, 0,20, 0, 0, 0, 0,30],\
            [ 0,31,32,33,34,35,36,37,38,39,40,41] ], dtype = int)
        x = np.argwhere(table == a)[0][1]
        y = np.argwhere(table == a)[0][0]
        coordinate = [x,y]
        return coordinate

    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])) * 0.6

    def calculate_mde(self, prediction_data_path):
        print('Calculate MDE...')
        total_mde = []
        errors = []
        for walk_str in self.walk_class:
            mde = 0
            predict_file = pd.read_csv(f'{prediction_data_path}\\{walk_str}_predictions.csv')
            for i in range(len(predict_file)):
                y = predict_file['label'].iloc[i] # Label
                y_hat = predict_file['pred'].iloc[i] # Predicted value
                de = self.euclidean_distance(self.class_to_coordinate(y), self.class_to_coordinate(y_hat))
                errors.append(de)
                mde += de
            mde = mde / len(predict_file)
            total_mde.append(mde)
            
            csv_file_path = f'{walk_str}_cdf.csv'
            x = np.arange(0, 8.2, 0.2) # 0~8m
            # 判斷CSV文件是否存在
            if not os.path.exists(csv_file_path):
                # 如果文件不存在，写入头部信息
                with open(csv_file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(x.tolist())

            # 继续写入概率信息
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                errors_pd = pd.DataFrame(errors)
                prob = []
                for i in x:
                    result = errors_pd.where(errors_pd <= i).count().sum()
                    prob.append(result / len(errors_pd))
                writer.writerow(prob)
        print(total_mde)
        print(f'average MDE: {sum(total_mde) / len(total_mde)}')
        return total_mde
    
    def test(self, predicion_data_path_list, total_model_name):
        mde_list = []
        label_list = []
        for predicion_data_path in predicion_data_path_list:
            mde_list.append(self.calculate_mde(f'predictions\\{predicion_data_path}'))
            split_path = predicion_data_path.split('\\')
            label = date2domain[f'{split_path[-1]}']
            label_list.append(label)
        # X轴标签
        labels = ["scripted_walk", "stationary", "freewalk"]

        # 柱状图的宽度
        bar_width = 0.2

        # X轴的位置
        x = range(len(labels))

        # 绘制柱状图
        for i, mde in enumerate(mde_list):
            plt.bar([j + i * bar_width for j in x], mde_list[i], width=bar_width, label=label_list[i])

        # 设置X轴标签
        plt.xticks([i + bar_width/2 for i in x], labels)

        # 添加图例
        plt.legend()

        # 在柱形上显示数据标签
        for i, mde in enumerate(mde_list):
            for j, v in enumerate(mde):
                plt.text(x[j] + i * bar_width, v, f'{v:.3f}', ha='center', va='bottom')

        # 设置Y轴范围
        plt.ylim(0, 3)
        
        # 设置图表标题和轴标签
        plt.title(f"{total_model_name} MDE Comparison")
        plt.xlabel("Test Cases")
        plt.ylabel("MDE Value")

        # 保存图像到文件
        plt.savefig("mde_comparison.png")


# 使用示例
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Indoor Localization')

    parser.add_argument('--model_name', type=str, default='', help='Would show on the pigure title')
    parser.add_argument('--directory', type=str, help='Evaluated model directory which has \'predictions\' directory')

    # 解析命令行参数
    args = parser.parse_args()

    os.chdir(args.directory)
    predicion_data_path_list = os.listdir('predictions/')
    total_model_name = date2domain[f'{args.model_name}'] + f' {args.directory}'
    evaluator = Evaluator()
    evaluator.test(predicion_data_path_list, total_model_name)


