import argparse
import pandas as pd
import numpy as np
import joblib
import csv
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint


class DNNIndoorLocalization:
    def __init__(self, input_size, output_size, hidden_sizes):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.model = self.build_model()
        self.history = None
        self.phone_list = ['GalaxyA51', 'hTCU11', 'hTCU19e', 'sharp025']
        self.scripted_walk = [
            'one_way_walk_1','one_way_walk_2','one_way_walk_3','one_way_walk_4','one_way_walk_5','one_way_walk_6','one_way_walk_7','one_way_walk_8',
            'round_trip_walk_2','round_trip_walk_3','round_trip_walk_4'
        ]       # No 'round_trip_walk_1'
        self.stationary = ['stationary_1']
        self.freewalk = [
            'freewalk_1','freewalk_2','freewalk_3','freewalk_4','freewalk_5','freewalk_6','freewalk_7','freewalk_8','freewalk_9'
        ]
        self.walk_class = [('scripted_walk', self.scripted_walk), ('stationary', self.stationary), ('freewalk', self.freewalk)]

    def load_data(self, training_data_path):
        data = pd.read_csv(training_data_path)
        X = data.iloc[:, 1:]
        y = data['label']
        y_adjusted = y - 1
        print(X.shape)

        # 进行one-hot编码
        one_hot_y = to_categorical(y_adjusted, num_classes=output_size)
        # print(one_hot_y.shape)
        # print(f"sample size: {self.X.shape}")

        #shuffle
        random_seed = 42  # 选择适当的随机种子
        np.random.seed(random_seed)
        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)

        self.X = np.array(X)[indices]
        self.y = one_hot_y[indices]

    def build_model(self):
        model = tf.keras.Sequential()

        # 添加输入层
        model.add(tf.keras.layers.Input(shape=(self.input_size,)))

        # 添加隐藏层
        for size in self.hidden_sizes:
            model.add(tf.keras.layers.Dense(size, activation='relu'))

        # 添加输出层
        model.add(tf.keras.layers.Dense(self.output_size, activation='softmax'))

        # 编译模型
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
        
    def train_model(self, model_path, epochs=10, batch_size=32):
        checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
        self.history = self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpointer])
        self.plot_loss_and_accuracy_history()

    def plot_loss_and_accuracy_history(self):
        plt.figure(figsize=(12, 6))

        # 画 loss 图
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 画 accuracy 图
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # 保存 loss 图
        plt.savefig("loss_and_accuracy.png")
        plt.clf()

    def predict(self, input_data):
        # Assuming input_data is a list of RSSI values for Beacon_1 to Beacon_7
        # input_data = [input_data]  # Scikit-Learn's predict method expects a 2D array
        return self.model.predict(input_data)
    
    # def save_model(self, model_path):
    #     # 使用 TensorFlow 的模型保存函数来保存模型
    #     self.model.save(model_path)
    #     print(f"模型已保存到 {model_path}")

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def generate_predictions(self, testing_data_path):
        for walk_str, walk_list in self.walk_class:
            prediction_results = {
                'label': [],
                'pred': []
            }
            for walk in walk_list:
                for phone in self.phone_list:
                    data_path = f"{testing_data_path}\\{walk}\\{phone}.csv"

                    # 加載數據
                    self.load_data(data_path)

                    # 進行預測
                    predicted_labels = self.predict(self.X)
                    predicted_labels = np.argmax(predicted_labels, axis=1) + 1  # 加 1 是为了将索引转换为 1 到 41 的标签
                    label = np.argmax(self.y, axis=1) + 1
                    # 將預測結果保存到 prediction_results 中
                    prediction_results['label'].extend(label.tolist())
                    prediction_results['pred'].extend(predicted_labels.tolist())
            df = pd.DataFrame(prediction_results)
            df.to_csv(f'{walk_str}_predictions.csv', index=False)

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

    def calculate_mde(self):
        print('Calculate MDE...')
        total_mde = []
        errors = []
        for walk_str, _ in self.walk_class:
            mde = 0
            predict_file = pd.read_csv(f'{walk_str}_predictions.csv')
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
    def test(self, testing_data_path_list):
        mde_list = []
        for testing_data_path in testing_data_path_list:
            self.generate_predictions(testing_data_path)
            mde_list.append(self.calculate_mde())
        label_list = ["0318 testing data", "1028 testing data"]
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
        plt.title("MDE Comparison")
        plt.xlabel("Test Cases")
        plt.ylabel("MDE Value")

        # 保存图像到文件
        plt.savefig("mde_comparison.png")

        # # 显示图表
        # plt.show()

# 使用示例
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='DNN Indoor Localization')

    # 添加参数选项
    parser.add_argument('--training_data', type=str, help='csv file of the training data')
    parser.add_argument('--test', action='store_true', help='Test the model')

    # 解析命令行参数
    args = parser.parse_args()

    model_path = "my_model.h5"
    # 在根据参数来执行相应的操作
    input_size = 7
    output_size = 41
    hidden_sizes = [8, 16, 32]
    dnn_model = DNNIndoorLocalization(input_size, output_size, hidden_sizes)
    if args.training_data:
        dnn_model.load_data(args.training_data)
        dnn_model.train_model(model_path, epochs=100)
    elif args.test:
        dnn_model.load_model(model_path)
        testing_data_path_list = ["D:\\ExperimentData\\0318_92589_test", "D:\\ExperimentData\\1028_92589_test"]
        dnn_model.test(testing_data_path_list)
    else:
        print('Please specify --training_data or --test option.')


    
    
    


    
