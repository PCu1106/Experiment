'''
python .\DNN.py \
--training_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
--model_path 231116.h5

python .\dnn.py \
--testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                    D:\Experiment\data\220318\GalaxyA51\routes \
                    D:\Experiment\data\231117\GalaxyA51\routes \
--model_path 231116.h5
'''

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import os

class DNN:
    def __init__(self, input_size, output_size, hidden_sizes):
        if not os.path.exists('DNN'):
            os.makedirs('DNN')
        os.chdir('DNN')
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.model = self.build_model()
        self.history = None
        self.scripted_walk = [
            'one_way_walk_1','one_way_walk_2','one_way_walk_3','one_way_walk_4','one_way_walk_5','one_way_walk_6','one_way_walk_7','one_way_walk_8',
            'round_trip_walk_1', 'round_trip_walk_2','round_trip_walk_3','round_trip_walk_4'
        ]       
        self.stationary = ['stationary_1']
        self.freewalk = [
            'freewalk_1','freewalk_2','freewalk_3','freewalk_4','freewalk_5','freewalk_6','freewalk_7','freewalk_8','freewalk_9'
        ]
        self.walk_class = [('scripted_walk', self.scripted_walk), ('stationary', self.stationary), ('freewalk', self.freewalk)]

    def load_data(self, training_data_path, shuffle = True):
        data = pd.read_csv(training_data_path)
        X = data.iloc[:, 1:]
        y = data['label']
        y_adjusted = y - 1
        print(X.shape)

        # 进行one-hot编码
        one_hot_y = to_categorical(y_adjusted, num_classes=output_size)

        #shuffle
        indices = np.arange(y.shape[0])
        if shuffle:
            random_seed = 42  # 选择适当的随机种子
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        self.X = np.array(X)[indices]
        self.y = one_hot_y[indices]


    def build_model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Input(shape=(self.input_size,)))
        for size in self.hidden_sizes:
            model.add(tf.keras.layers.Dense(size, activation='relu'))
        model.add(tf.keras.layers.Dense(self.output_size, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
        
    def train_model(self, model_path, epochs=10, batch_size=32):
        checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
        self.history = self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpointer])
        self.plot_loss_and_accuracy_history(model_path)

    def plot_loss_and_accuracy_history(self, model_path):
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

        # 添加整张图的标题
        plt.suptitle(f"{model_path[:-3]} Training Curves")

        # 保存 loss 图
        plt.savefig("loss_and_accuracy.png")
        plt.clf()

    def predict(self, input_data):
        # Assuming input_data is a list of RSSI values for Beacon_1 to Beacon_7
        # input_data = [input_data]  # Scikit-Learn's predict method expects a 2D array
        return self.model.predict(input_data)
    
    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def generate_predictions(self, testing_data_path):
        for walk_str, walk_list in self.walk_class:
            prediction_results = {
                'label': [],
                'pred': []
            }
            for walk in walk_list:
                data_path = f"{testing_data_path}\\{walk}.csv"

                # 加載數據
                self.load_data(data_path, shuffle=False)

                # 進行預測
                predicted_labels = self.predict(self.X)
                predicted_labels = np.argmax(predicted_labels, axis=1) + 1  # 加 1 是为了将索引转换为 1 到 41 的标签
                label = np.argmax(self.y, axis=1) + 1
                # 將預測結果保存到 prediction_results 中
                prediction_results['label'].extend(label.tolist())
                prediction_results['pred'].extend(predicted_labels.tolist())
            df = pd.DataFrame(prediction_results)
            
            split_path = testing_data_path.split('\\')
            predictions_dir = f'predictions/{split_path[3]}'
            os.makedirs(predictions_dir, exist_ok=True)
            df.to_csv(os.path.join(predictions_dir, f'{walk_str}_predictions.csv'), index=False)

    def test(self, testing_data_path_list, model_path):
        self.load_model(model_path)
        for testing_data_path in testing_data_path_list:
            self.generate_predictions(testing_data_path)

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='DNN Indoor Localization')

    # 添加参数选项
    parser.add_argument('--training_data', type=str, help='csv file of the training data')
    parser.add_argument('--testing_data_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='path of .h5 file of model')

    # 解析命令行参数
    args = parser.parse_args()

    model_path = args.model_path
    # 在根据参数来执行相应的操作
    input_size = 7
    output_size = 41
    hidden_sizes = [8, 16, 32]
    dnn_model = DNN(input_size, output_size, hidden_sizes)
    if args.training_data:
        dnn_model.load_data(args.training_data)
        dnn_model.train_model(model_path, epochs=100)
    elif args.testing_data_list:
        testing_data_path_list = args.testing_data_list
        dnn_model.test(testing_data_path_list, model_path)
    else:
        print('Please specify --training_data or --test option.')