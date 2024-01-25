'''
python .\DNN.py \
    --training_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv \
    --model_path 220318.h5 \
    --work_dir 220318\DNN 

python .\DNN.py \
    --testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                        D:\Experiment\data\220318\GalaxyA51\routes \
                        D:\Experiment\data\231117\GalaxyA51\routes \
    --model_path 220318.h5 \
    --work_dir 220318\DNN 
'''

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from walk_definitions import walk_class

class DNN:
    def __init__(self, input_size=7, output_size=41, hidden_sizes=[8, 16, 32], work_dir='.'):
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        os.chdir(work_dir)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.model = self.build_model()
        self.history = None

    def load_data(self, training_data_path, shuffle = True):
        data = pd.read_csv(training_data_path)
        X = data.iloc[:, 1:]
        y = data['label']
        y_adjusted = y - 1
        print(X.shape)

        # 进行one-hot编码
        one_hot_y = to_categorical(y_adjusted, num_classes=self.output_size)

        #shuffle
        indices = np.arange(y.shape[0])
        if shuffle:
            random_seed = 42  # 选择适当的随机种子
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        self.X = np.array(X)[indices]
        self.y = one_hot_y[indices]

    def load_two_data(self, source_domain_file, target_domain_file, shuffle = True, data_drop_out = None):
        # Read source domain data
        source_domain = pd.read_csv(source_domain_file)
        self.source_domain_data = source_domain.iloc[:, 1:]
        source_domain_labels = source_domain['label']
        source_domain_labels = source_domain_labels - 1
        self.source_domain_labels = to_categorical(source_domain_labels, num_classes=self.output_size)

        # Read target domain data
        target_domain = pd.read_csv(target_domain_file)
        self.target_domain_data = target_domain.iloc[:, 1:]
        target_domain_labels = target_domain['label']
        target_domain_labels = target_domain_labels - 1
        self.target_domain_labels = to_categorical(target_domain_labels, num_classes=self.output_size)

        if data_drop_out is not None:
            drop_indices = np.random.choice(len(self.target_domain_data), int(len(self.target_domain_data) * data_drop_out), replace=False)
            self.target_domain_data = np.delete(self.target_domain_data, drop_indices, axis=0)
            self.target_domain_labels = np.delete(self.target_domain_labels, drop_indices, axis=0)

        # Combine source and target domain data and labels
        combined_data = np.vstack([self.source_domain_data, self.target_domain_data])
        combined_labels = np.vstack([self.source_domain_labels, self.target_domain_labels])

        if shuffle:
            # Shuffle the data
            indices = np.arange(len(combined_data))
            np.random.shuffle(indices)
            combined_data = combined_data[indices]
            combined_labels = combined_labels[indices]
        self.X = combined_data
        self.y = combined_labels

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

    def build_transfer_model(self, freeze_layers=None):
        # 如果未指定 freeze_layers，則預設全部可訓練 (weight initialization)
        if freeze_layers is None:
            freeze_layers = 0

        # 凍結指定數量的層，如果freeze_layers超過上限，會直接變成到最後一層
        for layer in self.model.layers[:freeze_layers]:
            layer.trainable = False
        for layer in self.model.layers[freeze_layers:]:
            layer.trainable = True

        transfer_model = tf.keras.Sequential()
        transfer_model.add(self.model)

        # 編譯模型
        transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = transfer_model


    def generate_predictions(self, model_path):
        self.load_model(model_path)
        prediction_results = {
            'label': [],
            'pred': []
        }
        # 進行預測
        predicted_labels = self.predict(self.X)
        predicted_labels = np.argmax(predicted_labels, axis=1) + 1  # 加 1 是为了将索引转换为 1 到 41 的标签
        label = np.argmax(self.y, axis=1) + 1
        # 將預測結果保存到 prediction_results 中
        prediction_results['label'].extend(label.tolist())
        prediction_results['pred'].extend(predicted_labels.tolist())
        
        return pd.DataFrame(prediction_results)


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='DNN Indoor Localization')

    # 添加参数选项
    parser.add_argument('--training_data', type=str, help='csv file of the training data')
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
    dnn_model = DNN(input_size, output_size, hidden_sizes, args.work_dir)
    if args.training_data:
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