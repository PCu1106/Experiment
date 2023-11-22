'''
python .\AutoEncoder.py \
--training_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
--model_path 231116.h5

python .\AutoEncoder.py \
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
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from walk_definitions import walk_class

class AutoEncoder:
    def __init__(self, input_size=7, code_size=8, work_dir='AE'):
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        os.chdir(work_dir)
        self.input_size = input_size
        self.code_size = code_size
        self.model = self.build_model()
        self.history = None
        
    def load_data(self, training_data_path, shuffle = True):
        data = pd.read_csv(training_data_path)
        X = data.iloc[:, 1:]

        #shuffle
        indices = np.arange(X.shape[0])
        if shuffle:
            random_seed = 42  # 选择适当的随机种子
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        self.X = np.array(X)[indices]


    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=(self.input_size,))
    
        # 编码器
        encoded = tf.keras.layers.Dense(16, activation='relu')(input_layer)
        encoded = tf.keras.layers.Dense(self.code_size, activation='relu')(encoded)
        
        # 解码器
        decoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(self.input_size, activation='sigmoid')(decoded)
        
        autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        
        return autoencoder
        
    def train_model(self, model_path, epochs=10, batch_size=32):
        checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
        self.history = self.model.fit(self.X, self.X, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpointer])
        self.plot_loss_and_accuracy_history(model_path)

    def plot_loss_and_accuracy_history(self, model_path):
        plt.figure(figsize=(12, 6))

        # 画 loss 图
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()

        # 添加整张图的标题
        plt.suptitle(f"{model_path[:-3]} Training Curves")

        # 保存 loss 图
        plt.savefig("loss.png")
        plt.clf()

    def predict(self, input_data):
        # Assuming input_data is a list of RSSI values for Beacon_1 to Beacon_7
        # input_data = [input_data]  # Scikit-Learn's predict method expects a 2D array
        return self.model.predict(input_data)
    
    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def getEncoder(self):
        return tf.keras.models.Model(inputs=self.model.input, outputs=self.model.layers[2].output)

    def getDecoder(self):
        encoded_input = tf.keras.layers.Input(shape=(8,))  # Assuming the code layer has size 8
        decoder_layer = self.model.layers[-2](encoded_input)  # Use the second-to-last layer for decoding
        decoder_layer = self.model.layers[-1](decoder_layer)  # Use the last layer for decoding
        decoder = tf.keras.models.Model(inputs=encoded_input, outputs=decoder_layer)
        return decoder


    def generate_predictions(self, model_path):
        self.load_model(model_path)
        prediction_results = {
            'input': [],
            'code': [],
            'ouput': []
        }
        # 進行預測
        code = self.getEncoder().predict(self.X)
        decoded = self.getDecoder().predict(code)
        # 將預測結果保存到 prediction_results 中
        prediction_results['input'].extend(self.X.tolist())
        prediction_results['code'].extend(code.tolist())
        prediction_results['ouput'].extend(decoded.tolist())
        return pd.DataFrame(prediction_results)


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='AutoEncoder Indoor Localization')

    # 添加参数选项
    parser.add_argument('--training_data', type=str, help='csv file of the training data')
    parser.add_argument('--testing_data_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='path of .h5 file of model')
    parser.add_argument('--work_dir', type=str, default='AE', help='create new directory to save result')


    # 解析命令行参数
    args = parser.parse_args()

    model_path = args.model_path
    # 在根据参数来执行相应的操作
    input_size = 7
    code_size = 8
    ae_model = AutoEncoder(input_size, code_size, args.work_dir)
    if args.training_data:
        ae_model.load_data(args.training_data)
        ae_model.train_model(model_path, epochs=500)
    elif args.testing_data_list:
        testing_data_path_list = args.testing_data_list
        for testing_data_path in testing_data_path_list:
            for walk_str, walk_list in walk_class:
                prediction_results = pd.DataFrame()
                for walk in walk_list:
                    # 加載數據
                    ae_model.load_data(f"{testing_data_path}\\{walk}.csv", shuffle=False)
                    results = ae_model.generate_predictions(model_path)
                    prediction_results = pd.concat([prediction_results, results], ignore_index=True)
                split_path = testing_data_path.split('\\')
                predictions_dir = f'predictions/{split_path[3]}'
                os.makedirs(predictions_dir, exist_ok=True)
                prediction_results.to_csv(os.path.join(predictions_dir, f'{walk_str}_predictions.csv'), index=False)
    else:
        print('Please specify --training_data or --test option.')