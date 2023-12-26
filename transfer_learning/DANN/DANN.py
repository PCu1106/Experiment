'''
python .\DANN.py \
    --training_source_domain_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
    --training_target_domain_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv \
    --model_path 231116_220318.h5 \
    --work_dir 1layerLP\231116_220318\1_0
python .\DANN.py \
    --testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                        D:\Experiment\data\220318\GalaxyA51\routes \
                        D:\Experiment\data\231117\GalaxyA51\routes \
    --model_path 231116_220318.h5 \
    --work_dir 1layerLP\231116_220318\1_0
python .\DANN.py \
    --fine_tune_data D:\Experiment\data\231117\GalaxyA51\wireless_training.csv \
    --model_path D:\Experiment\transfer_learning\DANN\231116_220318\1_3\231116_220318.h5 \
    --work_dir 231116_220318_231117
python ..\..\model_comparison\evaluator.py \
    --model_name DANN \
    --directory 220318_231116\12_0.0 \
    --source_domain 220318 \
    --target_domain 231116
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.utils import plot_model
import sys
sys.path.append('..\\..\\model_comparison')
from walk_definitions import walk_class
from evaluator import Evaluator

def plot_lines(data_drop_out_list, target_domain1, source_domain, target_domain2, output_path, title):
    plt.plot(data_drop_out_list, target_domain1, marker='o', label='Target Domain1', color='blue')
    # plt.plot(freeze_layers, source_domain, marker='o', label='Source Domain', color='orange')
    # plt.plot(freeze_layers, target_domain2, marker='o', label='Target Domain2', color='green')

    plt.xlabel('data dropout ratio')
    plt.ylabel('MDE (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(data_drop_out_list, labels=[f'{x:.1f}' for x in data_drop_out_list])
    plt.ylim(0, 3)

    # 在每個點的上方顯示數字
    for x, y1, y2, y3 in zip(data_drop_out_list, target_domain1, source_domain, target_domain2):
        plt.text(x, y1, f'{y1:.3f}', ha='center', va='bottom')
        # plt.text(x, y2, f'{y2:.3f}', ha='center', va='bottom')
        # plt.text(x, y3, f'{y3:.3f}', ha='center', va='bottom')
    
    plt.savefig(output_path)

@tf.custom_gradient
def GradientReversalOperator(x):
	def grad(dy):
		return -1 * dy
	return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):
	def __init__(self):
		super(GradientReversalLayer, self).__init__()
		
	def call(self, inputs):
		return GradientReversalOperator(inputs)

class DANNModel:
    def __init__(self, input_shape, num_classes, work_dir):
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        os.chdir(work_dir)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        # 注册自定义层
        tf.keras.utils.get_custom_objects()['GradientReversalLayer'] = GradientReversalLayer

    def load_data(self, source_domain_file, target_domain_file = None, data_drop_out = None, shuffle = True, one_file = False):
        if one_file: # test or fine-tune
            data = pd.read_csv(source_domain_file)
            X = data.iloc[:, 1:]
            y = data['label']
            y_adjusted = y - 1
            one_hot_y = to_categorical(y_adjusted, num_classes=self.num_classes)

            #shuffle
            indices = np.arange(y.shape[0])
            if shuffle:
                random_seed = 42  # 选择适当的随机种子
                np.random.seed(random_seed)
                np.random.shuffle(indices)

            self.X = np.array(X)[indices]
            self.yl = one_hot_y[indices]
            self.yd = np.ones(len(self.X)) # for fine-tune
        else: # train
            # Read source domain data
            source_domain = pd.read_csv(source_domain_file)
            self.source_domain_data = source_domain.iloc[:, 1:]
            source_domain_labels = source_domain['label']
            source_domain_labels = source_domain_labels - 1
            self.source_domain_labels = to_categorical(source_domain_labels, num_classes=self.num_classes)

            # Read target domain data
            target_domain = pd.read_csv(target_domain_file)
            self.target_domain_data = target_domain.iloc[:, 1:]
            target_domain_labels = target_domain['label']
            target_domain_labels = target_domain_labels - 1
            self.target_domain_labels = to_categorical(target_domain_labels, num_classes=self.num_classes)

            if data_drop_out is not None:
                drop_indices = np.random.choice(len(self.target_domain_data), int(len(self.target_domain_data) * data_drop_out), replace=False)
                self.target_domain_data = np.delete(self.target_domain_data, drop_indices, axis=0)
                self.target_domain_labels = np.delete(self.target_domain_labels, drop_indices, axis=0)

            # Combine source and target domain data and labels
            combined_data = np.vstack([self.source_domain_data, self.target_domain_data])
            combined_labels = np.vstack([self.source_domain_labels, self.target_domain_labels])

            # Create domain labels (1 for source domain, 0 for target domain)
            combined_domain_labels = np.concatenate([np.ones(len(self.source_domain_data)), np.zeros(len(self.target_domain_data))])
            if shuffle:
                # Shuffle the data
                indices = np.arange(len(combined_data))
                np.random.shuffle(indices)
                combined_data = combined_data[indices]
                combined_labels = combined_labels[indices]
                combined_domain_labels = combined_domain_labels[indices]
            self.X = combined_data
            self.yl = combined_labels
            self.yd = combined_domain_labels

    def add_noise_to_data(self, noise_level=0.02):
        noise = np.random.normal(loc=0, scale=noise_level, size=self.X.shape)
        self.X = self.X + noise

    def build_model(self):
        # Input layer
        input_data = layers.Input(shape=self.input_shape, name='input_data')

        # Feature extractor
        feature_extractor = self.build_feature_extractor(input_data)

        # Label predictor
        label_predictor = self.build_label_predictor(feature_extractor)

        # Domain classifier
        domain_classifier = self.build_domain_classifier(feature_extractor)

        # Build DANN model
        dann_model = models.Model(inputs=input_data, outputs=[label_predictor, domain_classifier])

        return dann_model

    def build_feature_extractor(self, input_data):
        # Shared feature extraction layers
        x = layers.Dense(8, activation='relu', name='feature_extractor_1')(input_data)
        x = layers.Dense(16, activation='relu', name='feature_extractor_2')(x)

        return x

    def build_label_predictor(self, feature_extractor):
        # Label predictor layers
        x = layers.Dense(32, activation='relu', name='label_predictor_1')(feature_extractor)
        label_predictor_output = layers.Dense(self.num_classes, activation='softmax', name='label_predictor_output')(x)

        return label_predictor_output

    def build_domain_classifier(self, feature_extractor):
        # Domain classifier layers
        x = GradientReversalLayer()(feature_extractor)
        x = layers.Dense(8, activation='relu', name='domain_classifier_1')(x)
        domain_classifier_output = layers.Dense(1, activation='sigmoid', name='domain_classifier_output')(x)

        return domain_classifier_output

    def plot_training_history(self, history, model_path):
        # Plot training and validation loss for Label Predictor
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history.history['label_predictor_output_loss'], label='Train Label Predictor Loss')
        plt.plot(history.history['val_label_predictor_output_loss'], label='Validation Label Predictor Loss')
        plt.title('Label Predictor Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history.history['label_predictor_output_accuracy'], label='Train Label Predictor Accuracy')
        plt.plot(history.history['val_label_predictor_output_accuracy'], label='Validation Label Predictor Accuracy')
        plt.title('Label Predictor Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot training and validation loss for Domain Classifier
        plt.subplot(2, 2, 3)
        plt.plot(history.history['domain_classifier_output_loss'], label='Train Domain Classifier Loss')
        plt.plot(history.history['val_domain_classifier_output_loss'], label='Validation Domain Classifier Loss')
        plt.title('Domain Classifier Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(history.history['domain_classifier_output_accuracy'], label='Train Domain Classifier Accuracy')
        plt.plot(history.history['val_domain_classifier_output_accuracy'], label='Validation Domain Classifier Accuracy')
        plt.title('Domain Classifier Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # 添加整张图的标题
        plt.suptitle(f"{model_path[:-3]} Training Curves")
        plt.tight_layout()
        plt.savefig('loss_and_accuracy.png')

    def train(self, model_path, batch_size=32, epochs=50):
        # Compile the DANN model
        self.model.compile(optimizer='adam',
                           loss={'label_predictor_output': 'categorical_crossentropy', 'domain_classifier_output': 'binary_crossentropy'},
                           loss_weights={'label_predictor_output': 0.5, 'domain_classifier_output': 0.5},
                           metrics={'label_predictor_output': 'accuracy', 'domain_classifier_output': 'accuracy'})

        # Define the ModelCheckpoint callback to save the model with the minimum total loss
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        # Train the DANN model with validation split
        history = self.model.fit(self.X, {'label_predictor_output': self.yl, 'domain_classifier_output': self.yd},
                                 batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[checkpoint])

        # Plot training history
        self.plot_training_history(history, model_path)

        return history
    
    def predict(self, input_data):
        # Assuming input_data is a list of RSSI values for Beacon_1 to Beacon_7
        # input_data = [input_data]  # Scikit-Learn's predict method expects a 2D array
        return self.model.predict(input_data)
    
    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def load_weight(self, model_path):
        self.model.load_weights(model_path)

    def generate_predictions(self, model_path):
        self.load_model(model_path)
        prediction_results = {
            'label': [],
            'pred': []
        }
        # 進行預測
        predicted_labels = self.predict(self.X)[0]
        predicted_labels = np.argmax(predicted_labels, axis=1) + 1  # 加 1 是为了将索引转换为 1 到 41 的标签
        label = np.argmax(self.yl, axis=1) + 1
        # 將預測結果保存到 prediction_results 中
        prediction_results['label'].extend(label.tolist())
        prediction_results['pred'].extend(predicted_labels.tolist())
        
        return pd.DataFrame(prediction_results)
    
    def fine_tune(self, model_path, batch_size=32, epochs=50):
        # 加载预训练模型
        self.load_weight(model_path)
        self.model.summary()
        # 冻结参数
        for layer in self.model.layers:
            if layer.name.startswith('feature_extractor'):
                layer.trainable = False
            elif layer.name.startswith('domain_classifier'):
                layer.trainable = False
        

        # 编译模型，仅优化 label predictor 部分
        self.model.compile(optimizer='adam',
                           loss={'label_predictor_output': 'categorical_crossentropy', 'domain_classifier_output': 'binary_crossentropy'},
                           loss_weights={'label_predictor_output': 0.33, 'domain_classifier_output': 0.67},
                           metrics={'label_predictor_output': 'accuracy', 'domain_classifier_output': 'accuracy'})

        # 定义 ModelCheckpoint 回调以保存 fine-tuned 模型
        fine_tune_checkpoint = ModelCheckpoint('fine_tuned_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        # 在 fine_tune_data 上进行 fine-tune
        fine_tune_history = self.model.fit(self.X, {'label_predictor_output': self.yl, 'domain_classifier_output': self.yd},
                                 batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[fine_tune_checkpoint])
         
        # 绘制 fine-tune 的训练历史
        self.plot_training_history(fine_tune_history, 'fine_tuned_model.h5')

        return fine_tune_history
    

if __name__ == "__main__":
    # 使用 argparse 處理命令列參數
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
    parser.add_argument('--testing_data_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--fine_tune_data', type=str, help='Path to the fine-tune data file')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='path of .h5 file of model')
    parser.add_argument('--work_dir', type=str, default='DANN', help='create new directory to save result')
    parser.add_argument('--noise', action='store_true', default=False, help='add noise or not')

    args = parser.parse_args()

    target_domain1_result = []
    source_domain_reult = []
    target_domain2_result = []

    # 設定 input shape 和 num_classes
    input_shape = (7,)
    num_classes = 41  # 這裡的數字要根據你的問題設定
    batch_size=32
    epochs=1000
    data_drop_out_list = np.arange(0.0, 0.1, 0.1)
    
    for data_drop_out in data_drop_out_list:
        # 創建 DANNModel    
        dann_model = DANNModel(input_shape, num_classes, f'{args.work_dir}_{data_drop_out:.1f}')
        plot_model(dann_model.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # 讀取資料
        if args.training_source_domain_data and args.training_target_domain_data:
            dann_model.load_data(args.training_source_domain_data, args.training_target_domain_data, data_drop_out)
            # 訓練模型
            history = dann_model.train(args.model_path, batch_size, epochs)
        elif args.testing_data_list:
            testing_data_path_list = args.testing_data_list
            for testing_data_path in testing_data_path_list:
                for walk_str, walk_list in walk_class:
                    prediction_results = pd.DataFrame()
                    for walk in walk_list:
                        # 加載數據
                        dann_model.load_data(f"{testing_data_path}\\{walk}.csv", shuffle=False, one_file=True)
                        if args.noise:
                            dann_model.add_noise_to_data()
                        results = dann_model.generate_predictions(args.model_path)
                        prediction_results = pd.concat([prediction_results, results], ignore_index=True)
                    split_path = testing_data_path.split('\\')
                    predictions_dir = f'predictions/{split_path[3]}'
                    os.makedirs(predictions_dir, exist_ok=True)
                    prediction_results.to_csv(os.path.join(predictions_dir, f'{walk_str}_predictions.csv'), index=False)
            predicion_data_path_list = os.listdir('predictions/')
            evaluator = Evaluator()
            dir = 'noise' if args.noise else None
            mde_list = evaluator.test(predicion_data_path_list, f'{args.work_dir}_{data_drop_out}', dir)
            target_domain1_result.append(mde_list[0][1])
            source_domain_reult.append(mde_list[1][1])
            target_domain2_result.append(mde_list[2][1])
        elif args.fine_tune_data:
            dann_model.load_data(args.fine_tune_data, one_file=True)
            dann_model.fine_tune(args.model_path, batch_size, epochs)
        else:
            print('Please specify --training_source_domain_data/--training_target_domain_data or --testing_data_list option.')

        os.chdir('..\\..\\..')
    
    if args.testing_data_list:
        plot_lines(data_drop_out_list, target_domain1_result, source_domain_reult, target_domain2_result, 'Dropout_Data.png', 'Source_domain_to_Target_domain1')