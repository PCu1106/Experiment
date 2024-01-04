'''
python .\DANN_AE.py \
    --training_source_domain_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv \
    --training_target_domain_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
    --model_path 220318_231116.h5 \
    --work_dir 220318_231116\343
python .\DANN_AE.py \
    --testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                        D:\Experiment\data\220318\GalaxyA51\routes \
                        D:\Experiment\data\231117\GalaxyA51\routes \
    --model_path 220318_231116.h5 \
    --work_dir 220318_231116\343
python ..\..\model_comparison\evaluator.py \
    --model_name DANN_AE \
    --directory 220318_231116\343_0.0 \
    --source_domain 220318 \
    --target_domain 231116
'''

import sys
sys.path.append('..\\DANN')
import argparse
from DANN import DANNModel
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.utils import plot_model
sys.path.append('..\\..\\model_comparison')
from walk_definitions import walk_class
from evaluator import Evaluator
sys.path.append('..')
from drop_out_plot import plot_lines



class AutoencoderDANNModel(DANNModel):
    def build_model(self):
        input_data = layers.Input(shape=self.input_shape, name='input_data')

        # Build Autoencoder
        autoencoder, feature_extractor = self.build_feature_extractor(input_data)

        # Build Label Predictor
        label_predictor_output = self.build_label_predictor(feature_extractor)

        # Build Domain Classifier
        domain_classifier_output = self.build_domain_classifier(feature_extractor)

        # Combine all outputs
        model_output = [label_predictor_output, domain_classifier_output, autoencoder.output]

        # Build the final model
        model = models.Model(inputs=input_data, outputs=model_output)

        return model
    
    def build_feature_extractor(self, input_data):
        # Autoencoder layers
        x = layers.Dense(8, activation='relu', name='encoder_1')(input_data)
        encoded = layers.Dense(16, activation='relu', name='encoder_ouput')(x)
        x = layers.Dense(8, activation='relu', name='decoder_1')(encoded)
        decoded = layers.Dense(self.input_shape, activation='sigmoid', name='decoder_output')(x)
        
        autoencoder = models.Model(inputs=input_data, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        autoencoder.summary()
        return autoencoder, encoded
    
    def train(self, model_path, batch_size=32, epochs=50):
        # Compile the AutoencoderDANN model with additional reconstruction loss
        self.model.compile(
            optimizer='adam',
            loss={
                'label_predictor_output': 'categorical_crossentropy',
                'domain_classifier_output': 'binary_crossentropy',
                'decoder_output': 'mse'  # Reconstruction loss
            },
            loss_weights={
                'label_predictor_output': 0.2,
                'domain_classifier_output': 0.4,
                'decoder_output': 0.4  # Adjust the weight for the reconstruction loss
            },
            metrics={
                'label_predictor_output': 'accuracy',
                'domain_classifier_output': 'accuracy'
            }
        )

        # Define the ModelCheckpoint callback to save the model with the minimum total loss
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        # Train the AutoencoderDANN model with validation split
        history = self.model.fit(
            self.X,
            {
                'label_predictor_output': self.yl,
                'domain_classifier_output': self.yd,
                'decoder_output': self.X  # Use the input data as target for reconstruction loss
            },
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[checkpoint]
        )

        # Plot training history
        self.plot_training_history(history, model_path)

        return history
    
    def plot_training_history(self, history, model_path):
        # Plot training and validation loss for Label Predictor
        plt.figure(figsize=(16, 12))

        # Subplot for Label Predictor Loss
        plt.subplot(3, 2, 1)
        plt.plot(history.history['label_predictor_output_loss'], label='Train Label Predictor Loss')
        plt.plot(history.history['val_label_predictor_output_loss'], label='Validation Label Predictor Loss')
        plt.title('Label Predictor Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Subplot for Label Predictor Accuracy
        plt.subplot(3, 2, 2)
        plt.plot(history.history['label_predictor_output_accuracy'], label='Train Label Predictor Accuracy')
        plt.plot(history.history['val_label_predictor_output_accuracy'], label='Validation Label Predictor Accuracy')
        plt.title('Label Predictor Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Subplot for Domain Classifier Loss
        plt.subplot(3, 2, 3)
        plt.plot(history.history['domain_classifier_output_loss'], label='Train Domain Classifier Loss')
        plt.plot(history.history['val_domain_classifier_output_loss'], label='Validation Domain Classifier Loss')
        plt.title('Domain Classifier Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Subplot for Domain Classifier Accuracy
        plt.subplot(3, 2, 4)
        plt.plot(history.history['domain_classifier_output_accuracy'], label='Train Domain Classifier Accuracy')
        plt.plot(history.history['val_domain_classifier_output_accuracy'], label='Validation Domain Classifier Accuracy')
        plt.title('Domain Classifier Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Subplot for Autoencoder Loss
        plt.subplot(3, 2, 5)
        plt.plot(history.history['decoder_output_loss'], label='Train Autoencoder Loss')
        plt.plot(history.history['val_decoder_output_loss'], label='Validation Autoencoder Loss')
        plt.title('Autoencoder Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Add a title for the entire figure
        plt.suptitle(f"{model_path[:-3]} Training Curves")
        plt.tight_layout()
        plt.savefig('loss_and_accuracy.png')

    
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

    domain1_result = []
    domain2_result = []
    domain3_result = []

    # 設定 input shape 和 num_classes
    input_shape = 7
    num_classes = 41  # 這裡的數字要根據你的問題設定
    batch_size=32
    epochs=500
    data_drop_out_list = np.arange(0.0, 1.05, 0.1)
    
    for data_drop_out in data_drop_out_list:
        # 創建 DANNModel    
        dann_model = AutoencoderDANNModel(input_shape, num_classes, f'{args.work_dir}_{data_drop_out:.1f}')
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
            domain1_result.append(mde_list[0][1])
            domain2_result.append(mde_list[1][1])
            domain3_result.append(mde_list[2][1])
        elif args.fine_tune_data:
            dann_model.load_data(args.fine_tune_data, one_file=True)
            dann_model.fine_tune(args.model_path, batch_size, epochs)
        else:
            print('Please specify --training_source_domain_data/--training_target_domain_data or --testing_data_list option.')

        os.chdir('..\\..')
    
    if args.testing_data_list:
        plot_lines(data_drop_out_list, domain1_result, domain2_result, domain3_result, args.work_dir, 'Source_domain_to_Target_domain')