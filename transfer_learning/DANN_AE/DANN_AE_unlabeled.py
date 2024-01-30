'''
python .\DANN_AE_unlabeled.py \
    --training_source_domain_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv \
    --training_target_domain_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
    --model_path 220318_231116.h5 \
    --work_dir unlabeled\220318_231116
python .\DANN_AE_unlabeled.py \
    --testing_data_list D:\Experiment\data\220318\GalaxyA51\routes \
                        D:\Experiment\data\231116\GalaxyA51\routes \
                        D:\Experiment\data\231117\GalaxyA51\routes \
    --model_path 220318_231116.h5 \
    --work_dir unlabeled\220318_231116
python ..\..\model_comparison\evaluator.py \
    --model_name DANN \
    --directory 220318_231116\12_0.0 \
    --source_domain 220318 \
    --target_domain 231116
'''
from DANN_AE import AutoencoderDANNModel
import argparse
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('..\\..\\model_comparison')
from walk_definitions import walk_class
from evaluator import Evaluator
sys.path.append('..\\model_comparison')
from drop_out_plot import plot_lines

def combine_history(history1, history2):
    # Convert History objects to dictionaries
    if isinstance(history1, tf.keras.callbacks.History):
        history1 = history1.history
    if isinstance(history2, tf.keras.callbacks.History):
        history2 = history2.history

    # Combine the histories
    combined_history = {
        'loss': history1['loss'] + history2['loss'],
        'label_predictor_output_loss': history1['label_predictor_output_loss'] + history2['label_predictor_output_loss'],
        'domain_classifier_output_loss': history1['domain_classifier_output_loss'] + history2['domain_classifier_output_loss'],
        'decoder_output_loss': history1['decoder_output_loss'] + history2['decoder_output_loss'],
        'label_predictor_output_accuracy': history1['label_predictor_output_accuracy'] + history2['label_predictor_output_accuracy'],
        'domain_classifier_output_accuracy': history1['domain_classifier_output_accuracy'] + history2['domain_classifier_output_accuracy'],
        'val_loss': history1['val_loss'] + history2['val_loss'],
        'val_label_predictor_output_loss': history1['val_label_predictor_output_loss'] + history2['val_label_predictor_output_loss'],
        'val_domain_classifier_output_loss': history1['val_domain_classifier_output_loss'] + history2['val_domain_classifier_output_loss'],
        'val_decoder_output_loss': history1['val_decoder_output_loss'] + history2['val_decoder_output_loss'],
        'val_label_predictor_output_accuracy': history1['val_label_predictor_output_accuracy'] + history2['val_label_predictor_output_accuracy'],
        'val_domain_classifier_output_accuracy': history1['val_domain_classifier_output_accuracy'] + history2['val_domain_classifier_output_accuracy'],
    }

    # Create a dummy History object
    combined_history_object = tf.keras.callbacks.History()
    combined_history_object.history = combined_history

    return combined_history_object

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
    epochs=5
    data_drop_out_list = np.arange(0.9, 0.95, 0.1)
    
    for data_drop_out in data_drop_out_list:
        # 創建 DANNModel    
        dann_model = AutoencoderDANNModel(input_shape, num_classes, f'{args.work_dir}_{data_drop_out:.1f}')
        plot_model(dann_model.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # 讀取資料
        if args.training_source_domain_data and args.training_target_domain_data:
            for i in range(10):
                print(f'epoch: {i}')
                dann_model.load_data(args.training_source_domain_data, args.training_target_domain_data, data_drop_out)
                history1 = dann_model.train_step_1(args.model_path, batch_size, epochs=50)
                if i == 0:
                    combined_history = history1
                else:
                    combined_history = combine_history(combined_history, history1)
                dann_model.load_data(args.training_source_domain_data, one_file=True)
                history2 = dann_model.train_step_2(args.model_path, batch_size, epochs=50)
                combined_history = combine_history(combined_history, history2)
            # dann_model.load_data(args.training_source_domain_data, args.training_target_domain_data, data_drop_out)
            # history1 = dann_model.train_step_1(args.model_path, batch_size, epochs)
            # dann_model.load_data(args.training_source_domain_data, one_file=True)
            # history2 = dann_model.train_step_2(args.model_path, batch_size, epochs)
            # combined_history = combine_history(history1, history2)
            dann_model.plot_training_history(combined_history, args.model_path)

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