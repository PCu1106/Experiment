'''
python .\distribution.py \
    --training_source_domain_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv \
    --training_target_domain_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
    --model_path 220318_231116.h5  \
    --work_dir ..\DANN\220318_231116\12_0.9
'''

import sys
sys.path.append('..\\DANN')
sys.path.append('..\\DANN_AE')
sys.path.append('..\\DANN_CORR')
import argparse
from DANN import DANNModel
from DANN_AE import AutoencoderDANNModel
from DANN_CORR import HistCorrDANNModel
import matplotlib.pyplot as plt

# Modify the __main__ block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='path of .h5 file of model')
    parser.add_argument('--work_dir', type=str, default='DANN', help='create new directory to save result')
    args = parser.parse_args()

    input_shape = 7
    num_classes = 41  # 這裡的數字要根據你的問題設定
    # dann_model = DANNModel(input_shape, num_classes, f'{args.work_dir}')
    # dann_model = AutoencoderDANNModel(input_shape, num_classes, f'{args.work_dir}')
    dann_model = HistCorrDANNModel(input_shape, num_classes, f'{args.work_dir}')
    dann_model.load_model(args.model_path)

    # Extract features for label 1 in source domain data before feature extractor
    dann_model.load_data(args.training_source_domain_data, shuffle=False, one_file=True)
    source_domain_data_label1 = dann_model.X[dann_model.yl[:, 0] == 1]

    # Extract features for label 41 in source domain data before feature extractor
    dann_model.load_data(args.training_source_domain_data, shuffle=False, one_file=True)
    source_domain_data_label41 = dann_model.X[dann_model.yl[:, 40] == 1]

    # Extract features for label 1 in target domain data before feature extractor
    dann_model.load_data(args.training_target_domain_data, shuffle=False, one_file=True)
    target_domain_data_label1 = dann_model.X[dann_model.yl[:, 0] == 1]

    # Extract features for label 41 in target domain data before feature extractor
    dann_model.load_data(args.training_target_domain_data, shuffle=False, one_file=True)
    target_domain_data_label41 = dann_model.X[dann_model.yl[:, 40] == 1]

    # Plot 2D scatter plot before feature extractor
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.scatter(source_domain_data_label1[:, 0], source_domain_data_label1[:, 2], c='blue', marker='o', label='Source Domain (Label 1)')
    plt.scatter(source_domain_data_label41[:, 0], source_domain_data_label41[:, 2], c='blue', marker='^', label='Source Domain (Label 41)')
    plt.scatter(target_domain_data_label1[:, 0], target_domain_data_label1[:, 2], c='orange', marker='o', label='Target Domain (Label 1)')
    plt.scatter(target_domain_data_label41[:, 0], target_domain_data_label41[:, 2], c='orange', marker='^', label='Target Domain (Label 41)')
    plt.title('2D Scatter Plot Before Feature Extractor')
    plt.xlabel('Beacon_1')
    plt.ylabel('Beacon_2')
    plt.legend()

    # Extract features for label 1 in source domain data after feature extractor
    source_domain_features_label1 = dann_model.extract_features(source_domain_data_label1)

    # Extract features for label 41 in source domain data after feature extractor
    source_domain_features_label41 = dann_model.extract_features(source_domain_data_label41)

    # Extract features for label 1 in target domain data after feature extractor
    target_domain_features_label1 = dann_model.extract_features(target_domain_data_label1)

    # Extract features for label 41 in target domain data after feature extractor
    target_domain_features_label41 = dann_model.extract_features(target_domain_data_label41)

    # Plot 2D scatter plot after feature extractor
    plt.subplot(1, 2, 2)
    plt.scatter(source_domain_features_label1[:, 4], source_domain_features_label1[:, 7], c='blue', marker='o', label='Source Domain (Label 1)')
    plt.scatter(source_domain_features_label41[:, 4], source_domain_features_label41[:, 7], c='blue', marker='^', label='Source Domain (Label 41)')
    plt.scatter(target_domain_features_label1[:, 4], target_domain_features_label1[:, 7], c='orange', marker='o', label='Target Domain (Label 1)')
    plt.scatter(target_domain_features_label41[:, 4], target_domain_features_label41[:, 7], c='orange', marker='^', label='Target Domain (Label 41)')
    plt.title('2D Scatter Plot After Feature Extractor')
    plt.xlabel('Feature from Beacon_1')
    plt.ylabel('Feature from Beacon_2')
    plt.legend()

    plt.tight_layout()
    plt.savefig('scatter.png')