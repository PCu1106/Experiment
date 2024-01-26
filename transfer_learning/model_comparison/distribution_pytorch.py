'''
python .\distribution_pytorch.py \
    --training_source_domain_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv \
    --training_target_domain_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
    --model_path 220318_231116.pth  \
    --work_dir ..\DANN_CORR\220318_231116\0.1_10_0.9
    --experiment_name time_variation
'''

import sys
sys.path.append('..\\DANN_CORR')
import argparse
from DANN_CORR import HistCorrDANNModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sys.path.append('..\\..\\model_comparison')
from walk_definitions import date2color

def extract_features(dataset, model):
    features_list = []
    labels_list = []
    
    for data, label in dataset:
        # Use your model to extract features
        features, _ = model(data)
        features_list.append(features.detach().numpy())
        labels_list.append(label.item())
    
    return features_list, labels_list

# Modify the __main__ block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
    parser.add_argument('--model_path', type=str, default='my_model.h5', help='path of .h5 file of model')
    parser.add_argument('--work_dir', type=str, default='DANN', help='create new directory to save result')
    parser.add_argument('--experiment_name', type=str, default='', required = True, help='Would show on the pigure name')

    args = parser.parse_args()


    dann_model = HistCorrDANNModel(model_save_path=args.model_path, work_dir=f'{args.work_dir}')
    dann_model.load_train_data(args.training_source_domain_data, args.training_target_domain_data)
    dann_model.load_model(args.model_path)


    # Extract features for source and target datasets
    source_features, source_labels = extract_features(dann_model.source_dataset, dann_model.domain_adaptation_model)
    target_features, target_labels = extract_features(dann_model.target_dataset, dann_model.domain_adaptation_model)
    # Convert features to a NumPy array for ease of slicing
    source_features = np.array(source_features)
    target_features = np.array(target_features)

    # Convert features to a pandas DataFrame for ease of plotting
    source_df_original = pd.DataFrame(dann_model.source_dataset.data, columns=['label', 'Beacon_1', 'Beacon_2', 'Beacon_3', 'Beacon_4', 'Beacon_5', 'Beacon_6', 'Beacon_7'])

    choosen_feature = [4, 9]

    # Choose two features for plotting (e.g., Feature_1 and Feature_2)
    source_df_features = pd.DataFrame(source_features[:, choosen_feature], columns=['Feature_1', 'Feature_2'])
    source_df_features['label'] = source_labels

    target_df_original = pd.DataFrame(dann_model.target_dataset.data, columns=['label', 'Beacon_1', 'Beacon_2', 'Beacon_3', 'Beacon_4', 'Beacon_5', 'Beacon_6', 'Beacon_7'])

    # Choose two features for plotting (e.g., Feature_1 and Feature_2)
    target_df_features = pd.DataFrame(target_features[:, choosen_feature], columns=['Feature_1', 'Feature_2'])
    target_df_features['label'] = target_labels

    if args.experiment_name == 'time_variation':
        color_list = [date2color['220318'], date2color['231116']]
    elif args.experiment_name == 'spatial_variation':
        color_list = [date2color['231116'], date2color['231117']]

    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    # Scatter plot for Original Source Domain
    axes[0].scatter(source_df_original[source_df_original['label'] == 1]['Beacon_1'], source_df_original[source_df_original['label'] == 1]['Beacon_3'], label='Source Domain (Circle)', color=color_list[0], marker='o')
    axes[0].scatter(source_df_original[source_df_original['label'] == 41]['Beacon_1'], source_df_original[source_df_original['label'] == 41]['Beacon_3'], label='Source Domain (Triangle)', color=color_list[0], marker='^')
    axes[0].scatter(target_df_original[target_df_original['label'] == 1]['Beacon_1'], target_df_original[target_df_original['label'] == 1]['Beacon_3'], label='Target Domain (Circle)', color=color_list[1], marker='o')
    axes[0].scatter(target_df_original[target_df_original['label'] == 41]['Beacon_1'], target_df_original[target_df_original['label'] == 41]['Beacon_3'], label='Target Domain (Triangle)', color=color_list[1], marker='^')
    axes[0].set_title('Original Source Domain')
    axes[0].set_xlabel('Beacon_1')
    axes[0].set_ylabel('Beacon_2')
    axes[0].legend()

    # Scatter plot for Original Target Domain
    axes[1].scatter(source_df_features[source_df_features['label'] == 1]['Feature_1'], source_df_features[source_df_features['label'] == 1]['Feature_2'], label='Source Domain (Circle)', color=color_list[0], marker='o')
    axes[1].scatter(source_df_features[source_df_features['label'] == 40]['Feature_1'], source_df_features[source_df_features['label'] == 40]['Feature_2'], label='Source Domain (Triangle)', color=color_list[0], marker='^')
    axes[1].scatter(target_df_features[target_df_features['label'] == 1]['Feature_1'], target_df_features[target_df_features['label'] == 1]['Feature_2'], label='Target Domain (Circle)', color=color_list[1], marker='o')
    axes[1].scatter(target_df_features[target_df_features['label'] == 40]['Feature_1'], target_df_features[target_df_features['label'] == 40]['Feature_2'], label='Target Domain (Triangle)', color=color_list[1], marker='^')
    axes[1].set_title('Original Target Domain')
    axes[1].set_xlabel('Beacon_1')
    axes[1].set_ylabel('Beacon_2')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{args.experiment_name}_scatter.png')