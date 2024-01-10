'''
python .\test.py --source_domain_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv --target_domain_data 
D:\Experiment\data\231116\GalaxyA51\wireless_training.csv
'''

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import argparse
import cv2


parser = argparse.ArgumentParser(description='test for corr')
parser.add_argument('--source_domain_data', type=str, required=True, help='Path to the source domain data file')
parser.add_argument('--target_domain_data', type=str, required=True, help='Path to the target domain data file')

args = parser.parse_args()
# Generate random data for source and target features
source_features = pd.read_csv(args.source_domain_data).iloc[:, 1:]
target_features = pd.read_csv(args.target_domain_data).iloc[:, 1:]

# Assuming source domain label is 1 and target domain label is 0
yd = np.concatenate([np.ones(len(source_features)), np.zeros(len(target_features))])

# Convert to TensorFlow tensors
source_features_tensor = tf.convert_to_tensor(source_features, dtype=tf.float32)
target_features_tensor = tf.convert_to_tensor(target_features, dtype=tf.float32)
yd_tensor = tf.convert_to_tensor(yd, dtype=tf.float32)

# # # Select features based on domain label
# source_features_selected = tf.boolean_mask(source_features_tensor, tf.equal(yd_tensor, 1))
# target_features_selected = tf.boolean_mask(target_features_tensor, tf.equal(yd_tensor, 0))
source_features_selected = source_features_tensor
target_features_selected = target_features_tensor

min_size = min(source_features_selected.shape[0], target_features_selected.shape[0])
source_features_selected = source_features_selected[:min_size]
target_features_selected = target_features_selected[:min_size]
print(min_size)

# Calculate correlation
correlation_matrix = tfp.stats.correlation(source_features_selected, target_features_selected, sample_axis=0, event_axis=-1)
print(correlation_matrix)
correlation = tf.reduce_mean(tf.linalg.diag_part(tf.abs(correlation_matrix)))

print(correlation.numpy())

bins = 60
hist1, _ = np.histogram(source_features, bins=bins, range=(0, 1))
hist2, _ = np.histogram(target_features, bins=bins, range=(0, 1))

hist1 = hist1.astype(np.float32)  # 转换为32位浮点数
hist2 = hist2.astype(np.float32)  # 转换为32位浮点数

hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
print(f'hist_similarity: {hist_similarity}')