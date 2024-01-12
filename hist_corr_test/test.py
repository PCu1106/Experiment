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

bins = 100
hist1, _ = np.histogram(source_features, bins=bins, range=(0, 1))
hist2, _ = np.histogram(target_features, bins=bins, range=(0, 1))

hist1 = hist1.astype(np.float32)  # 转换为32位浮点数
hist2 = hist2.astype(np.float32)  # 转换为32位浮点数

hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
print(f'hist_similarity: {hist_similarity}')

test_feature = source_features[:2]
print(test_feature)
hist3, _ = np.histogram(test_feature, bins=bins, range=(0, 1))
print(hist3)

import tensorflow as tf

# 自定义损失函数 - HISTCMP_CORREL
def histcmp_correl_loss(hist1, hist2):
    correl = tf.image.ssim(hist1, hist2, max_val=1.0)
    return 1.0 - correl

# 自定义损失函数 - Domain loss
def domain_loss(y_true, y_pred):
    # 提取每个domain的样本
    target_samples = tf.boolean_mask(y_pred, y_true == 0)
    source_samples = tf.boolean_mask(y_pred, y_true == 1)
    
    # 计算直方图
    hist_target = tf.histogram_fixed_width(target_samples, [0.0, 1.0], nbins=100)
    hist_source = tf.histogram_fixed_width(source_samples, [0.0, 1.0], nbins=100)
    
    # 计算HISTCMP_CORREL损失
    loss = histcmp_correl_loss(hist_target, hist_source)
    return loss

# 构建模型
input_dim = 7
output_dim = 41

model = tf.keras.models.Sequential([
    # 特征提取器
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(128, activation='relu'),
    
    # 标签预测器
    tf.keras.layers.Dense(output_dim, activation='softmax', name='class_label_output'),
    
    # Domain预测器
    tf.keras.layers.Dense(1, activation='sigmoid', name='domain_label_output')
])

# 编译模型并定义损失函数
model.compile(optimizer='adam', 
              loss={'class_label_output': 'categorical_crossentropy', 'domain_label_output': domain_loss},
              metrics={'class_label_output': 'accuracy', 'domain_label_output': 'accuracy'})
