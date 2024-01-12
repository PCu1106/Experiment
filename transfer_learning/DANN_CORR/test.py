import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import numpy as np
import cv2
import argparse

# 定义域不变性损失函数
def domain_invariance_loss(source_hist, target_hist):
    # 计算直方图相关性，越接近1越好
    correlation = cv2.compareHist(source_hist, target_hist, cv2.HISTCMP_CORREL)
    return 1 - correlation

# 定义特征提取器（ResNet18作为例子）
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 在这里定义你自己的特征提取器
        # 例如，可以使用全连接层或卷积层
        self.fc1 = nn.Linear(7, 8)
        self.fc2 = nn.Linear(8, 16)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return x

# 定义标签预测器
class LabelPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# 定义整体模型
class DomainAdaptationModel(nn.Module):
    def __init__(self, feature_extractor, label_predictor):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor

    def forward(self, x):
        features = self.feature_extractor(x)
        labels = self.label_predictor(features)
        return features, labels

# 定义自定义的Dataset类
class IndoorLocalizationDataset(Dataset):
    def __init__(self, file_path):
        # 从文件加载数据
        self.data = np.loadtxt(file_path, skiprows=1, delimiter=',', dtype='float')  # 指定逗号分隔符和数据类型为float

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx, 0] - 1  # 将标签减去1
        features = self.data[idx, 1:]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

parser = argparse.ArgumentParser(description='Train DANN Model')
parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
args = parser.parse_args()

# 加载数据
source_dataset = IndoorLocalizationDataset(args.training_source_domain_data)
target_dataset = IndoorLocalizationDataset(args.training_target_domain_data)

batch_size = 32
source_data_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
target_data_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
feature_extractor = FeatureExtractor()
label_predictor = LabelPredictor(16, num_classes=41)  # 41个Reference points
domain_adaptation_model = DomainAdaptationModel(feature_extractor, label_predictor)
optimizer = optim.Adam(domain_adaptation_model.parameters(), lr=0.001)

# 定义损失函数
domain_criterion = nn.CrossEntropyLoss()  # 交叉熵损失用于标签预测

# Lists to store training statistics
train_losses = []
label_losses = []
domain_losses = []
source_accuracies = []
target_accuracies = []

# 训练模型
num_epochs = 10
epochs_list = np.arange(0, 10, 1, dtype=int)
for epoch in range(num_epochs):
    source_correct_predictions = 0
    source_total_samples = 0
    target_correct_predictions = 0
    target_total_samples = 0
    for source_batch, target_batch in zip(source_data_loader, target_data_loader):
        source_features, source_labels = source_batch
        target_features, target_labels = target_batch

        # Ensure both source and target batches have the same size
        min_batch_size = min(source_labels.size(0), target_labels.size(0))
        source_features, source_labels = source_features[:min_batch_size], source_labels[:min_batch_size]
        target_features, target_labels = target_features[:min_batch_size], target_labels[:min_batch_size]

        # 提取特征
        source_features, source_labels_pred = domain_adaptation_model(source_features) # dim = 41 (not softmax)
        target_features, target_labels_pred = domain_adaptation_model(target_features) # dim = 41 (not softmax)

        # 计算标签预测损失
        label_loss_source = domain_criterion(source_labels_pred, source_labels)
        label_loss_target = domain_criterion(target_labels_pred, target_labels)
        label_loss = (label_loss_source + label_loss_target) / 2

        # 计算直方图并计算域不变性损失
        source_hist = cv2.calcHist([source_features.detach().numpy().flatten()], [0], None, [100], [0, 1])
        target_hist = cv2.calcHist([target_features.detach().numpy().flatten()], [0], None, [100], [0, 1])
        domain_loss = domain_invariance_loss(source_hist, target_hist)

        # 计算总损失
        loss_weight = [0.5, 0.5]
        total_loss = loss_weight[0] * label_loss + loss_weight[1] * domain_loss

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Compute accuracy for source domain
        _, source_preds = torch.max(source_labels_pred, 1) # argmax
        source_correct_predictions += (source_preds == source_labels).sum().item() # [0, 40] and [0, 40]
        source_total_samples += source_labels.size(0)
        source_accuracy = source_correct_predictions / source_total_samples

        # Compute accuracy for target domain
        _, target_preds = torch.max(target_labels_pred, 1) # argmax
        target_correct_predictions += (target_preds == target_labels).sum().item()
        target_total_samples += target_labels.size(0)
        target_accuracy = target_correct_predictions / target_total_samples

    # Append the losses and accuracies
    train_losses.append(total_loss)
    label_losses.append(label_loss)
    domain_losses.append(domain_loss)
    source_accuracies.append(source_accuracy)
    target_accuracies.append(target_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss}, Label Loss: {label_loss}, Domain Loss: {domain_loss}, Source Accuracy: {source_accuracy}, Target Accuracy: {target_accuracy}')

# 训练完成后，可以使用feature_extractor提取特征，并计算特征之间的域不变性

# Extracting numeric values from tensors for plotting
train_losses_values = [loss.item() for loss in train_losses]
label_losses_values = [loss.item() for loss in label_losses]

# Plotting training loss
plt.figure(figsize=(10, 5))
plt.plot(epochs_list, train_losses_values, label='Total Loss', color='blue')
plt.plot(epochs_list, label_losses_values, label='Label Loss', color='green')
plt.plot(epochs_list, domain_losses, label='Domain Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')
plt.show()

# Plotting training accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs_list, source_accuracies, label='Source Accuracy', color='blue')
plt.plot(epochs_list, target_accuracies, label='Target Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy')
plt.show()