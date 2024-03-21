'''
python .\DANN_CORR_reverse.py --training_source_domain_data D:\Experiment\data\\UM_DSI_DB_v1.0.0_lite\data\tony_data_reverse\2020-02-19\wireless_training.csv ^
                      --training_target_domain_data D:\Experiment\data\\UM_DSI_DB_v1.0.0_lite\data\tony_data_reverse\2019-06-11\wireless_training.csv ^
                      --work_dir 200219_190611\\unlabeled\0.1_10
python .\DANN_CORR_reverse.py --test --work_dir 200219_190611\\unlabeled\0.1_10
python ..\..\model_comparison\evaluator.py \
    --model_name DANN_CORR \
    --directory 220318_231116\0.1_10_0.0 \
    --source_domain 220318 \
    --target_domain 231116
'''
import torch
import torch.nn as nn
from itertools import cycle
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchviz import make_dot
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2
import argparse
import os
import sys

class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return x

class LabelPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        return x

class DomainAdaptationModel(nn.Module):
    def __init__(self, feature_extractor, label_predictor):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor

    def forward(self, x):
        features = self.feature_extractor(x)
        labels = self.label_predictor(features)
        return features, labels

class IndoorLocalizationDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.loadtxt(file_path, skiprows=1, delimiter=',', dtype='float')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx, 0] - 1
        features = self.data[idx, 1:]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class HistCorrDANNModel:
    def __init__(self, model_save_path='saved_model.pth', loss_weights=None, lr=0.001, work_dir=None):
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        os.chdir(work_dir)
        self.batch_size = 32
        self.loss_weights = loss_weights
        self.lr = lr
        self.input_size = 191
        self.feature_extractor_neurons = [128, 64]

        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_metrics()

        self.model_save_path = model_save_path
        self.best_val_total_loss = float('inf')  # Initialize with positive infinity

    def load_train_data(self, source_data_path, target_data_path, drop_out_rate=0.0):
        self.source_dataset = IndoorLocalizationDataset(source_data_path)
        self.target_dataset = IndoorLocalizationDataset(target_data_path)

        # Drop samples from the target domain based on drop_out_rate
        if drop_out_rate > 0:
            num_samples_to_drop = int(len(self.target_dataset) * drop_out_rate)
            if drop_out_rate >= 1.0:
                num_samples_to_drop = num_samples_to_drop - 2
            drop_indices = np.random.choice(len(self.target_dataset), num_samples_to_drop, replace=False)
            self.target_dataset.data = np.delete(self.target_dataset.data, drop_indices, axis=0)

        print(f"Total Source Dataset Size: {len(self.source_dataset)}")
        print(f"Total Target Dataset Size: {len(self.target_dataset)}")

        # Split source data into training and validation sets
        source_train, source_val = train_test_split(self.source_dataset, test_size=0.2, random_state=42)
        self.source_train_loader = DataLoader(source_train, batch_size=self.batch_size, shuffle=True)
        self.source_val_loader = DataLoader(source_val, batch_size=self.batch_size, shuffle=False)

        # Split target data into training and validation sets
        target_train, target_val = train_test_split(self.target_dataset, test_size=0.2, random_state=42)
        self.target_train_loader = DataLoader(target_train, batch_size=self.batch_size, shuffle=True)
        self.target_val_loader = DataLoader(target_val, batch_size=self.batch_size, shuffle=False)

    def load_test_data(self, test_data_path):
        self.test_dataset = IndoorLocalizationDataset(test_data_path)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False)

    def _initialize_model(self):
        self.feature_extractor = FeatureExtractor(self.input_size, self.feature_extractor_neurons[0], self.feature_extractor_neurons[1])
        self.label_predictor = LabelPredictor(self.feature_extractor_neurons[1], num_classes=49)
        self.domain_adaptation_model = DomainAdaptationModel(self.feature_extractor, self.label_predictor)

    def _initialize_optimizer(self):
        self.optimizer = optim.Adam(self.domain_adaptation_model.parameters(), lr=self.lr)
        self.domain_criterion = nn.CrossEntropyLoss()

    def _initialize_metrics(self):
        self.total_losses, self.label_losses, self.domain_losses = [], [], []
        self.source_accuracies, self.target_accuracies, self.total_accuracies = [], [], []
        self.val_total_losses, self.val_label_losses, self.val_domain_losses = [], [], []
        self.val_source_accuracies, self.val_target_accuracies, self.val_total_accuracies = [], [], []

    def domain_invariance_loss(self, source_hist, target_hist):
        correlation = cv2.compareHist(source_hist, target_hist, cv2.HISTCMP_CORREL)
        return 1 - correlation

    def train(self, num_epochs=10, unlabeled=False):
        unlabeled = unlabeled
        for epoch in range(num_epochs):
            loss_list, acc_list = self._run_epoch([self.source_train_loader, self.target_train_loader], training=True, unlabeled=unlabeled)

            self.total_losses.append(loss_list[0])
            self.label_losses.append(loss_list[1])
            self.domain_losses.append(loss_list[2])
            self.total_accuracies.append(acc_list[0])
            self.source_accuracies.append(acc_list[1])
            self.target_accuracies.append(acc_list[2])

            # Validation
            with torch.no_grad():
                val_loss_list, val_acc_list = self._run_epoch([self.source_val_loader, self.target_val_loader], training=False, unlabeled=unlabeled)

                self.val_total_losses.append(val_loss_list[0])
                self.val_label_losses.append(val_loss_list[1])
                self.val_domain_losses.append(val_loss_list[2])
                self.val_total_accuracies.append(val_acc_list[0])
                self.val_source_accuracies.append(val_acc_list[1])
                self.val_target_accuracies.append(val_acc_list[2])
                # print(f'Validation Epoch [{epoch+1}/{num_epochs}], Total Loss: {val_total_loss}, Label Loss: {val_label_loss}, Domain Loss: {val_domain_loss}, Source Accuracy: {val_source_accuracy}, Target Accuracy: {val_target_accuracy}')
            
            # print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss}, Label Loss: {label_loss}, Domain Loss: {domain_loss}, Source Accuracy: {source_accuracy}, Target Accuracy: {target_accuracy}')
            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {self.total_losses[-1]:.4f}, label loss: {self.label_losses[-1]:.4f}, domain loss: {self.domain_losses[-1]:.4f}, acc: {self.total_accuracies[-1]:.4f},\nval_loss: {self.val_total_losses[-1]:.4f}, val_label loss: {self.val_label_losses[-1]:.4f}, val_domain loss: {self.val_domain_losses[-1]:.4f}, val_acc: {self.val_total_accuracies[-1]:.4f}')
            
            # Check if the current total loss is the best so far
            if self.val_total_losses[-1] < self.best_val_total_loss:
                # Save the model parameters
                print(f'val_total_loss: {self.val_total_losses[-1]:.4f} < best_val_total_loss: {self.best_val_total_loss:.4f}', end=', ')
                self.save_model()
                self.best_val_total_loss = self.val_total_losses[-1]

    def _run_epoch(self, data_loader, training=False, unlabeled=False):
        source_correct_predictions, source_total_samples = 0, 0
        target_correct_predictions, target_total_samples = 0, 0
        # Create infinite iterators over datasets
        source_iter = cycle(data_loader[0])
        target_iter = cycle(data_loader[1])
        # Calculate num_batches based on the larger dataset
        num_batches = math.ceil(max(len(data_loader[0]), len(data_loader[1])))

        for _ in range(num_batches):
            source_features, source_labels = next(source_iter)
            target_features, target_labels = next(target_iter)
            source_features, source_labels_pred = self.domain_adaptation_model(source_features)
            target_features, target_labels_pred = self.domain_adaptation_model(target_features)

            label_loss_source = self.domain_criterion(source_labels_pred, source_labels)
            label_loss_target = self.domain_criterion(target_labels_pred, target_labels)
            if unlabeled:
                label_loss = label_loss_source
            else:
                label_loss = (label_loss_source + label_loss_target) / 2

            source_hist = cv2.calcHist([source_features.detach().numpy().flatten()], [0], None, [100], [0, 1])
            target_hist = cv2.calcHist([target_features.detach().numpy().flatten()], [0], None, [100], [0, 1])
            domain_loss = self.domain_invariance_loss(source_hist, target_hist)

            total_loss = self.loss_weights[0] * label_loss + self.loss_weights[1] * domain_loss

            if training:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            _, source_preds = torch.max(source_labels_pred, 1)
            source_correct_predictions += (source_preds == source_labels).sum().item()
            source_total_samples += source_labels.size(0)
            source_accuracy = source_correct_predictions / source_total_samples

            _, target_preds = torch.max(target_labels_pred, 1)
            target_correct_predictions += (target_preds == target_labels).sum().item()
            target_total_samples += target_labels.size(0)
            target_accuracy = target_correct_predictions / target_total_samples
            loss_list = [total_loss.item(), label_loss.item(), domain_loss]
            acc_list = [(source_accuracy + target_accuracy) / 2, source_accuracy, target_accuracy]
        return loss_list, acc_list

    def save_model(self):
        torch.save(self.domain_adaptation_model.state_dict(), self.model_save_path)
        print(f"Model parameters saved to {self.model_save_path}")

    def plot_training_results(self):
        epochs_list = np.arange(0, len(self.total_losses), 1)
        label_losses_values = [loss for loss in self.label_losses]
        val_label_losses_values = [loss for loss in self.val_label_losses]

        plt.figure(figsize=(12, 8))
        
        # Subplot for Label Predictor Training Loss (Top Left)
        plt.subplot(2, 2, 1)
        plt.plot(epochs_list, label_losses_values, label='Label Loss', color='blue')
        plt.plot(epochs_list, val_label_losses_values, label='Val Label Loss', color='darkorange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Label Predictor Training Loss')

        # Subplot for Training Accuracy (Top Right)
        plt.subplot(2, 2, 2)
        plt.plot(epochs_list, self.total_accuracies, label='Accuracy', color='blue')
        plt.plot(epochs_list, self.val_total_accuracies, label='Val Accuracy', color='darkorange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training Accuracy')

        # Subplot for Domain Discriminator Training Loss (Bottom Left)
        plt.subplot(2, 2, 3)
        plt.plot(epochs_list, self.domain_losses, label='Domain Loss', color='blue')
        plt.plot(epochs_list, self.val_domain_losses, label='Val Domain Loss', color='darkorange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Domain Discriminator Training Loss')

        # Remove empty subplot (Bottom Right)
        plt.subplot(2, 2, 4)
        plt.axis('off')

        # Add a title for the entire figure
        plt.suptitle('Training Curve')

        plt.tight_layout()  # Adjust layout for better spacing
        plt.savefig('loss_and_accuracy.png')

    def save_model_architecture(self, file_path='model_architecture'):
        # Create a dummy input for visualization
        dummy_input = torch.randn(1, 191)  # Assuming input size is (batch_size, 7)

        # Generate a graph of the model architecture
        graph = make_dot(self.domain_adaptation_model(dummy_input), params=dict(self.domain_adaptation_model.named_parameters()))

        # Save the graph as an image file
        graph.render(file_path, format='png')
        print(f"Model architecture saved as {file_path}")

    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.domain_adaptation_model.load_state_dict(torch.load(model_path))
        else:
            print(f"Error: Model file not found at {model_path}")

    def predict(self, features):
        self.domain_adaptation_model.eval()
        with torch.no_grad():
            features, labels_pred = self.domain_adaptation_model(features)
        return labels_pred

    def generate_predictions(self, file_path, output_path):
        predictions = {'label': [], 'pred': []}
        self.load_test_data(file_path)
        with torch.no_grad():
            for test_batch, true_label_batch in self.test_loader:
                labels_pred = self.predict(test_batch)
                _, preds = torch.max(labels_pred, 1)
                predicted_labels = preds + 1  # 加 1 是为了将索引转换为 1 到 49 的标签
                label = true_label_batch + 1
                # 將預測結果保存到 predictions 中
                predictions['label'].extend(label.tolist())
                predictions['pred'].extend(predicted_labels.tolist())
        # 将预测结果保存为 CSV 文件
        results = pd.DataFrame({'label': predictions['label'], 'pred': predictions['pred']})
        results.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
    parser.add_argument('--test', action='store_true' , help='for test')
    parser.add_argument('--model_path', type=str, default='my_model.pth', help='path of .pth file of model')
    parser.add_argument('--work_dir', type=str, default='DANN_CORR', help='create new directory to save result')
    args = parser.parse_args()
    loss_weights = [0.1, 10]
    epoch = 100
    unlabeled = True
    
    domain1_result = []
    domain2_result = []
    domain3_result = []

    data_drop_out_list = np.arange(0.0, 0.05, 0.1)
    
    for data_drop_out in data_drop_out_list:
        # 創建 DANNModel    
        dann_model = HistCorrDANNModel(model_save_path=args.model_path, loss_weights=loss_weights, work_dir=f'{args.work_dir}_{data_drop_out:.1f}')
        dann_model.save_model_architecture()
        # 讀取資料
        if args.training_source_domain_data and args.training_target_domain_data:
            # 訓練模型
            dann_model.load_train_data(args.training_source_domain_data, args.training_target_domain_data, data_drop_out)
            dann_model.train(num_epochs=epoch, unlabeled=unlabeled)
            dann_model.plot_training_results()
        elif args.test:
            dann_model.load_model(args.model_path)
            testing_file_paths = [
                        r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data_reverse\2019-06-11\wireless_testing.csv',
                        r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data_reverse\2019-10-09\wireless_testing.csv',
                        r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data_reverse\2020-02-19\wireless_testing.csv'
                    ]
            output_paths = ['predictions/190611_results.csv', 'predictions/191009_results.csv', 'predictions/200219_results.csv']
            if not os.path.exists('predictions'):
                os.makedirs('predictions')
            for testing_file_path, output_path in zip(testing_file_paths, output_paths):
                dann_model.generate_predictions(testing_file_path, output_path)
        else:
            print('Please specify --training_source_domain_data/--training_target_domain_data or --testing_data_list option.')

        os.chdir('..\\..')
