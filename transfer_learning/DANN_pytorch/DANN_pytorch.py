'''
python .\DANN_pytorch.py \
    --training_source_domain_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv \
    --training_target_domain_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
    --model_path 220318_231116.pth \
    --work_dir 220318_231116\0.1_10
python .\DANN_pytorch.py \
    --testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                        D:\Experiment\data\220318\GalaxyA51\routes \
                        D:\Experiment\data\231117\GalaxyA51\routes \
    --model_path 220318_231116.pth \
    --work_dir 220318_231116\0.1_10
python ..\..\model_comparison\evaluator.py \
    --model_name DANN_CORR \
    --directory 220318_231116\0.1_10_0.0 \
    --source_domain 220318 \
    --target_domain 231116
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
from itertools import cycle
import math
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('..\\..\\model_comparison')
from walk_definitions import walk_class
from evaluator import Evaluator
sys.path.append('..\\model_comparison')
from drop_out_plot import plot_lines

class IndoorLocalizationDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.loadtxt(file_path, skiprows=1, delimiter=',', dtype='float')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx, 0] - 1
        features = self.data[idx, 1:]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layer1 = nn.Linear(7, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 16)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        return x

class ClassClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ClassClassifier, self).__init__()
        self.layer3 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.layer4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer5 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.layer5(x)
        return x

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(nn.Module):
    def __init__(self, num_classes, epochs, model_save_path='saved_model.pth', loss_weights=None, work_dir=None):
        super(DANN, self).__init__()
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        os.chdir(work_dir)
        self.feature_extractor = FeatureExtractor()
        self.num_classes = num_classes
        self.class_classifier = ClassClassifier(num_classes)
        self.domain_classifier = DomainClassifier()
        self.batch_size = 32
        self.epochs = epochs
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.alpha = 1.0
        self.model_save_path = model_save_path
        self.best_val_total_loss = float('inf')
        self.loss_weights = loss_weights
        self._initialize_metrics()

    def _initialize_metrics(self):
        self.total_losses, self.label_losses, self.domain_losses = [], [], []
        self.source_accuracies, self.target_accuracies, self.total_accuracies = [], [], []
        self.source_domain_accuracies, self.target_domain_accuracies, self.total_domain_accuracies = [], [], []
        self.val_total_losses, self.val_label_losses, self.val_domain_losses = [], [], []
        self.val_source_accuracies, self.val_target_accuracies, self.val_total_accuracies = [], [], []
        self.val_source_domain_accuracies, self.val_target_domain_accuracies, self.val_total_domain_accuracies = [], [], []

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)

        # Domain classification loss
        domain_features = GRL.apply(features, alpha)
        domain_output = self.domain_classifier(domain_features)

        # Class prediction
        class_output = self.class_classifier(features)

        return class_output, domain_output

    def load_train_data(self, source_data_path, target_data_path, drop_out_rate=None):
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

    def train(self, unlabeled=False):
        for epoch in range(self.epochs):
            loss_list, acc_list = self._run_epoch([self.source_train_loader, self.target_train_loader], training=True, unlabeled=unlabeled)

            self.total_losses.append(loss_list[0])
            self.label_losses.append(loss_list[1])
            self.domain_losses.append(loss_list[2])
            self.total_accuracies.append(acc_list[0])
            self.source_accuracies.append(acc_list[1])
            self.target_accuracies.append(acc_list[2])
            self.total_domain_accuracies.append(acc_list[3])
            self.source_domain_accuracies.append(acc_list[4])
            self.target_domain_accuracies.append(acc_list[5])

            # Validation
            with torch.no_grad():
                val_loss_list, val_acc_list = self._run_epoch([self.source_val_loader, self.target_val_loader], training=False, unlabeled=unlabeled)

                self.val_total_losses.append(val_loss_list[0])
                self.val_label_losses.append(val_loss_list[1])
                self.val_domain_losses.append(val_loss_list[2])
                self.val_total_accuracies.append(val_acc_list[0])
                self.val_source_accuracies.append(val_acc_list[1])
                self.val_target_accuracies.append(val_acc_list[2])
                self.val_total_domain_accuracies.append(acc_list[3])
                self.val_source_domain_accuracies.append(acc_list[4])
                self.val_target_domain_accuracies.append(acc_list[5])
                # print(f'Validation Epoch [{epoch+1}/{num_epochs}], Total Loss: {val_total_loss}, Label Loss: {val_label_loss}, Domain Loss: {val_domain_loss}, Source Accuracy: {val_source_accuracy}, Target Accuracy: {val_target_accuracy}')
            
            # print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss}, Label Loss: {label_loss}, Domain Loss: {domain_loss}, Source Accuracy: {source_accuracy}, Target Accuracy: {target_accuracy}')
            print(f'Epoch [{epoch+1}/{self.epochs}], loss: {self.total_losses[-1]:.4f}, label loss: {self.label_losses[-1]:.4f}, domain loss: {self.domain_losses[-1]:.4f}, acc: {self.total_accuracies[-1]:.4f},\nval_loss: {self.val_total_losses[-1]:.4f}, val_label loss: {self.val_label_losses[-1]:.4f}, val_domain loss: {self.val_domain_losses[-1]:.4f}, val_acc: {self.val_total_accuracies[-1]:.4f}')
            
            # Check if the current total loss is the best so far
            if self.val_total_losses[-1] < self.best_val_total_loss:
                # Save the model parameters
                print(f'val_total_loss: {self.val_total_losses[-1]:.4f} < best_val_total_loss: {self.best_val_total_loss:.4f}', end=', ')
                self.save_model()
                self.best_val_total_loss = self.val_total_losses[-1]

            # Update the learning rate scheduler
            # self.scheduler.step()

    def _run_epoch(self, data_loader, training=False, unlabeled=False):
        source_correct_predictions, source_total_samples = 0, 0
        target_correct_predictions, target_total_samples = 0, 0
        source_domain_correct_predictions, source_domain_total_samples = 0, 0
        target_domain_correct_predictions, target_domain_total_samples = 0, 0

        # Create infinite iterators over datasets
        source_iter = cycle(data_loader[0])
        target_iter = cycle(data_loader[1])
        # Calculate num_batches based on the larger dataset
        num_batches = math.ceil(max(len(data_loader[0]), len(data_loader[1])))

        for _ in range(num_batches):
            source_features, source_labels = next(source_iter)
            target_features, target_labels = next(target_iter)
            # Forward pass
            source_labels_pred, source_domain_output = self.forward(source_features, self.alpha)
            target_labels_pred, target_domain_output = self.forward(target_features, self.alpha)

            label_loss_source = nn.CrossEntropyLoss()(source_labels_pred, source_labels)
            label_loss_target = nn.CrossEntropyLoss()(target_labels_pred, target_labels)
            if unlabeled:
                label_loss = label_loss_source
            else:
                label_loss = (label_loss_source + label_loss_target) / 2

            source_domain_loss = nn.CrossEntropyLoss()(source_domain_output, torch.ones(source_domain_output.size(0)).long())
            target_domain_loss = nn.CrossEntropyLoss()(target_domain_output, torch.zeros(target_domain_output.size(0)).long())
            domain_loss = source_domain_loss + target_domain_loss

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

            _, source_domain_preds = torch.max(source_domain_output, 1)
            source_domain_correct_predictions += (source_domain_preds == 1).sum().item()  # Assuming 1 for source domain
            source_domain_total_samples += source_domain_output.size(0)
            source_domain_accuracy = source_domain_correct_predictions / source_domain_total_samples

            _, target_domain_preds = torch.max(target_domain_output, 1)
            target_domain_correct_predictions += (target_domain_preds == 0).sum().item()  # Assuming 0 for target domain
            target_domain_total_samples += target_domain_output.size(0)
            target_domain_accuracy = target_domain_correct_predictions / target_domain_total_samples

            loss_list = [total_loss.item(), label_loss.item(), domain_loss]
            acc_list = [
                (source_accuracy + target_accuracy) / 2,
                source_accuracy,
                target_accuracy,
                (source_domain_accuracy + target_domain_accuracy) / 2,
                source_domain_accuracy,
                target_domain_accuracy
            ]
        return loss_list, acc_list

    def save_model(self):
        torch.save(self.state_dict(), self.model_save_path)
        print(f"Model parameters saved to {self.model_save_path}")

    def plot_training_results(self):
        
        label_losses_values = [loss for loss in self.label_losses]
        val_label_losses_values = [loss for loss in self.val_label_losses]
        domain_losses_values = [loss.detach() for loss in self.domain_losses]
        val_domain_losses_values = [loss.detach() for loss in self.val_domain_losses]

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
        plt.plot(epochs_list, domain_losses_values, label='Domain Loss', color='blue')
        plt.plot(epochs_list, val_domain_losses_values, label='Val Domain Loss', color='darkorange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Domain Discriminator Training Loss')

        # Remove empty subplot (Bottom Right)
        plt.subplot(2, 2, 4)
        plt.plot(epochs_list, self.total_domain_accuracies, label='Accuracy', color='blue')
        plt.plot(epochs_list, self.val_total_domain_accuracies, label='Val Accuracy', color='darkorange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training Accuracy')

        # Add a title for the entire figure
        plt.suptitle('Training Curve')

        plt.tight_layout()  # Adjust layout for better spacing
        plt.savefig('loss_and_accuracy.png')

    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
        else:
            print(f"Error: Model file not found at {model_path}")

    def generate_predictions(self, model_path):
        self.load_model(model_path)
        prediction_results = {
            'label': [],
            'pred': []
        }
        # 進行預測
        with torch.no_grad():
            for test_batch, true_label_batch in self.test_loader:
                labels_pred, domain_output = self.forward(test_batch)
                _, preds = torch.max(labels_pred, 1)
                predicted_labels = preds + 1  # 加 1 是为了将索引转换为 1 到 41 的标签
                label = true_label_batch + 1
                # 將預測結果保存到 prediction_results 中
                prediction_results['label'].extend(label.tolist())
                prediction_results['pred'].extend(predicted_labels.tolist())
        return pd.DataFrame(prediction_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
    parser.add_argument('--testing_data_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--model_path', type=str, default='my_model.pth', help='path of .pth file of model')
    parser.add_argument('--work_dir', type=str, default='DANN_CORR', help='create new directory to save result')
    args = parser.parse_args()

    num_classes = 41
    epochs = 500
    loss_weights = [1, 1]
    unlabeled = False
    
    domain1_result = []
    domain2_result = []
    domain3_result = []

    data_drop_out_list = np.arange(0.0, 0.05, 0.1)
    
    for data_drop_out in data_drop_out_list:
        # 創建 DANNModel    
        dann_model = DANN(num_classes, model_save_path=args.model_path, loss_weights=loss_weights, epochs=epochs, work_dir=f'{args.work_dir}_{data_drop_out:.1f}')
        summary(dann_model, (7,))
        # 讀取資料
        if args.training_source_domain_data and args.training_target_domain_data:
            # 訓練模型
            dann_model.load_train_data(args.training_source_domain_data, args.training_target_domain_data, data_drop_out)
            dann_model.train(unlabeled=unlabeled)
            dann_model.plot_training_results()
        elif args.testing_data_list:
            testing_data_path_list = args.testing_data_list
            for testing_data_path in testing_data_path_list:
                for walk_str, walk_list in walk_class:
                    prediction_results = pd.DataFrame()
                    for walk in walk_list:
                        # 加載數據
                        dann_model.load_test_data(f"{testing_data_path}\\{walk}.csv")
                        results = dann_model.generate_predictions(args.model_path)
                        prediction_results = pd.concat([prediction_results, results], ignore_index=True)
                    split_path = testing_data_path.split('\\')
                    predictions_dir = f'predictions/{split_path[3]}'
                    os.makedirs(predictions_dir, exist_ok=True)
                    prediction_results.to_csv(os.path.join(predictions_dir, f'{walk_str}_predictions.csv'), index=False)
            predicion_data_path_list = os.listdir('predictions/')
            evaluator = Evaluator()
            mde_list = evaluator.test(predicion_data_path_list, f'{args.work_dir}_{data_drop_out}')
            domain1_result.append(mde_list[0][1])
            domain2_result.append(mde_list[1][1])
            domain3_result.append(mde_list[2][1])
        else:
            print('Please specify --training_source_domain_data/--training_target_domain_data or --testing_data_list option.')

        os.chdir('..\\..')

    if args.testing_data_list:
        plot_lines(data_drop_out_list, domain2_result, domain_name='231116', output_path=args.work_dir, title='Source_domain_to_Target_domain')
    