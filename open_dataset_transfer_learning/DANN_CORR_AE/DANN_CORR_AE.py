'''
python DANN_CORR_AE.py --training_source_domain_data D:\Experiment\data\\UM_DSI_DB_v1.0.0_lite\data\tony_data\2019-06-11\wireless_training.csv ^
                       --training_target_domain_data D:\Experiment\data\\UM_DSI_DB_v1.0.0_lite\data\tony_data\2020-02-19\wireless_training.csv ^
                       --work_dir 190611_200219\0.1_2_2
python .\DANN_CORR_AE.py \
    --testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                        D:\Experiment\data\220318\GalaxyA51\routes \
                        D:\Experiment\data\231117\GalaxyA51\routes \
    --model_path 220318_231116.pth \
    --work_dir 220318_231116\0.1_2_2
'''
import torch.nn.functional as F
import torch
import torch.nn as nn
from itertools import cycle
import math
import cv2
import sys
sys.path.append('..\\DANN_CORR')
from DANN_CORR import HistCorrDANNModel, LabelPredictor, DomainAdaptationModel
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class AutoencoderFeatureExtractor(nn.Module):
    def __init__(self):
        super(AutoencoderFeatureExtractor, self).__init__()
        self.encoder_fc1 = nn.Linear(168, 128)
        self.encoder_fc2 = nn.Linear(128, 64)
        self.decoder_fc1 = nn.Linear(64, 128)
        self.decoder_fc2 = nn.Linear(128, 168)

    def forward(self, x, use_decoder=False):
        # Encoder
        x = torch.relu(self.encoder_fc1(x))
        encoded_features = torch.relu(self.encoder_fc2(x))

        if use_decoder:
            # Decoder
            x = torch.relu(self.decoder_fc1(encoded_features))
            x = torch.sigmoid(self.decoder_fc2(x))  # Using sigmoid for reconstruction in [0, 1]
            return x
        else:
            return encoded_features

class HistCorrAutoencoderDANNModel(HistCorrDANNModel):
    def _initialize_model(self):
        self.feature_extractor = AutoencoderFeatureExtractor()
        self.label_predictor = LabelPredictor(64, num_classes=49)
        self.domain_adaptation_model = DomainAdaptationModel(self.feature_extractor, self.label_predictor)

    def _initialize_metrics(self):
        self.total_losses, self.label_losses, self.domain_losses, self.reconstruction_losses = [], [], [], []
        self.source_accuracies, self.target_accuracies, self.total_accuracies = [], [], []
        self.val_total_losses, self.val_label_losses, self.val_domain_losses, self.val_reconstruction_losses = [], [], [], []
        self.val_source_accuracies, self.val_target_accuracies, self.val_total_accuracies = [], [], []

    def train(self, num_epochs=10, unlabeled=False):
        unlabeled = unlabeled
        for epoch in range(num_epochs):
            loss_list, acc_list = self._run_epoch([self.source_train_loader, self.target_train_loader], training=True, unlabeled=unlabeled)

            self.total_losses.append(loss_list[0])
            self.label_losses.append(loss_list[1])
            self.domain_losses.append(loss_list[2])
            self.reconstruction_losses.append(loss_list[3])
            self.total_accuracies.append(acc_list[0])
            self.source_accuracies.append(acc_list[1])
            self.target_accuracies.append(acc_list[2])

            # Validation
            with torch.no_grad():
                val_loss_list, val_acc_list = self._run_epoch([self.source_val_loader, self.target_val_loader], training=False, unlabeled=unlabeled)

                self.val_total_losses.append(val_loss_list[0])
                self.val_label_losses.append(val_loss_list[1])
                self.val_domain_losses.append(val_loss_list[2])
                self.val_reconstruction_losses.append(val_loss_list[3])
                self.val_total_accuracies.append(val_acc_list[0])
                self.val_source_accuracies.append(val_acc_list[1])
                self.val_target_accuracies.append(val_acc_list[2])
                
            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {self.total_losses[-1]:.4f}, label loss: {self.label_losses[-1]:.4f}, domain loss: {self.domain_losses[-1]:.4f}, reconstruction loss: {self.reconstruction_losses[-1]:.4f}, acc: {self.total_accuracies[-1]:.4f},\nval_loss: {self.val_total_losses[-1]:.4f}, val_label loss: {self.val_label_losses[-1]:.4f}, val_domain loss: {self.val_domain_losses[-1]:.4f}, val_reconstruction loss: {self.val_reconstruction_losses[-1]:.4f}, val_acc: {self.val_total_accuracies[-1]:.4f}')
            
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
            # Autoencoder forward pass
            source_encoded_features = self.feature_extractor(source_features, use_decoder=False)
            target_encoded_features = self.feature_extractor(target_features, use_decoder=False)

            # Classifier forward pass
            source_labels_pred = self.label_predictor(source_encoded_features)
            target_labels_pred = self.label_predictor(target_encoded_features)

            # Reconstruction loss
            reconstruction_loss_source = F.mse_loss(self.feature_extractor(source_features, use_decoder=True), source_features)
            reconstruction_loss_target = F.mse_loss(self.feature_extractor(target_features, use_decoder=True), target_features)
            reconstruction_loss = (reconstruction_loss_source + reconstruction_loss_target) / 2

            # Classifier loss
            label_loss_source = self.domain_criterion(source_labels_pred, source_labels)
            label_loss_target = self.domain_criterion(target_labels_pred, target_labels)
            if unlabeled:
                label_loss = label_loss_source
            else:
                label_loss = (label_loss_source + label_loss_target) / 2
            source_hist = cv2.calcHist([source_encoded_features.detach().numpy().flatten()], [0], None, [100], [0, 1])
            target_hist = cv2.calcHist([target_encoded_features.detach().numpy().flatten()], [0], None, [100], [0, 1])
            domain_loss = self.domain_invariance_loss(source_hist, target_hist)

            total_loss = self.loss_weights[0] * label_loss + self.loss_weights[1] * domain_loss + self.loss_weights[2] * reconstruction_loss

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

            loss_list = [total_loss.item(), label_loss.item(), domain_loss, reconstruction_loss.item()]
            acc_list = [(source_accuracy + target_accuracy) / 2, source_accuracy, target_accuracy]

        return loss_list, acc_list
    
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
        plt.plot(epochs_list, self.reconstruction_losses, label='Reconstruction Loss', color='blue')
        plt.plot(epochs_list, self.val_reconstruction_losses, label='Val Reconstruction Loss', color='darkorange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Domain Discriminator Training Loss')

        # Add a title for the entire figure
        plt.suptitle('Training Curve')

        plt.tight_layout()  # Adjust layout for better spacing
        plt.savefig('loss_and_accuracy.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
    parser.add_argument('--test', action='store_true' , help='for test')
    parser.add_argument('--model_path', type=str, default='my_model.pth', help='path of .pth file of model')
    parser.add_argument('--work_dir', type=str, default='DANN_CORR', help='create new directory to save result')
    args = parser.parse_args()
    loss_weights = [0.1, 2, 2]
    epoch = 100
    unlabeled = False

    domain1_result = []
    domain2_result = []
    domain3_result = []

    data_drop_out_list = np.arange(0.9, 0.95, 0.1)
    
    for data_drop_out in data_drop_out_list:
        # 創建 DANNModel    
        dann_model = HistCorrAutoencoderDANNModel(model_save_path=args.model_path, loss_weights=loss_weights, lr=0.0005, work_dir=f'{args.work_dir}_{data_drop_out:.1f}')
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
                        r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data\2019-06-11\wireless_testing.csv',
                        r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data\2019-10-09\wireless_testing.csv',
                        r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\tony_data\2020-02-19\wireless_testing.csv'
                    ]
            output_paths = ['predictions/190611_results.csv', 'predictions/191009_results.csv', 'predictions/200219_results.csv']
            if not os.path.exists('predictions'):
                os.makedirs('predictions')
            for testing_file_path, output_path in zip(testing_file_paths, output_paths):
                dann_model.generate_predictions(testing_file_path, output_path)
        else:
            print('Please specify --training_source_domain_data/--training_target_domain_data or --testing_data_list option.')

        os.chdir('..\\..')
