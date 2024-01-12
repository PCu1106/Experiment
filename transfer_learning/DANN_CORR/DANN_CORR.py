import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchviz import make_dot
from sklearn.model_selection import train_test_split

import numpy as np
import cv2
import argparse

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(7, 8)
        self.fc2 = nn.Linear(8, 16)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return x

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
    def __init__(self, source_data_path, target_data_path, model_save_path='saved_model.pth'):
        self.source_dataset = IndoorLocalizationDataset(source_data_path)
        self.target_dataset = IndoorLocalizationDataset(target_data_path)

        self.batch_size = 32
        self.source_data_loader = DataLoader(self.source_dataset, batch_size=self.batch_size, shuffle=True)
        self.target_data_loader = DataLoader(self.target_dataset, batch_size=self.batch_size, shuffle=True)

        self.feature_extractor = FeatureExtractor()
        self.label_predictor = LabelPredictor(16, num_classes=41)
        self.domain_adaptation_model = DomainAdaptationModel(self.feature_extractor, self.label_predictor)

        self.optimizer = optim.Adam(self.domain_adaptation_model.parameters(), lr=0.001)
        self.domain_criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.label_losses = []
        self.domain_losses = []
        self.source_accuracies = []
        self.target_accuracies = []

        self.model_save_path = model_save_path
        self.best_total_loss = float('inf')  # Initialize with positive infinity

    def domain_invariance_loss(self, source_hist, target_hist):
        correlation = cv2.compareHist(source_hist, target_hist, cv2.HISTCMP_CORREL)
        return 1 - correlation

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            source_correct_predictions = 0
            source_total_samples = 0
            target_correct_predictions = 0
            target_total_samples = 0

            for source_batch, target_batch in zip(self.source_data_loader, self.target_data_loader):
                source_features, source_labels = source_batch
                target_features, target_labels = target_batch

                min_batch_size = min(source_labels.size(0), target_labels.size(0))
                source_features, source_labels = source_features[:min_batch_size], source_labels[:min_batch_size]
                target_features, target_labels = target_features[:min_batch_size], target_labels[:min_batch_size]

                source_features, source_labels_pred = self.domain_adaptation_model(source_features)
                target_features, target_labels_pred = self.domain_adaptation_model(target_features)

                label_loss_source = self.domain_criterion(source_labels_pred, source_labels)
                label_loss_target = self.domain_criterion(target_labels_pred, target_labels)
                label_loss = (label_loss_source + label_loss_target) / 2

                source_hist = cv2.calcHist([source_features.detach().numpy().flatten()], [0], None, [100], [0, 1])
                target_hist = cv2.calcHist([target_features.detach().numpy().flatten()], [0], None, [100], [0, 1])
                domain_loss = self.domain_invariance_loss(source_hist, target_hist)

                loss_weight = [0.5, 0.5]
                total_loss = loss_weight[0] * label_loss + loss_weight[1] * domain_loss

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

                self.train_losses.append(total_loss.item())
                self.label_losses.append(label_loss.item())
                self.domain_losses.append(domain_loss)
                self.source_accuracies.append(source_accuracy)
                self.target_accuracies.append(target_accuracy)

            print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss}, Label Loss: {label_loss}, Domain Loss: {domain_loss}, Source Accuracy: {source_accuracy}, Target Accuracy: {target_accuracy}')
            # Check if the current total loss is the best so far
            if total_loss < self.best_total_loss:
                self.best_total_loss = total_loss
                # Save the model parameters
                print(f'total_loss: {total_loss} < best_total_loss: {self.best_total_loss}')
                self.save_model()

    def save_model(self):
        torch.save(self.domain_adaptation_model.state_dict(), self.model_save_path)
        print(f"Model parameters saved to {self.model_save_path}")

    def plot_training_results(self):
        epochs_list = np.arange(0, len(self.train_losses), 1)
        train_losses_values = [loss for loss in self.train_losses]
        label_losses_values = [loss for loss in self.label_losses]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_list, train_losses_values, label='Total Loss', color='blue')
        plt.plot(epochs_list, label_losses_values, label='Label Loss', color='green')
        plt.plot(epochs_list, self.domain_losses, label='Domain Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_list, self.source_accuracies, label='Source Accuracy', color='blue')
        plt.plot(epochs_list, self.target_accuracies, label='Target Accuracy', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training Accuracy')
        plt.show()

    def save_model_architecture(self, file_path='model_architecture'):
        # Create a dummy input for visualization
        dummy_input = torch.randn(1, 7)  # Assuming input size is (batch_size, 7)

        # Generate a graph of the model architecture
        graph = make_dot(self.domain_adaptation_model(dummy_input), params=dict(self.domain_adaptation_model.named_parameters()))

        # Save the graph as an image file
        graph.render(file_path, format='png')
        print(f"Model architecture saved as {file_path}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--training_source_domain_data', type=str, required=True, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, required=True, help='Path to the target domain data file')
    args = parser.parse_args()

    hist_corr_dann_model = HistCorrDANNModel(args.training_source_domain_data, args.training_target_domain_data)
    hist_corr_dann_model.save_model_architecture()
    hist_corr_dann_model.train(num_epochs=10)
    hist_corr_dann_model.plot_training_results()
