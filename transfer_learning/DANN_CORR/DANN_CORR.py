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
    def __init__(self, source_data_path, target_data_path, model_save_path='saved_model.pth', loss_weights=None):
        self.source_dataset = IndoorLocalizationDataset(source_data_path)
        self.target_dataset = IndoorLocalizationDataset(target_data_path)
        self.batch_size = 32
        self.source_data_loader = DataLoader(self.source_dataset, batch_size=self.batch_size, shuffle=True)
        self.target_data_loader = DataLoader(self.target_dataset, batch_size=self.batch_size, shuffle=True)
        self.loss_weights = loss_weights if loss_weights is not None else [0.1, 10]

        self._initialize_data_loaders()
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_metrics()

        self.model_save_path = model_save_path
        self.best_val_total_loss = float('inf')  # Initialize with positive infinity

    def _initialize_data_loaders(self):
        # Split source data into training and validation sets
        source_train, source_val = train_test_split(self.source_dataset, test_size=0.2, random_state=42)
        self.source_train_loader = DataLoader(source_train, batch_size=self.batch_size, shuffle=True)
        self.source_val_loader = DataLoader(source_val, batch_size=self.batch_size, shuffle=False)

        # Split target data into training and validation sets
        target_train, target_val = train_test_split(self.target_dataset, test_size=0.2, random_state=42)
        self.target_train_loader = DataLoader(target_train, batch_size=self.batch_size, shuffle=True)
        self.target_val_loader = DataLoader(target_val, batch_size=self.batch_size, shuffle=False)

    def _initialize_model(self):
        self.feature_extractor = FeatureExtractor()
        self.label_predictor = LabelPredictor(16, num_classes=41)
        self.domain_adaptation_model = DomainAdaptationModel(self.feature_extractor, self.label_predictor)

    def _initialize_optimizer(self):
        self.optimizer = optim.Adam(self.domain_adaptation_model.parameters(), lr=0.001)
        self.domain_criterion = nn.CrossEntropyLoss()

    def _initialize_metrics(self):
        self.train_losses, self.label_losses, self.domain_losses = [], [], []
        self.source_accuracies, self.target_accuracies, self.total_accuracies = [], [], []
        self.val_train_losses, self.val_label_losses, self.val_domain_losses = [], [], []
        self.val_source_accuracies, self.val_target_accuracies, self.val_total_accuracies = [], [], []

    def domain_invariance_loss(self, source_hist, target_hist):
        correlation = cv2.compareHist(source_hist, target_hist, cv2.HISTCMP_CORREL)
        return 1 - correlation

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            source_correct_predictions = 0
            source_total_samples = 0
            target_correct_predictions = 0
            target_total_samples = 0
            # Training
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

                total_loss = self.loss_weights[0] * label_loss + self.loss_weights[1] * domain_loss

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
            self.total_accuracies.append((source_accuracy + target_accuracy) / 2)
            # Validation
            with torch.no_grad():
                val_source_correct_predictions = 0
                val_source_total_samples = 0
                val_target_correct_predictions = 0
                val_target_total_samples = 0

                for val_source_batch, val_target_batch in zip(self.source_val_loader, self.target_val_loader):
                    val_source_features, val_source_labels = val_source_batch
                    val_target_features, val_target_labels = val_target_batch

                    val_source_features, val_source_labels_pred = self.domain_adaptation_model(val_source_features)
                    val_target_features, val_target_labels_pred = self.domain_adaptation_model(val_target_features)

                    val_label_loss_source = self.domain_criterion(val_source_labels_pred, val_source_labels)
                    val_label_loss_target = self.domain_criterion(val_target_labels_pred, val_target_labels)
                    val_label_loss = (val_label_loss_source + val_label_loss_target) / 2

                    val_source_hist = cv2.calcHist([val_source_features.detach().numpy().flatten()], [0], None, [100], [0, 1])
                    val_target_hist = cv2.calcHist([val_target_features.detach().numpy().flatten()], [0], None, [100], [0, 1])
                    val_domain_loss = self.domain_invariance_loss(val_source_hist, val_target_hist)

                    val_total_loss = self.loss_weights[0] * val_label_loss + self.loss_weights[1] * val_domain_loss

                    _, val_source_preds = torch.max(val_source_labels_pred, 1)
                    val_source_correct_predictions += (val_source_preds == val_source_labels).sum().item()
                    val_source_total_samples += val_source_labels.size(0)

                    _, val_target_preds = torch.max(val_target_labels_pred, 1)
                    val_target_correct_predictions += (val_target_preds == val_target_labels).sum().item()
                    val_target_total_samples += val_target_labels.size(0)

                    val_source_accuracy = val_source_correct_predictions / val_source_total_samples
                    val_target_accuracy = val_target_correct_predictions / val_target_total_samples

                self.val_train_losses.append(val_total_loss.item())
                self.val_label_losses.append(val_label_loss.item())
                self.val_domain_losses.append(val_domain_loss)
                self.val_source_accuracies.append(val_source_accuracy)
                self.val_target_accuracies.append(val_target_accuracy)
                self.val_total_accuracies.append((val_source_accuracy + val_target_accuracy) / 2)
                # print(f'Validation Epoch [{epoch+1}/{num_epochs}], Total Loss: {val_total_loss}, Label Loss: {val_label_loss}, Domain Loss: {val_domain_loss}, Source Accuracy: {val_source_accuracy}, Target Accuracy: {val_target_accuracy}')
            
            # print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss}, Label Loss: {label_loss}, Domain Loss: {domain_loss}, Source Accuracy: {source_accuracy}, Target Accuracy: {target_accuracy}')
            total_acc = (source_accuracy + target_accuracy) / 2
            val_total_acc = (val_source_accuracy + val_target_accuracy) / 2
            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {total_loss:.4f}, label loss: {label_loss:4f}, domain loss: {domain_loss:4f}, acc: {total_acc:.4f},\nval_loss: {val_total_loss:.4f}, val_label loss: {val_label_loss:4f}, val_domain loss: {val_domain_loss:4f}, val_acc: {val_total_acc:.4f}')
            
            # Check if the current total loss is the best so far
            if val_total_loss < self.best_val_total_loss:
                # Save the model parameters
                print(f'val_total_loss: {val_total_loss:.4f} < best_val_total_loss: {self.best_val_total_loss:.4f}', end=', ')
                self.save_model()
                self.best_val_total_loss = val_total_loss

    def save_model(self):
        torch.save(self.domain_adaptation_model.state_dict(), self.model_save_path)
        print(f"Model parameters saved to {self.model_save_path}")

    def plot_training_results(self):
        epochs_list = np.arange(0, len(self.train_losses), 1)
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
    loss_weights = [1, 1]
    hist_corr_dann_model = HistCorrDANNModel(args.training_source_domain_data, args.training_target_domain_data, loss_weights=loss_weights)
    hist_corr_dann_model.save_model_architecture()
    hist_corr_dann_model.train(num_epochs=10)
    hist_corr_dann_model.plot_training_results()
