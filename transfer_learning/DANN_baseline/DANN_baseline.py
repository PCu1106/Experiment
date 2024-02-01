'''
python .\DANN_baseline.py \
    --training_source_domain_data D:\Experiment\data\220318\GalaxyA51\wireless_training.csv \
    --training_target_domain_data D:\Experiment\data\231116\GalaxyA51\wireless_training.csv \
    --model_path 220318_231116.pth \
    --work_dir 220318_231116\0.1_0.1_10
python .\DANN_1DCAE.py \
    --testing_data_list D:\Experiment\data\231116\GalaxyA51\routes \
                        D:\Experiment\data\220318\GalaxyA51\routes \
                        D:\Experiment\data\231117\GalaxyA51\routes \
    --model_path 220318_231116.pth \
    --work_dir 220318_231116\0.1_0.1_10
python ..\..\model_comparison\evaluator.py \
    --model_name DANN_CORR \
    --directory 220318_231116\0.1_10_0.0 \
    --source_domain 220318 \
    --target_domain 231116
'''
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch
import torch.optim as optim
import sys
sys.path.append('..\\DANN_1DCAE')
from DANN_1DCAE import DANNWithCAE
sys.path.append('..\\DANN_pytorch')
from DANN_pytorch import GRL
from itertools import cycle
import torch.nn.functional as F
import torch.nn as nn
import os
from torchsummary import summary
import math
import pandas as pd
import argparse
import numpy as np
sys.path.append('..\\..\\model_comparison')
from walk_definitions import walk_class
from evaluator import Evaluator
sys.path.append('..\\model_comparison')
from drop_out_plot import plot_lines

# Custom PyTorch module for PassiveAggressiveClassifier
class PassiveAggressiveModule(nn.Module):
    def __init__(self):
        super(PassiveAggressiveModule, self).__init__()
        self.model = PassiveAggressiveClassifier()
        self.is_fitted = False

    def forward(self, x):
        # You can define the forward behavior if needed
        pass

    def fit(self, X, y):
        # Fit the underlying scikit-learn model
        self.model.fit(X, y)
        self.is_fitted = True

    def partial_fit(self, X, y, classes=None):
        # if self.is_fitted:
            # Perform partial fit
        self.model.partial_fit(X, y, classes=classes)
        # else:
        #     self.fit(X, y)

    def predict(self, x):
        # Add a predict method using the underlying scikit-learn model
        return self.model.predict(x)

class DANNWithCAEAndPA(DANNWithCAE):
    def __init__(self, num_classes, epochs, model_save_path='saved_model.pth', loss_weights=None, work_dir=None):
        super(DANNWithCAEAndPA, self).__init__(num_classes, epochs, model_save_path, loss_weights, work_dir)

        # Replace the ClassClassifier with PassiveAggressiveClassifier
        self.class_classifier = PassiveAggressiveModule()

        # Modify optimizer initialization to include all parameters
        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.domain_classifier.parameters()),
            lr=0.001
        )

    def _initialize_metrics(self):
        super()._initialize_metrics()  # Call the base class method

    def forward(self, x, alpha=1.0):
        encoded, decoded = self.feature_extractor(x)

        # Domain classification loss
        domain_features = GRL.apply(encoded, alpha)
        domain_output = self.domain_classifier(domain_features)

        # # Class prediction using PassiveAggressiveClassifier
        # class_labels = torch.tensor(self.class_classifier.predict(encoded.detach().numpy()))
        
        # # Convert class labels to one-hot encoding
        # class_one_hot = F.one_hot(class_labels.long(), num_classes=self.num_classes).float()

        return domain_output, encoded, decoded

    def train(self, unlabeled=False):
        for epoch in range(self.epochs):
            loss_list, acc_list = self._run_epoch([self.source_train_loader, self.target_train_loader], training=True, unlabeled=unlabeled)

            self.total_losses.append(loss_list[0])
            self.label_losses.append(loss_list[1])
            self.domain_losses.append(loss_list[2])
            self.reconstruction_losses.append(loss_list[3])
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
                self.val_reconstruction_losses.append(val_loss_list[3])
                self.val_total_accuracies.append(val_acc_list[0])
                self.val_source_accuracies.append(val_acc_list[1])
                self.val_target_accuracies.append(val_acc_list[2])
                self.val_total_domain_accuracies.append(val_acc_list[3])
                self.val_source_domain_accuracies.append(val_acc_list[4])
                self.val_target_domain_accuracies.append(val_acc_list[5])
            
            print(f'Epoch [{epoch+1}/{self.epochs}], loss: {self.total_losses[-1]:.4f}, label loss: {self.label_losses[-1]:.4f}, domain loss: {self.domain_losses[-1]:.4f}, reconstruction loss: {self.reconstruction_losses[-1]:.4f}, acc: {self.total_accuracies[-1]:.4f},\nval_loss: {self.val_total_losses[-1]:.4f}, val_label loss: {self.val_label_losses[-1]:.4f}, val_domain loss: {self.val_domain_losses[-1]:.4f}, val_reconstruction loss: {self.val_reconstruction_losses[-1]:.4f}, val_acc: {self.val_total_accuracies[-1]:.4f}')
            
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
            source_domain_output, source_encoded, source_decoded = self.forward(source_features, self.alpha)
            target_domain_output, target_encoded, target_decoded = self.forward(target_features, self.alpha)

            # Reconstruction loss
            reconstruction_loss_source = F.mse_loss(source_decoded, source_features)
            reconstruction_loss_target = F.mse_loss(target_decoded, target_features)
            reconstruction_loss = (reconstruction_loss_source + reconstruction_loss_target) / 2

            # Convert PyTorch tensors to NumPy arrays for sklearn
            source_np_encoded = source_encoded.detach().numpy()
            source_np_labels = source_labels.numpy()
            target_np_encoded = target_encoded.detach().numpy()
            target_np_labels = target_labels.numpy()
            # Fit PassiveAggressiveClassifier
            self.class_classifier.partial_fit(source_np_encoded, source_np_labels, np.arange(41))
            self.class_classifier.partial_fit(target_np_encoded, target_np_labels, np.arange(41))

            # Class prediction using PassiveAggressiveClassifier
            source_labels_pred = torch.tensor(self.class_classifier.predict(source_np_encoded))
            target_labels_pred = torch.tensor(self.class_classifier.predict(target_np_encoded))
            
            # Convert class labels to one-hot encoding
            source_labels_pred = F.one_hot(source_labels_pred.long(), num_classes=self.num_classes).float()
            target_labels_pred = F.one_hot(target_labels_pred.long(), num_classes=self.num_classes).float()

            label_loss_source = nn.CrossEntropyLoss()(source_labels_pred, source_labels)
            label_loss_target = nn.CrossEntropyLoss()(target_labels_pred, target_labels)
            if unlabeled:
                label_loss = label_loss_source
            else:
                label_loss = (label_loss_source + label_loss_target) / 2

            source_domain_loss = nn.CrossEntropyLoss()(source_domain_output, torch.ones(source_domain_output.size(0)).long())
            target_domain_loss = nn.CrossEntropyLoss()(target_domain_output, torch.zeros(target_domain_output.size(0)).long())
            domain_loss = source_domain_loss + target_domain_loss

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

            _, source_domain_preds = torch.max(source_domain_output, 1)
            source_domain_correct_predictions += (source_domain_preds == 1).sum().item()  # Assuming 1 for source domain
            source_domain_total_samples += source_domain_output.size(0)
            source_domain_accuracy = source_domain_correct_predictions / source_domain_total_samples

            _, target_domain_preds = torch.max(target_domain_output, 1)
            target_domain_correct_predictions += (target_domain_preds == 0).sum().item()  # Assuming 0 for target domain
            target_domain_total_samples += target_domain_output.size(0)
            target_domain_accuracy = target_domain_correct_predictions / target_domain_total_samples

            loss_list = [total_loss.item(), label_loss.item(), domain_loss, reconstruction_loss]
            acc_list = [
                (source_accuracy + target_accuracy) / 2,
                source_accuracy,
                target_accuracy,
                (source_domain_accuracy + target_domain_accuracy) / 2,
                source_domain_accuracy,
                target_domain_accuracy
            ]
        return loss_list, acc_list
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--training_source_domain_data', type=str, help='Path to the source domain data file')
    parser.add_argument('--training_target_domain_data', type=str, help='Path to the target domain data file')
    parser.add_argument('--testing_data_list', nargs='+', type=str, help='List of testing data paths')
    parser.add_argument('--model_path', type=str, default='my_model.pth', help='path of .pth file of model')
    parser.add_argument('--work_dir', type=str, default='DANN_CORR', help='create new directory to save result')
    args = parser.parse_args()

    num_classes = 41
    epochs = 5
    loss_weights = [0.1, 0.1, 10]
    unlabeled = True
    
    domain1_result = []
    domain2_result = []
    domain3_result = []

    data_drop_out_list = np.arange(0.0, 0.05, 0.1)
    
    for data_drop_out in data_drop_out_list:
        # 創建 DANNModel    
        dann_model = DANNWithCAEAndPA(num_classes, model_save_path=args.model_path, loss_weights=loss_weights, epochs=epochs, work_dir=f'{args.work_dir}_{data_drop_out:.1f}')
        # summary(dann_model, (7,))
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
    