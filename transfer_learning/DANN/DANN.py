import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


@tf.custom_gradient
def GradientReversalOperator(x):
	def grad(dy):
		return -1 * dy
	return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):
	def __init__(self):
		super(GradientReversalLayer, self).__init__()
		
	def call(self, inputs):
		return GradientReversalOperator(inputs)

class DANNModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def load_data(self, source_domain_file, target_domain_file):
        # Read source domain data
        source_domain = pd.read_csv(source_domain_file)
        source_domain_data = source_domain.iloc[:, 1:]
        source_domain_labels = source_domain['label']
        source_domain_labels = source_domain_labels - 1
        source_domain_labels = to_categorical(source_domain_labels, num_classes=self.num_classes)

        # Read target domain data
        target_domain = pd.read_csv(target_domain_file)
        target_domain_data = target_domain.iloc[:, 1:]
        target_domain_labels = target_domain['label']
        target_domain_labels = target_domain_labels - 1
        target_domain_labels = to_categorical(target_domain_labels, num_classes=self.num_classes)

        return source_domain_data, source_domain_labels, target_domain_data, target_domain_labels

    def build_model(self):
        # Input layer
        input_data = layers.Input(shape=self.input_shape, name='input_data')

        # Feature extractor
        feature_extractor = self.build_feature_extractor(input_data)

        # Label predictor
        label_predictor = self.build_label_predictor(feature_extractor)

        # Domain classifier
        domain_classifier = self.build_domain_classifier(feature_extractor)

        # Build DANN model
        dann_model = models.Model(inputs=input_data, outputs=[label_predictor, domain_classifier])

        return dann_model

    def build_feature_extractor(self, input_data):
        # Shared feature extraction layers
        x = layers.Dense(8, activation='relu')(input_data)
        # x = layers.Dropout(0.5)(x)
        x = layers.Dense(16, activation='relu')(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(32, activation='relu')(x)

        return x

    def build_label_predictor(self, feature_extractor):
        # Label predictor layers
        x = layers.Dense(32, activation='relu')(feature_extractor)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(32, activation='relu')(x)
        # x = layers.Dropout(0.5)(x)
        label_predictor_output = layers.Dense(self.num_classes, activation='softmax', name='label_predictor')(x)

        return label_predictor_output

    def build_domain_classifier(self, feature_extractor):
        # Domain classifier layers
        x = GradientReversalLayer()(feature_extractor)
        x = layers.Dense(8, activation='relu')(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(4, activation='relu')(x)
        # x = layers.Dropout(0.5)(x)
        domain_classifier_output = layers.Dense(1, activation='sigmoid', name='domain_classifier')(x)

        return domain_classifier_output

    def plot_training_history(self, history):
        # Plot training and validation loss for Label Predictor
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history.history['label_predictor_loss'], label='Train Label Predictor Loss')
        plt.plot(history.history['val_label_predictor_loss'], label='Validation Label Predictor Loss')
        plt.title('Label Predictor Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history.history['label_predictor_accuracy'], label='Train Label Predictor Accuracy')
        plt.plot(history.history['val_label_predictor_accuracy'], label='Validation Label Predictor Accuracy')
        plt.title('Label Predictor Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot training and validation loss for Domain Classifier
        plt.subplot(2, 2, 3)
        plt.plot(history.history['domain_classifier_loss'], label='Train Domain Classifier Loss')
        plt.plot(history.history['val_domain_classifier_loss'], label='Validation Domain Classifier Loss')
        plt.title('Domain Classifier Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(history.history['domain_classifier_accuracy'], label='Train Domain Classifier Accuracy')
        plt.plot(history.history['val_domain_classifier_accuracy'], label='Validation Domain Classifier Accuracy')
        plt.title('Domain Classifier Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('loss_and_accuracy.png')

    def train(self, source_domain_data, source_domain_labels, target_domain_data, target_domain_labels, batch_size=32, epochs=50):
        # Combine source and target domain data and labels
        combined_data = np.vstack([source_domain_data, target_domain_data])
        combined_labels = np.vstack([source_domain_labels, target_domain_labels])

        # Create domain labels (1 for source domain, 0 for target domain)
        combined_domain_labels = np.concatenate([np.ones(len(source_domain_data)), np.zeros(len(target_domain_data))])

        # Shuffle the data
        indices = np.arange(len(combined_data))
        np.random.shuffle(indices)
        combined_data = combined_data[indices]
        combined_labels_one_hot = combined_labels[indices]
        combined_domain_labels = combined_domain_labels[indices]

        # Compile the DANN model
        self.model.compile(optimizer='adam',
                           loss={'label_predictor': 'categorical_crossentropy', 'domain_classifier': 'binary_crossentropy'},
                           loss_weights={'label_predictor': 1.0, 'domain_classifier': 1.0},
                           metrics={'label_predictor': 'accuracy', 'domain_classifier': 'accuracy'})

        # Define the ModelCheckpoint callback to save the model with the minimum total loss
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        # Train the DANN model with validation split
        history = self.model.fit(combined_data, {'label_predictor': combined_labels_one_hot, 'domain_classifier': combined_domain_labels},
                                 batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[checkpoint])

        # Plot training history
        self.plot_training_history(history)

        return history


if __name__ == "__main__":
    # 使用 argparse 處理命令列參數
    parser = argparse.ArgumentParser(description='Train DANN Model')
    parser.add_argument('--source_domain_file', type=str, help='Path to the source domain data file', required=True)
    parser.add_argument('--target_domain_file', type=str, help='Path to the target domain data file', required=True)
    args = parser.parse_args()

    # 設定 input shape 和 num_classes
    input_shape = (7,)
    num_classes = 41  # 這裡的數字要根據你的問題設定

    # 創建 DANNModel
    dann_model = DANNModel(input_shape, num_classes)

    # 讀取資料
    source_domain_data, source_domain_labels, target_domain_data, target_domain_labels = dann_model.load_data(args.source_domain_file, args.target_domain_file)
    
    # 訓練模型
    batch_size=32
    epochs=100
    history = dann_model.train(source_domain_data, source_domain_labels, target_domain_data, target_domain_labels, batch_size=batch_size, epochs=epochs)

    # 顯示模型摘要
    # dann_model.model.summary()

