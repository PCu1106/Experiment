import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import matplotlib.pyplot as plt
import os



class IndoorLocalizationDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        data = pd.read_csv(os.path.join(self.data_path), header=0)
        self.X = data.iloc[:, 1:]
        self.y = data['label']
        return self.X, self.y
    
    def add_data(self, data_path, ratio=1.0):
        data = pd.read_csv(os.path.join(data_path), header=0)
        X_new = data.iloc[:, 1:]
        y_new = data['label']
        if ratio < 1.0:
            X_new, _, y_new, _ = train_test_split(X_new, y_new, train_size=ratio, stratify=y_new)
        self.X = pd.concat([self.X, X_new], ignore_index=True)
        self.y = pd.concat([self.y, y_new], ignore_index=True)
        return self.X, self.y

    def preprocess_data(self, data, labels, test_size=0.2):
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=test_size)
        return X_train_val, X_test, y_train_val, y_test

class IndoorLocalizationDNN:
    def __init__(self, input_dim=None, output_dim=None, model_architecture=None, model_path=None):
        self.model = self.build_model(input_dim, output_dim, model_architecture)
        self.model_path = model_path

    def build_model(self, input_dim, output_dim, model_architecture):
        model = tf.keras.Sequential(model_architecture)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        # 使用 ModelCheckpoint 回調函數保存最低 loss 的模型
        checkpoint_callback = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val, y_val), callbacks=checkpoint_callback)
        return self.history
    
    def plot_loss_and_accuracy_history(self):
        plt.figure(figsize=(12, 6))

        # 画 loss 图
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 画 accuracy 图
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # 添加整张图的标题
        plt.suptitle(f"Training Curves")

        # 保存 loss 图
        plt.savefig("loss_and_accuracy.png")
        plt.clf()

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

if __name__ == '__main__':
    source_data_path = r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\site_surveys\2019-06-11\wireless_training.csv'
    target_data_path = r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\site_surveys\2019-12-11\wireless_training.csv'
    data_loader = IndoorLocalizationDataLoader(source_data_path)
    data_loader.load_data()
    data, labels = data_loader.add_data(target_data_path, ratio=0.1)
    X_train_scaled, X_val_scaled, y_train, y_val = data_loader.preprocess_data(data, labels)
    source_test_data_path = r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\site_surveys\2019-06-11\wireless_testing.csv'
    source_test_data_loader = IndoorLocalizationDataLoader(source_test_data_path)
    source_X_test, source_y_test = source_test_data_loader.load_data()
    target_test_data_path = r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\site_surveys\2019-12-11\wireless_testing.csv'
    target_test_data_loader = IndoorLocalizationDataLoader(target_test_data_path)
    target_X_test, target_y_test = target_test_data_loader.load_data()

    # Modify labels to be 0-based
    y_train -= 1
    y_val -= 1

    model_architecture = [
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(49)
    ]
    model_path = 'my_model.h5'
    model = IndoorLocalizationDNN(X_train_scaled.shape[1], y_train.nunique(), model_architecture, model_path)
    history = model.train(X_train_scaled, y_train, X_val_scaled, y_val, epochs=10)
    model.plot_loss_and_accuracy_history()

    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    model.load_model(model_path)
    source_predictions = model.predict(source_X_test)
    target_predictions = model.predict(target_X_test)

    # Modify labels to be 1-based
    results = pd.DataFrame({'label': source_y_test, 'pred': source_predictions.argmax(axis=1) + 1})
    results.to_csv(r'predictions/0611_results.csv', index=False)
    results = pd.DataFrame({'label': target_y_test, 'pred': target_predictions.argmax(axis=1) + 1})
    results.to_csv(r'predictions/1211_results.csv', index=False)
    
    source_test_loss, source_test_acc = model.evaluate(source_X_test, source_y_test-1)
    print(f'Source test accuracy: {source_test_acc}')
    target_test_loss, target_test_acc = model.evaluate(target_X_test, target_y_test-1)
    print(f'Target test accuracy: {target_test_acc}')