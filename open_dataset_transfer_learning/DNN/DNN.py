import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import matplotlib.pyplot as plt


label_to_coordinate = {1: (-53.56836, 5.83747), 2: (-50.051947, 5.855995), 3: (-46.452556, 5.869534), 
                       4: (-42.853167, 5.883073), 5: (-44.659589, 7.011051), 6: (-44.751032, 11.879306), 
                       7: (-40.626278, 11.865147), 8: (-37.313205, 14.650224), 9: (-40.672748, 7.050528), 
                       10: (-39.253777, 5.896612), 11: (-35.654387, 5.91015), 12: (-32.054999, 5.923687), 
                       13: (-29.658016, 7.136601), 14: (-29.715037, 10.176074), 15: (-28.455609, 5.937224), 
                       16: (-24.856221, 5.950761), 17: (-21.256833, 5.964297), 18: (-21.06986, 12.146254), 
                       19: (-17.657445, 5.977833), 20: (-14.058057, 5.991368), 21: (-14.001059, 12.211117), 
                       22: (-10.458671, 6.004903), 23: (-6.859283, 6.018437), 24: (-6.616741, 8.258015), 
                       25: (-3.259896, 6.031971), 26: (0.33949, 6.045505), 27: (0.297446, 12.26954), 
                       28: (3.938876, 6.059038), 29: (7.538262, 6.07257), 30: (7.525253, 12.321256), 
                       31: (11.137647, 6.086102), 32: (14.737032, 6.099633), 33: (14.705246, 2.374095), 
                       34: (14.717918, 12.321068), 35: (18.336417, 6.113164), 36: (21.935801, 6.126695), 
                       37: (21.899795, 12.339099), 38: (21.921602, 2.423358), 39: (36.238672, 6.108184), 
                       40: (32.733952, 6.167284), 41: (31.779903, 2.442016), 42: (29.134569, 6.153754), 
                       43: (25.535185, 6.140225), 44: (38.088066, 7.394376), 45: (38.040971, 10.951591), 
                       46: (37.993873, 14.508804), 47: (29.037591, 12.318132), 48: (44.93136, 6.314889), 
                       49: (44.816113, 13.54513)}

class IndoorLocalizationDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        data = pd.read_csv(os.path.join(self.data_path, 'data.csv'), header=0)
        labels = pd.read_csv(os.path.join(self.data_path, 'labels.csv'), header=0)
        return data, labels

    def preprocess_data(self, data, labels, test_size=0.1):
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

class IndoorLocalizationDNN:
    def __init__(self, input_dim, output_dim, model_architecture, model_path):
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
    data_path = r'D:\Experiment\data\UM_DSI_DB_v1.0.0_lite\data\site_surveys\2019-06-11'
    data_loader = IndoorLocalizationDataLoader(data_path)
    data, labels = data_loader.load_data()
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = data_loader.preprocess_data(data, labels)
    # Modify labels to be 0-based
    y_train -= 1
    y_val -= 1

    model_architecture = [
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(49)
    ]
    model_path = 'my_model.h5'
    model = IndoorLocalizationDNN(X_train_scaled.shape[1], y_train.nunique(), model_architecture, model_path)
    history = model.train(X_train_scaled, y_train, X_val_scaled, y_val, epochs=10)
    model.plot_loss_and_accuracy_history()

    model.load_model(model_path)
    predictions = model.predict(X_test_scaled)
    # Modify labels to be 1-based
    results = pd.DataFrame({'label': y_test['label'], 'pred': predictions.argmax(axis=1) + 1})
    results.to_csv('results.csv', index=False)

    # 讀取結果
    results = pd.read_csv('results.csv')

    # 計算每個預測點的距離誤差
    errors = []
    for idx, row in results.iterrows():
        pred_label = row['pred']
        pred_coord = label_to_coordinate[pred_label]
        actual_coord = label_to_coordinate[row['label']]
        distance_error = np.linalg.norm(np.array(pred_coord) - np.array(actual_coord))
        errors.append(distance_error)

    # 計算平均距離誤差
    mean_distance_error = np.mean(errors)
    print(f'MDE: {mean_distance_error}')
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test['label']-1)
    print(f'Test accuracy: {test_acc}')
