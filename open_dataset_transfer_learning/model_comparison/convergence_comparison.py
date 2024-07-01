import csv
import sys
import time
sys.path.append('..\\DANN_pytorch')
from DANN_pytorch import DANN
sys.path.append('..\\DANN_AE')
from DANN_AE import DANNWithAE
sys.path.append('..\\DANN_1DCAE')
from DANN_1DCAE import DANNWithCAE
sys.path.append('..\\AdapLoc')
from AdapLoc import AdapLoc
sys.path.append('..\\DANN_baseline')
from DANN_baseline import DANNWithCAEAndPA
sys.path.append('..\\DANN_CORR')
from DANN_CORR import HistCorrDANNModel
import matplotlib.pyplot as plt

labels = []
losses = []
training_times = []

input_shape = 168
num_classes = 49
epochs = 100
work_dir = '.'
model_path = 'model'
batch_size = 32
training_source_domain_data = '../../data/UM_DSI_DB_v1.0.0_lite/data/tony_data/2019-06-11/wireless_training.csv'
training_target_domain_data = '../../data/UM_DSI_DB_v1.0.0_lite/data/tony_data/2020-02-19/wireless_training.csv'

unlabeled = False
# labeled
if unlabeled:
    data_drop_out = 0.0
else:
    data_drop_out = 0.9

# DNN
loss_weights = [1, 0]
dnn_model = DANN(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
dnn_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
start_time = time.time()
dnn_model.train(unlabeled=unlabeled)
end_time = time.time()
losses_values = [loss for loss in dnn_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='DNN')
labels.append('DNN')
losses.append(losses_values)
training_times.append(end_time - start_time)

# DANN
loss_weights = [1, 1]
dann_model = DANN(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
dann_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
start_time = time.time()
dann_model.train(unlabeled=unlabeled)
end_time = time.time()
losses_values = [loss for loss in dann_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='DANN')
labels.append('DANN')
losses.append(losses_values)
training_times.append(end_time - start_time)

# DANN_AE
loss_weights = [0.1, 0.2, 0.2]
dann_ae_model = DANNWithAE(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
dann_ae_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
start_time = time.time()
dann_ae_model.train(unlabeled=unlabeled)
end_time = time.time()
losses_values = [loss for loss in dann_ae_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='DANN_AE')
labels.append('DANN_AE')
losses.append(losses_values)
training_times.append(end_time - start_time)


# DANN_1DCAE
loss_weights = [0.1, 0.1, 10]
dann_1dcae_model = DANNWithCAE(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
dann_1dcae_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
start_time = time.time()
dann_1dcae_model.train(unlabeled=unlabeled)
end_time = time.time()
losses_values = [loss for loss in dann_1dcae_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='DANN_1DCAE')
labels.append('DANN_1DCAE')
losses.append(losses_values)
training_times.append(end_time - start_time)

# AdapLoc
loss_weights = [1, 0.01]
adaploc_model = AdapLoc(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
adaploc_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
start_time = time.time()
adaploc_model.train(unlabeled=unlabeled)
end_time = time.time()
losses_values = [loss for loss in adaploc_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='AdapLoc')
labels.append('AdapLoc')
losses.append(losses_values)
training_times.append(end_time - start_time)

# Long
loss_weights = [0.1, 0.1, 10]
long_model = DANNWithCAEAndPA(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
long_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
start_time = time.time()
long_model.train(unlabeled=unlabeled)
end_time = time.time()
losses_values = [loss for loss in long_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='Long')
labels.append('Long')
losses.append(losses_values)
training_times.append(end_time - start_time)

# HistLoc
loss_weights = [0.1, 10]
histloc_model = HistCorrDANNModel(loss_weights=loss_weights, work_dir=work_dir)
histloc_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
start_time = time.time()
histloc_model.train(num_epochs=epochs, unlabeled=unlabeled)
end_time = time.time()
losses_values = [loss for loss in histloc_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='HistLoc')
labels.append('HistLoc')
losses.append(losses_values)
training_times.append(end_time - start_time)



plt.legend()
plt.savefig(f"convergence/{'unlabeled' if unlabeled else 'labeled'}_convergence.png")

# Write losses to CSV
with open(f"convergence/{'unlabeled' if unlabeled else 'labeled'}_model_losses.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch'] + [f'{i}' for i in range(epochs)])
    for i, label in enumerate(labels):
        writer.writerow([label] + losses[i])

with open(f"convergence/{'unlabeled' if unlabeled else 'labeled'}_model_training_time.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    for i, label in enumerate(labels):
        writer.writerow([label] + [training_times[i]])