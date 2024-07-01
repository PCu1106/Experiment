import csv
import sys
sys.path.append('..\\DANN_pytorch')
from DANN_pytorch import DANN
sys.path.append('..\\DANN_AE')
from DANN_AE import AutoencoderDANNModel
sys.path.append('..\\DANN_1DCAE')
from DANN_1DCAE import DANNWithCAE
sys.path.append('..\\AdapLoc')
from AdapLoc import AdapLoc
sys.path.append('..\\DANN_baseline')
from DANN_baseline import DANNWithCAEAndPA
sys.path.append('..\\DANN_CORR')
from DANN_CORR import HistCorrDANNModel
import matplotlib.pyplot as plt

# Only for DANN_AE QQ
import tensorflow as tf

def combine_history(history1, history2):
    # Convert History objects to dictionaries
    if isinstance(history1, tf.keras.callbacks.History):
        history1 = history1.history
    if isinstance(history2, tf.keras.callbacks.History):
        history2 = history2.history

    # Combine the histories
    combined_history = {
        'loss': history1['loss'] + history2['loss'],
        'label_predictor_output_loss': history1['label_predictor_output_loss'] + history2['label_predictor_output_loss'],
        'domain_classifier_output_loss': history1['domain_classifier_output_loss'] + history2['domain_classifier_output_loss'],
        'decoder_output_loss': history1['decoder_output_loss'] + history2['decoder_output_loss'],
        'label_predictor_output_accuracy': history1['label_predictor_output_accuracy'] + history2['label_predictor_output_accuracy'],
        'domain_classifier_output_accuracy': history1['domain_classifier_output_accuracy'] + history2['domain_classifier_output_accuracy'],
        'val_loss': history1['val_loss'] + history2['val_loss'],
        'val_label_predictor_output_loss': history1['val_label_predictor_output_loss'] + history2['val_label_predictor_output_loss'],
        'val_domain_classifier_output_loss': history1['val_domain_classifier_output_loss'] + history2['val_domain_classifier_output_loss'],
        'val_decoder_output_loss': history1['val_decoder_output_loss'] + history2['val_decoder_output_loss'],
        'val_label_predictor_output_accuracy': history1['val_label_predictor_output_accuracy'] + history2['val_label_predictor_output_accuracy'],
        'val_domain_classifier_output_accuracy': history1['val_domain_classifier_output_accuracy'] + history2['val_domain_classifier_output_accuracy'],
    }

    # Create a dummy History object
    combined_history_object = tf.keras.callbacks.History()
    combined_history_object.history = combined_history

    return combined_history_object


labels = []
losses = []

input_shape = 7
num_classes = 41
epochs = 500
work_dir = '.'
model_path = 'model'
batch_size = 32
training_source_domain_data = '../../data/220318/GalaxyA51/wireless_training.csv'
training_target_domain_data = '../../data/231116/GalaxyA51/wireless_training.csv'

unlabeled = True
# labeled
if unlabeled:
    data_drop_out = 0.0
else:
    data_drop_out = 0.9

# DNN
loss_weights = [1, 0]
dnn_model = DANN(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
dnn_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
dnn_model.train(unlabeled=unlabeled)
losses_values = [loss for loss in dnn_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='DNN')
labels.append('DNN')
losses.append(losses_values)

# DANN
loss_weights = [1, 1]
dann_model = DANN(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
dann_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
dann_model.train(unlabeled=unlabeled)
losses_values = [loss for loss in dann_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='DANN')
labels.append('DANN')
losses.append(losses_values)

# DANN_AE
loss_weights = [1, 2, 2]
dann_ae_model = AutoencoderDANNModel(input_shape, num_classes, work_dir)
if not unlabeled:
    dann_ae_model.load_data(training_source_domain_data, training_target_domain_data, data_drop_out)
    history = dann_ae_model.train(model_path, batch_size, epochs)
    losses_values = [sum(x * w for x, w in zip(values, loss_weights)) for values in zip(history.history['val_label_predictor_output_loss'], history.history['val_domain_classifier_output_loss'], history.history['val_decoder_output_loss'])]
    plt.plot(range(epochs), losses_values, label='DANN_AE')
    labels.append('DANN_AE')
    losses.append(losses_values)
else:
    my_epochs = int(epochs/2)
    for i in range(1):
        print(f'epoch: {i}')
        dann_ae_model.load_data(training_source_domain_data, training_target_domain_data, data_drop_out)
        history1 = dann_ae_model.train_step_1(model_path, batch_size, epochs=my_epochs)
        if i == 0:
            combined_history = history1
        else:
            combined_history = combine_history(combined_history, history1)
        dann_ae_model.load_data(training_source_domain_data, one_file=True)
        history2 = dann_ae_model.train_step_2(model_path, batch_size, epochs=my_epochs)
        combined_history = combine_history(combined_history, history2)
    losses_values = [sum(x * w for x, w in zip(values, loss_weights)) for values in zip(combined_history.history['val_label_predictor_output_loss'], combined_history.history['val_domain_classifier_output_loss'], combined_history.history['val_decoder_output_loss'])]
    plt.plot(range(epochs), losses_values, label='DANN_AE')
    labels.append('DANN_AE')
    losses.append(losses_values)


# DANN_1DCAE
loss_weights = [0.1, 0.1, 10]
dann_1dcae_model = DANNWithCAE(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
dann_1dcae_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
dann_1dcae_model.train(unlabeled=unlabeled)
losses_values = [loss for loss in dann_1dcae_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='DANN_1DCAE')
labels.append('DANN_1DCAE')
losses.append(losses_values)

# AdapLoc
loss_weights = [1, 0.01]
adaploc_model = AdapLoc(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
adaploc_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
adaploc_model.train(unlabeled=unlabeled)
losses_values = [loss for loss in adaploc_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='AdapLoc')
labels.append('AdapLoc')
losses.append(losses_values)

# Long
loss_weights = [0.1, 0.1, 10]
long_model = DANNWithCAEAndPA(num_classes, loss_weights=loss_weights, epochs=epochs, work_dir=work_dir)
long_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
long_model.train(unlabeled=unlabeled)
losses_values = [loss for loss in long_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='Long')
labels.append('Long')
losses.append(losses_values)

# HistLoc
loss_weights = [0.1, 10]
histloc_model = HistCorrDANNModel(loss_weights=loss_weights, work_dir=work_dir)
histloc_model.load_train_data(training_source_domain_data, training_target_domain_data, data_drop_out)
histloc_model.train(num_epochs=epochs, unlabeled=unlabeled)
losses_values = [loss for loss in histloc_model.val_total_losses]
plt.plot(range(epochs), losses_values, label='HistLoc')
labels.append('HistLoc')
losses.append(losses_values)



plt.legend()
plt.savefig(f"{'unlabeled' if unlabeled else 'labeled'}_convergence.png")

# Write losses to CSV
with open(f"{'unlabeled' if unlabeled else 'labeled'}_model_losses.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch'] + [f'{i}' for i in range(epochs)])
    for i, label in enumerate(labels):
        writer.writerow([label] + losses[i])

