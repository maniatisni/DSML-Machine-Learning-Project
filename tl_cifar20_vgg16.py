from __future__ import absolute_import, division, print_function, unicode_literals  # legacy compatibility

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import wandb

import numpy as np
import pandas as pd
import wandb
from wandb.keras import WandbCallback
import os
import matplotlib.pyplot as plt

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5
    },
    'parameters': {
        'batch_size': {
            'values': [8, 16, 32, 64, 128, 256, 512]
        },
        'learning_rate': {
            'values': [0.01, 0.005, 0.001, 0.0005, 0.0001]
        },
        'fc1': {
            'values': [32, 64, 128, 256, 512]
        },
        'dropout': {
            'values': [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        },
        'epochs': {
            'values': [10, 20, 30, 40, 50]
        }
    }
}


# @title
# helper functions

# select from from_list elements with index in index_list
def select_from_list(from_list, index_list):
    filtered_list = [from_list[i] for i in index_list]
    return (filtered_list)


# append in filtered_list the index of each element of unfilterd_list if it exists in in target_list
def get_ds_index(unfiliterd_list, target_list):
    index = 0
    filtered_list = []
    for i_ in unfiliterd_list:
        if i_[0] in target_list:
            filtered_list.append(index)
        index += 1
    return (filtered_list)


# select a url for a unique subset of CIFAR-100 with 20, 40, 60, or 80 classes
def select_classes_number(classes_number=20):
    cifar100_20_classes_url = "https://pastebin.com/raw/nzE1n98V"
    cifar100_40_classes_url = "https://pastebin.com/raw/zGX4mCNP"
    cifar100_60_classes_url = "https://pastebin.com/raw/nsDTd3Qn"
    cifar100_80_classes_url = "https://pastebin.com/raw/SNbXz700"
    if classes_number == 20:
        return cifar100_20_classes_url
    elif classes_number == 40:
        return cifar100_40_classes_url
    elif classes_number == 60:
        return cifar100_60_classes_url
    elif classes_number == 80:
        return cifar100_80_classes_url
    else:
        return -1


# load the entire dataset
(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

# REPLACE WITH YOUR C NUMBER
team_sead = 119

# select the number of classes
cifar100_classes_url = select_classes_number()

# @title
my_classes = pd.read_csv(cifar100_classes_url, sep=',', header=None)
CIFAR100_LABELS_LIST = \
    pd.read_csv('https://pastebin.com/raw/qgDaNggt', sep=',', header=None).astype(str).values.tolist()[0]

our_index = my_classes.iloc[team_sead, :].values.tolist()
our_classes = select_from_list(CIFAR100_LABELS_LIST, our_index)
train_index = get_ds_index(y_train_all, our_index)
test_index = get_ds_index(y_test_all, our_index)

x_train_ds = np.asarray(select_from_list(x_train_all, train_index))
y_train_ds = np.asarray(select_from_list(y_train_all, train_index))
x_test_ds = np.asarray(select_from_list(x_test_all, test_index))
y_test_ds = np.asarray(select_from_list(y_test_all, test_index))

# print our classes
print(our_classes)
CLASSES_NUM = len(our_classes)

# @title
# get (train) dataset dimensions
data_size, img_rows, img_cols, img_channels = x_train_ds.shape

# set validation set percentage (wrt the training set size)
validation_percentage = 0.15
val_size = round(validation_percentage * data_size)

# Reserve val_size samples for validation and normalize all values
x_val = x_train_ds[-val_size:] / 255
y_val = y_train_ds[-val_size:]
x_train = x_train_ds[:-val_size] / 255
y_train = y_train_ds[:-val_size]
x_test = x_test_ds / 255
y_test = y_test_ds

print(len(x_val))

# summarize loaded dataset
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Validation: X=%s, y=%s' % (x_val.shape, y_val.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))


# get class label from class index
def class_label_from_index(fine_category):
    return (CIFAR100_LABELS_LIST[fine_category.item(0)])


# we user prefetch https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
# see also AUTOTUNE
# the dataset is now "infinite"

BATCH_SIZE = 128
AUTOTUNE = tf.data.experimental.AUTOTUNE  # https://www.tensorflow.org/guide/data_performance


def _input_fn(x, y, BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=data_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def train():
    # Specify the hyperparameter to be tuned along with
    # an initial value

    # Initialize wandb with a sample project name
    wandb.init(config=sweep_config)

    train_ds = _input_fn(x_train, y_train, wandb.config.batch_size)  # PrefetchDataset object
    validation_ds = _input_fn(x_val, y_val, wandb.config.batch_size)  # PrefetchDataset object
    test_ds = _input_fn(x_test, y_test, wandb.config.batch_size)  # PrefetchDataset object

    # Iniialize model with hyperparameters
    tf.keras.backend.clear_session()
    model = VGG16(wandb.config.dropout, wandb.config.fc1)
    # Compile the model
    opt = tf.keras.optimizers.Adam(
        learning_rate=wandb.config.learning_rate)  # optimizer with different learning rate specified by config
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    _ = model.fit(train_ds,
                  epochs=wandb.config.epochs,
                  steps_per_epoch=50, validation_steps=30,
                  validation_data=validation_ds,
                  callbacks=[WandbCallback()])  # WandbCallback to automatically track metrics

    # Evaluate
    loss, accuracy = model.evaluate(test_ds, steps=10)
    print('Test Error Rate: ', round((1 - accuracy) * 100, 2))
    wandb.log({'Test Error Rate': round((1 - accuracy) * 100, 2)})  # wandb.log to tr


# transfer learning: VGG16 trained on ImageNet without the top layer

def VGG16(dropout=0.5, fc1=64):
    vgg_model = tf.keras.applications.VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet')

    VGG16_MODEL = vgg_model.layers[0](vgg_model)

    # unfreeze conv layers
    VGG16_MODEL.trainable = True

    dropout_layer = tf.keras.layers.Dropout(rate=dropout)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # add top layer for CIFAR100 classification
    dense = tf.keras.layers.Dense(fc1, activation='relu')
    prediction_layer = tf.keras.layers.Dense(len(CIFAR100_LABELS_LIST), activation='softmax')
    model = tf.keras.Sequential([VGG16_MODEL, dropout_layer, global_average_layer, dense, prediction_layer])

    return model


sweep_id = wandb.sweep(sweep_config, project="TL_CIFAR-20-VGG16")
wandb.agent(sweep_id, function=train)
