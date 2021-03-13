import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers


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
    return filtered_list


# select a url for a unique subset of CIFAR-100 with 20, 40, 60, or 80 classes
def select_classes_number(classes_number=20):
    if classes_number == 20:
        return "https://pastebin.com/raw/nzE1n98V"
    elif classes_number == 40:
        return "https://pastebin.com/raw/zGX4mCNP"
    elif classes_number == 60:
        return "https://pastebin.com/raw/nsDTd3Qn"
    elif classes_number == 80:
        return "https://pastebin.com/raw/SNbXz700"
    else:
        return -1


# get class label from class index
def class_label_from_index(fine_category):
    return CIFAR100_LABELS_LIST[fine_category.item(0)]


# load the entire dataset
(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

# REPLACE WITH YOUR C NUMBER
team_sead = 119

# select the number of classes
cifar100_classes_url = select_classes_number(40)

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

data_size, img_rows, img_cols, img_channels = x_train_ds.shape

# set validation set percentage (wrt the training set size)
validation_percentage = 0.15
val_size = round(validation_percentage * data_size)

# Reserve val_size samples for validation and normalize all values
x_val = x_train_ds[-val_size:]
y_val = y_train_ds[-val_size:]
x_train = x_train_ds[:-val_size]
y_train = y_train_ds[:-val_size]
x_test = x_test_ds
y_test = y_test_ds

IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
    layers.experimental.preprocessing.Rescaling(1. / 255)
])

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE


def prepare(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(data_size)

    ds.repeat()

    # Batch all datasets
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)


train_ds = prepare(tf.data.Dataset.from_tensor_slices((x_train, y_train)), shuffle=True, augment=True)
test_ds = prepare(tf.data.Dataset.from_tensor_slices((x_test, y_test)))
val_ds = prepare(tf.data.Dataset.from_tensor_slices((x_val, y_val)))


def VGG16(dropout=0.5, fc1=64):
    vgg_model = tf.keras.applications.VGG16(input_shape=(180, 180, 3), include_top=False, weights='imagenet')

    VGG16_MODEL = vgg_model.layers[0](vgg_model)

    # unfreeze conv layers
    VGG16_MODEL.trainable = True

    dropout_layer = tf.keras.layers.Dropout(rate=dropout)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # add top layer for CIFAR100 classification
    dense = tf.keras.layers.Dense(fc1, activation='relu')
    prediction_layer = tf.keras.layers.Dense(len(CIFAR100_LABELS_LIST), activation='softmax')
    model = tf.keras.Sequential([VGG16_MODEL, dropout_layer, global_average_layer, dense, prediction_layer])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


model = VGG16()
epochs = 10
# Train the model
_ = model.fit(train_ds,
              epochs=epochs,
              steps_per_epoch=40, validation_steps=20,
              validation_data=val_ds)  # WandbCallback to automatically track metrics

# Evaluate
loss, accuracy = model.evaluate(test_ds, steps=10)
