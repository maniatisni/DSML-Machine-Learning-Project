import tensorflow as tf
import numpy as np
import pandas as pd
import wandb
from tensorflow.keras import layers
import models as mdl
import helpers as hp

# select from from_list elements with index in index_list
from wandb.integration.keras import WandbCallback

PROJECT_NAME = 'xception80_data_aug'

if __name__ == '__main__':
    # Initialize wandb with your project name
    run = wandb.init(project=PROJECT_NAME,
                     config={  # and include hyperparameters and metadata
                         "learning_rate": 0.00008,
                         "epochs": 40,
                         "batch_size": 32,
                         "dropout": 0.6,
                         "fc1": 256,
                         "size": 224,
                         "loss_function": "sparse_categorical_crossentropy",
                         "architecture": "Xception+Dense",
                         "aug": True,
                         "dataset": "CIFAR-100"
                     })
    config = wandb.config  # We'll use this to configure our experiment

    x_train, y_train, x_val, y_val, x_test, y_test, data_size = hp.cifar_data_(num_classes=20)


    def prepare(ds, shuffle=False, augment=False, batch_size=32):
        # Resize and rescale all datasets
        ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                    num_parallel_calls=AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(data_size)

        ds = ds.repeat()

        # Batch all datasets
        ds = ds.batch(batch_size)

        # Use data augmentation only on the training set
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)

        # Use buffered prefecting on all datasets
        return ds.prefetch(buffer_size=AUTOTUNE)


    IMG_SIZE = wandb.config.size

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
        layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        # layers.experimental.preprocessing.RandomContrast(0.4)
        # layers.experimental.preprocessing.RandomRotation(0.2),
        # layers.experimental.preprocessing.RandomCrop(height=wandb.config.size, width=wandb.config.size),
        # layers.experimental.preprocessing.RandomCrop(0.2),
        # layers.experimental.preprocessing.CentralCrop(),
        layers.experimental.preprocessing.RandomHeight(3 / 32),
        layers.experimental.preprocessing.RandomWidth(3 / 32)
    ])

    batch_size = wandb.config.batch_size
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = prepare(tf.data.Dataset.from_tensor_slices((x_train, y_train)), shuffle=True,
                       augment=wandb.config.aug, batch_size=batch_size)
    test_ds = prepare(tf.data.Dataset.from_tensor_slices((x_test, y_test)), batch_size=batch_size)
    val_ds = prepare(tf.data.Dataset.from_tensor_slices((x_val, y_val)), batch_size=batch_size)

    # Initialize model like you usually do.
    tf.keras.backend.clear_session()

    # Check models.py for available models
    model = mdl.xception_(dropout=wandb.config.dropout, fc1=wandb.config.fc1, size=wandb.config.size)

    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    model.compile(optimizer, config.loss_function, metrics=['acc'])

    epochs = wandb.config.epochs
    # Train the model
    _ = model.fit(train_ds,
                  epochs=epochs,
                  steps_per_epoch=40, validation_steps=20,
                  validation_data=val_ds,
                  callbacks=[WandbCallback()])  # WandbCallback to automatically track metrics

    # Evaluate
    loss, accuracy = model.evaluate(test_ds, steps=10)
    print('Test Error Rate: ', round((1 - accuracy) * 100, 2))
    wandb.log({'Test Error Rate': round((1 - accuracy) * 100, 2)})
