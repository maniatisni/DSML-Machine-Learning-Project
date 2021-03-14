import tensorflow as tf
from tensorflow.keras import layers
import helpers as hp
import models as mdl
import wandb
from wandb.keras import WandbCallback
import os

os.environ["WANDB_MODE"] = "dryrun"
PROJECT_NAME = "TL_CIFAR-8011-VGG16"

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
        },
        'size': {
            'values': [32, 64, 75]
        }
    }
}

x_train, y_train, x_val, y_val, x_test, y_test, data_size = hp.cifar_data_(num_classes=20)

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomHeight(3 / 32),
])


def prepare(ds, shuffle=False, augment=False, batch_size=32):

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


def train():
    # Initialize wandb with a sample project name
    wandb.init(config=sweep_config)

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(wandb.config.size, wandb.config.size),
        layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    train_ds = prepare(
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(lambda x, y: (resize_and_rescale(x), y),
                                                                   num_parallel_calls=AUTOTUNE), shuffle=True,
        augment=False, batch_size=wandb.config.batch_size)
    test_ds = prepare(tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(lambda x, y: (resize_and_rescale(x), y),
                                                                               num_parallel_calls=AUTOTUNE),
                      batch_size=wandb.config.batch_size)
    val_ds = prepare(tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(lambda x, y: (resize_and_rescale(x), y),
                                                                            num_parallel_calls=AUTOTUNE),
                     batch_size=wandb.config.batch_size)

    # Iniialize model with hyperparameters
    tf.keras.backend.clear_session()
    model = mdl.VGG16(wandb.config.dropout, wandb.config.fc1, wandb.config.size)
    # Compile the model
    opt = tf.keras.optimizers.Adam(
        learning_rate=wandb.config.learning_rate)  # optimizer with different learning rate specified by config
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    _ = model.fit(train_ds,
                  epochs=wandb.config.epochs,
                  steps_per_epoch=50, validation_steps=30,
                  validation_data=val_ds,
                  callbacks=[WandbCallback()])  # WandbCallback to automatically track metrics

    # Evaluate
    loss, accuracy = model.evaluate(test_ds, steps=10)
    print('Test Error Rate: ', round((1 - accuracy) * 100, 2))
    wandb.log({'Test Error Rate': round((1 - accuracy) * 100, 2)})  # wandb.log to tr


sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
wandb.agent(sweep_id, function=train)
