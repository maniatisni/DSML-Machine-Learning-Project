import tensorflow as tf
import helpers as hp


def xception_(dropout=0.5, fc1=32, size=224):
    base_model = tf.keras.applications.Xception(input_shape=(size, size, 3), include_top=False,
                                                weights='imagenet')

    base_model.trainable = False

    dropout_layer = tf.keras.layers.Dropout(rate=dropout)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # add top layer for CIFAR100 classification
    dense = tf.keras.layers.Dense(fc1, activation='relu')
    prediction_layer = tf.keras.layers.Dense(len(hp.CIFAR100_LABELS_LIST), activation='softmax')
    model = tf.keras.Sequential([base_model, dropout_layer, global_average_layer, dense, prediction_layer])

    return model


def VGG16(dropout=0.5, fc1=64, size=32):
    vgg_model = tf.keras.applications.VGG16(input_shape=(size, size, 3), include_top=False, weights='imagenet')

    VGG16_MODEL = vgg_model.layers[0](vgg_model)

    # unfreeze conv layers
    VGG16_MODEL.trainable = True

    dropout_layer = tf.keras.layers.Dropout(rate=dropout)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # add top layer for CIFAR100 classification
    dense = tf.keras.layers.Dense(fc1, activation='relu')
    prediction_layer = tf.keras.layers.Dense(len(hp.CIFAR100_LABELS_LIST), activation='softmax')
    model = tf.keras.Sequential([VGG16_MODEL, dropout_layer, global_average_layer, dense, prediction_layer])

    return model


def Densenet201(fc1=256, dropout=0.4, size=224):
    base_model = tf.keras.applications.DenseNet201(include_top=False,
                                                   weights='imagenet',
                                                   input_shape=(size, size, 3),
                                                   pooling='max')
    base_model.trainable = False
    dropout_layer = tf.keras.layers.Dropout(rate=dropout)
    dense = tf.keras.layers.Dense(fc1, activation="relu")
    prediction_layer = tf.keras.layers.Dense(len(hp.CIFAR100_LABELS_LIST), activation='softmax')
    model = tf.keras.Sequential([base_model, dropout_layer, dense, prediction_layer])
    return model


def Mobilenet_(fc1=128, dropout=0.3, img_size=160):
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        weights='imagenet', input_shape=(img_size, img_size, 3),
        include_top=False, pooling='avg'
    )

    base_model.trainable = False
    dropout_layer = tf.keras.layers.Dropout(rate=dropout)
    dense = tf.keras.layers.Dense(fc1, activation="relu")
    prediction_layer = tf.keras.layers.Dense(len(hp.CIFAR100_LABELS_LIST), activation='softmax')
    model = tf.keras.Sequential([base_model, dropout_layer, dense, prediction_layer])

    return model


def VGG3(filter=32, filter2=64, filter3=128, fc1=128, pool_size=2):
    model = tf.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filter, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                                     input_shape=(32, 32, 3)))
    model.add(
        tf.keras.layers.Conv2D(filter, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((pool_size, pool_size)))
    model.add(
        tf.keras.layers.Conv2D(filter2, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(
        tf.keras.layers.Conv2D(filter2, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((pool_size, pool_size)))
    model.add(
        tf.keras.layers.Conv2D(filter3, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(
        tf.keras.layers.Conv2D(filter3, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((pool_size, pool_size)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(fc1, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='softmax'))

    return model
