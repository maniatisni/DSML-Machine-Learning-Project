import pandas as pd
import tensorflow as tf
import numpy as np

CIFAR100_LABELS_LIST = \
    pd.read_csv('https://pastebin.com/raw/qgDaNggt', sep=',', header=None).astype(str).values.tolist()[0]


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


def cifar_data_(num_classes=20, validation_pct=0.15, seed=119):
    # load the entire dataset
    (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.cifar100.load_data(label_mode='fine')


    # select the number of classes
    cifar100_classes_url = select_classes_number(num_classes)

    my_classes = pd.read_csv(cifar100_classes_url, sep=',', header=None)

    our_index = my_classes.iloc[seed, :].values.tolist()
    our_classes = select_from_list(CIFAR100_LABELS_LIST, our_index)
    train_index = get_ds_index(y_train_all, our_index)
    test_index = get_ds_index(y_test_all, our_index)

    x_train_ds = np.asarray(select_from_list(x_train_all, train_index))
    y_train_ds = np.asarray(select_from_list(y_train_all, train_index))
    x_test_ds = np.asarray(select_from_list(x_test_all, test_index))
    y_test_ds = np.asarray(select_from_list(y_test_all, test_index))

    # get (train) dataset dimensions
    data_size, img_rows, img_cols, img_channels = x_train_ds.shape

    # set validation set percentage (wrt the training set size)
    validation_percentage = validation_pct
    val_size = round(validation_percentage * data_size)

    # Reserve val_size samples for validation and normalize all values
    x_val = x_train_ds[-val_size:]
    y_val = y_train_ds[-val_size:]
    x_train = x_train_ds[:-val_size]
    y_train = y_train_ds[:-val_size]
    x_test = x_test_ds
    y_test = y_test_ds

    return x_train, y_train, x_val, y_val, x_test, y_test, data_size
