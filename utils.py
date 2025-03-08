import numpy as np
from keras.datasets import fashion_mnist


def accuracy(y_actual: list, y_pred: list):
    y_actual = np.concat(y_actual, axis=0)
    y_pred = np.concat(y_pred, axis=0)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_actual, axis=1)

    return np.mean(y_true_classes == y_pred_classes)


###############################
def data_loader(dataset: str = "Fashion_MNIST", config: dict = {}):
    if dataset == "Fashion_MNIST":
        (x_Train, y_train_int), (x_test, y_test_int) = fashion_mnist.load_data()
    ## Normalize the x
    x_Train = (x_Train - x_Train.mean()) / x_Train.std()
    x_test = (x_test - x_test.mean()) / x_test.std()
    # print("Train", x_Train.min(), x_Train.max())
    ## One hot encode the Y
    y_Train = np.zeros((y_train_int.size, y_train_int.max() + 1))
    y_Train[np.arange(y_train_int.size), y_train_int] = 1

    y_test = np.zeros((y_test_int.size, y_test_int.max() + 1))
    y_test[np.arange(y_test_int.size), y_test_int] = 1

    num_validation_samples = int(len(x_Train) * config["validation_split"])
    indices = np.arange(len(x_Train))
    np.random.shuffle(indices)

    # Split the data
    val_indices = indices[:num_validation_samples]
    train_indices = indices[num_validation_samples:]

    x_val = x_Train[val_indices]
    y_val = y_Train[val_indices]

    x_train = x_Train[train_indices]
    y_train = y_Train[train_indices]

    ## Split the x_train into validation

    x_train = x_train.reshape(len(x_train), -1)
    x_val = x_val.reshape(len(x_val), -1)
    x_test = x_test.reshape(len(x_test), -1)

    return x_train, y_train, x_val, y_val, x_test, y_test
