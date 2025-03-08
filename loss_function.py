import numpy as np


def cross_entropy(y_pred: np.array, y_label: np.array) -> np.array:
    epsilon = 1e-10
    dot_ = y_label * np.log(y_pred + epsilon)
    # dot_ = np.dot(y_label,np.log(y_pred))
    return -dot_  # .sum()


def mse(y_pred: np.array, y_label: np.array) -> np.array:
    # OP shape = [bs,]
    sq_error = np.multiply((y_pred - y_label), (y_pred - y_label)).sum(axis=1)

    # dot_ = np.dot(y_label,np.log(y_pred))
    return -sq_error  # .sum()
