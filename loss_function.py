import numpy as np


def cross_entropy(y_pred: np.array, y_label: np.array) -> np.array:
    epsilon = 1e-10
    dot_ = y_label * np.log(y_pred + epsilon)
    # dot_ = np.dot(y_label,np.log(y_pred))
    return -np.mean(dot_)  # .sum()


def mse(y_pred: np.array, y_label: np.array) -> np.array:
    # OP shape = [bs,]
    sq_error = np.mean((y_pred - y_label) ** 2)  # Mean squared error over the batch
    return sq_error
