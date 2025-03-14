import numpy as np


def cross_entropy(y_pred: np.array, y_label: np.array) -> np.array:
    """
    Compute the cross entropy loss.

    Args:
        y_pred (np.array): The predicted probability
        y_label (np.array): Actual label

    Returns:
        np.array: mean of the cross entropy
    """
    epsilon = 1e-10
    dot_ = y_label * np.log(y_pred + epsilon)
    # just to convert the matrix to vector by removing the zeros
    dot_ = dot_.sum(axis=1)
    op = -np.mean(dot_)
    return op  # .sum()


def mse(y_pred: np.array, y_label: np.array) -> np.array:
    """
    Compute the Mean squared error

    Args:
        y_pred (np.array): The predicted probability
        y_label (np.array): Actual label

    Returns:
        np.array: Mean squared error loss
    """
    temp = (y_pred - y_label) ** 2
    temp = temp.sum(axis=1)
    mean_sq_error = np.mean(temp)  # Mean squared error over the batch
    return mean_sq_error
