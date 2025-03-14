import numpy as np


def sigmoid(x: np.array) -> np.array:
    """
    Apply sigmoid activation function. This can handle
    input in batches also.

    Args:
        x (np.array): Input pre-activation array

    Returns:
        np.array: post-activation
    """
    x = np.clip(x, None, 709)  # Clip values at 709 to avoid overflow
    return 1 / (1 + np.exp(-x))


def softmax(x: np.array) -> np.array:
    """
    Apply softmax activation.

    Args:
        x (np.array): Input pre-activation array

    Returns:
        np.array: post-activation
    """

    x = x - np.max(x, axis=1, keepdims=True)
    x = np.clip(x, None, 709)  # Clip values at 709 to avoid overflow
    e_x = np.exp(x)
    op = e_x / e_x.sum(axis=1)[:, np.newaxis]
    return op


def tanh(x: np.array) -> np.array:
    """
    Apply tanh activation function

    Args:
        x (np.array): Input pre-activation array

    Returns:
        np.array: post-activation
    """
    x = np.clip(x, None, 709)  # Clip values at 709 to avoid overflow
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x: np.array) -> np.array:
    """
    Apply ReLU activation function

    Args:
        x (np.array): Input pre-activation array

    Returns:
        np.array: post-activation
    """
    return np.maximum(0, x)
