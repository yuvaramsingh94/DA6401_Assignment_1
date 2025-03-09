import numpy as np


def sigmoid(x: np.array) -> np.array:
    x = np.clip(x, None, 709)  # Clip values at 709 to avoid overflow
    return 1 / (1 + np.exp(-x))


def softmax(x: np.array) -> np.array:
    # For numerical stability, ReLU layer is causing issue
    x = x - np.max(x, axis=1, keepdims=True)
    x = np.clip(x, None, 709)  # Clip values at 709 to avoid overflow
    e_x = np.exp(x)
    return e_x / e_x.sum()


def tanh(x: np.array) -> np.array:
    x = np.clip(x, None, 709)  # Clip values at 709 to avoid overflow
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x: np.array) -> np.array:
    return np.maximum(0, x)
