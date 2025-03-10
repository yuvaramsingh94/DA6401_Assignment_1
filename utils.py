import numpy as np
from keras.datasets import fashion_mnist, mnist
import argparse


def accuracy(y_actual: list, y_pred: list) -> np.array:
    """_summary_

    Args:
        y_actual (list): List of one hot 
        y_pred (list): _description_

    Returns:
        np.array: _description_
    """
    y_actual = np.concat(y_actual, axis=0)
    y_pred = np.concat(y_pred, axis=0)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_actual, axis=1)

    return np.mean(y_true_classes == y_pred_classes)


###############################
def data_loader(dataset: str = "fashion_mnist", config: dict = {}):
    if dataset == "fashion_mnist":
        (x_Train, y_train_int), (x_test, y_test_int) = fashion_mnist.load_data()
    elif dataset == "mnist":
        (x_Train, y_train_int), (x_test, y_test_int) = mnist.load_data()
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


def parse_arguments(
    default_config: dict,
):
    parser = argparse.ArgumentParser(
        description="Train a Neural Network with command-line configuration."
    )

    # Define arguments with defaults
    parser.add_argument(
        "-wp",
        "--wandb_project",
        type=str,
        default=default_config["wandb_project"],
        help="Wandb project name",
    )
    parser.add_argument(
        "-we",
        "--wandb_entity",
        type=str,
        default=default_config["wandb_entity"],
        help="Wandb entity name",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=default_config["dataset"],
        choices=["mnist", "fashion_mnist"],
        help="Dataset to use",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=default_config["epochs"],
        help="Number of epochs",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=default_config["batch_size"],
        help="Batch size",
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        default=default_config["loss_fn"],
        choices=["mean_squared_error", "cross_entropy"],
        help="Loss function",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default=default_config["optimizer"],
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
        help="Optimizer",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=default_config["learning_rate"],
        help="Learning rate",
    )
    parser.add_argument(
        "-m",
        "--momentum",
        type=float,
        default=default_config["momentum_beta"],
        help="Momentum for momentum and NAG",
    )
    parser.add_argument(
        "-beta",
        "--beta",
        type=float,
        default=default_config["RMSprop_beta"],
        help="Beta for RMSprop",
    )
    parser.add_argument(
        "-beta1",
        "--beta1",
        type=float,
        default=default_config["adam_beta_1"],
        help="Beta1 for Adam and Nadam",
    )
    parser.add_argument(
        "-beta2",
        "--beta2",
        type=float,
        default=default_config["adam_beta_2"],
        help="Beta2 for Adam and Nadam",
    )
    parser.add_argument(
        "-eps",
        "--epsilon",
        type=float,
        default=default_config["RMS_epsilon"],
        help="Epsilon for optimizers/RMS epsilon",
    )
    parser.add_argument(
        "-w_d",
        "--weight_decay",
        type=float,
        default=default_config["L2_regularisation"],
        help="Weight decay L2 regularisation",
    )
    parser.add_argument(
        "-w_i",
        "--weight_init",
        type=str,
        default=default_config["weight_initialisation"],
        choices=["random", "Xavier"],
        help="Weight initialization method",
    )
    parser.add_argument(
        "-nhl",
        "--num_layers",
        type=int,
        default=default_config["num_hidden_layers"],
        help="Number of hidden layers",
    )
    parser.add_argument(
        "-sz",
        "--hidden_size",
        type=int,
        default=default_config["neurons_per_hidden_layer"][0],
        help="Hidden layer size (for single layer)",
    )
    parser.add_argument(
        "-a",
        "--activation",
        type=str,
        default=default_config["hidden_activation"],
        choices=["identity", "sigmoid", "tanh", "ReLU"],
        help="Activation function",
    )

    return parser.parse_args()


def update_configuration(args, default_config: dict) -> dict:
    """Updates the default configuration with command-line arguments."""
    config = default_config.copy()
    config["wandb_project"] = args.wandb_project
    config["wandb_entity"] = args.wandb_entity
    config["dataset"] = args.dataset
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["loss_fn"] = args.loss
    config["optimizer"] = args.optimizer
    config["learning_rate"] = args.learning_rate
    config["momentum_beta"] = args.momentum
    config["RMSprop_beta"] = args.beta
    config["adam_beta_1"] = args.beta1
    config["adam_beta_2"] = args.beta2
    config["RMS_epsilon"] = args.epsilon
    config["L2_regularisation"] = args.weight_decay
    config["weight_initialisation"] = args.weight_init
    config["num_hidden_layers"] = args.num_layers
    config["hidden_activation"] = args.activation
    # Handle hidden layer size, creating a list of sizes
    config["neurons_per_hidden_layer"] = [args.hidden_size] * config[
        "num_hidden_layers"
    ]
    return config
