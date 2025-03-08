import os
import wandb
from API_key import WANDB_API
import numpy as np
from keras.datasets import fashion_mnist
from layers import HiddenLayer, OutputLayer
from optimizers import SGD, momentum, Adam, Nadam, RMSprop
from loss_function import cross_entropy, mse
import tqdm
import copy
from configuration import config

wandb.require("core")
wandb.login(key=WANDB_API)

## TODO: Check the MSE implementation


wandb.init(
    # Set the project where this run will be logged
    project="Fashion_MNIST",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_v1",
    # Track hyperparameters and run metadata
    config=config,
)

## TODO:  split training data for validation 10%
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
print("Dataset summary")
print("Train", x_train.shape, y_train.shape)
print("Val", x_val.shape, y_val.shape)
print("Test", x_test.shape, y_test.shape)


def accuracy(y_actual: list, y_pred: list):
    y_actual = np.concat(y_actual, axis=0)
    y_pred = np.concat(y_pred, axis=0)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_actual, axis=1)

    return np.mean(y_true_classes == y_pred_classes)


class NeuralNetwork:
    def __init__(
        self,
        input_neuron: int = 784,
        num_hidden_layers: int = 3,
        neurons_per_hidden_layer: list = [32, 32, 32],
        num_of_output_neuron: int = 10,
        learning_rate: float = 0.0001,
        hidden_activation: str = "sigmoid",
        optimizer: str = "SGD",
    ):
        self.input_neuron = input_neuron
        self.num_hidden_layers = num_hidden_layers
        assert self.num_hidden_layers == len(neurons_per_hidden_layer)
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.num_of_output_neuron = num_of_output_neuron
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.optimizer = optimizer
        ## Build the NN
        self.build_nn()

    # wx+b
    def build_nn(self):
        self.nn_dict = {}
        for layer_i, neurons_i in enumerate(self.neurons_per_hidden_layer):
            if layer_i == 0:  # first layer
                self.nn_dict[f"Layer_{layer_i+1}"] = {
                    "layer": HiddenLayer(
                        num_of_nodes=neurons_i,
                        num_of_nodes_prev_layer=self.input_neuron,
                        activation=self.hidden_activation,
                    )
                }
            else:
                self.nn_dict[f"Layer_{layer_i+1}"] = {
                    "layer": HiddenLayer(
                        num_of_nodes=neurons_i,
                        num_of_nodes_prev_layer=self.neurons_per_hidden_layer[
                            layer_i - 1
                        ],
                        activation=self.hidden_activation,
                    )
                }
        ## Add the output layer
        self.nn_dict["Output_layer"] = {
            "layer": OutputLayer(
                self.num_of_output_neuron,
                self.neurons_per_hidden_layer[-1],
                activation="softmax",
            )
        }

    def forward_pass(self, x):
        self.weight_l2 = []
        self.bias_l2 = []
        for layer in self.nn_dict.values():
            ## forward pass
            layer["layer"].forward(x)
            layer["a"] = layer["layer"].a
            layer["h"] = layer["layer"].h
            x = layer["h"]
            self.weight_l2.append(layer["layer"].weight_l2)
            self.bias_l2.append(layer["layer"].bias_l2)

        return x

    def backpropagation(self, x_train: np.array, y_label: np.array):
        reverse_layers = list(self.nn_dict.items())[::-1]
        for count, (layer_name, layer) in enumerate(reverse_layers):  # reverse the

            if layer_name == "Output_layer":
                layer["layer"].backpropagation(
                    y_label=y_label,
                    prev_layer_h=reverse_layers[count + 1][-1]["layer"].h,
                )
            elif layer_name == "Layer_1":
                layer["layer"].backpropagation(
                    next_layer_w=reverse_layers[count - 1][-1]["layer"].weight,
                    next_layer_L_theta_by_a=reverse_layers[count - 1][-1][
                        "layer"
                    ].L_theta_by_a,
                    prev_layer_h=x_train,
                )
            else:
                layer["layer"].backpropagation(
                    next_layer_w=reverse_layers[count - 1][-1]["layer"].weight,
                    next_layer_L_theta_by_a=reverse_layers[count - 1][-1][
                        "layer"
                    ].L_theta_by_a,
                    prev_layer_h=reverse_layers[count + 1][-1]["layer"].h,
                )

    def update(self, epoch: int):

        for count, (layer_name, layer) in enumerate(list(self.nn_dict.items())):

            if self.optimizer == "SGD":
                SGD(layer, self.learning_rate)

            elif self.optimizer == "momentum":
                momentum(layer, self.learning_rate)
            elif self.optimizer == "RMSprop":
                RMSprop(layer, self.learning_rate)

            elif self.optimizer == "Adam":
                Adam(layer, self.learning_rate, epoch)
            elif self.optimizer == "Nadam":
                Nadam(layer, self.learning_rate, epoch)

    def NAG_look_weight_update(self):
        for count, (layer_name, layer) in enumerate(list(self.nn_dict.items())):
            layer["layer"].weight -= np.clip(
                config["momentum_beta"] * layer["layer"].u_w, a_min=-1e5, a_max=1e5
            )
            layer["layer"].bias -= np.clip(
                config["momentum_beta"] * layer["layer"].u_b, a_min=-1e5, a_max=1e5
            )

    def NAG_leep_weight_update(self, temp_net: "NeuralNetwork"):
        for count, ((layer_name, layer_actual), (_, layer_copy)) in enumerate(
            zip(list(self.nn_dict.items()), list(temp_net.nn_dict.items()))
        ):
            layer_actual["layer"].u_w = (
                config["momentum_beta"] * layer_actual["layer"].u_w
                + layer_copy["layer"].L_theta_by_w
            )
            layer_actual["layer"].u_b = (
                config["momentum_beta"] * layer_actual["layer"].u_b
                + layer_copy["layer"].L_theta_by_b
            )

            updated_weight = np.clip(
                layer_actual["layer"].u_w * self.learning_rate, a_min=-0.1, a_max=0.1
            )
            updated_bias = np.clip(
                layer_actual["layer"].u_b * self.learning_rate, a_min=-0.1, a_max=0.1
            )

            layer_actual["layer"].weight -= updated_weight
            layer_actual["layer"].bias -= updated_bias


my_net = NeuralNetwork(
    num_hidden_layers=config["num_hidden_layers"],
    neurons_per_hidden_layer=config["neurons_per_hidden_layer"],
    num_of_output_neuron=config["num_of_output_neuron"],
    learning_rate=config["learning_rate"],
    hidden_activation=config["hidden_activation"],
    optimizer=config["optimizer"],
)
BATCH_SIZE = config["batch_size"]
for epoch in range(1, config["epochs"] + 1):
    training_loss_list = []
    validation_loss_list = []
    y_pred_list = []
    y_list = []
    for i in tqdm.tqdm(range(x_train.shape[0] // 4)):
        train_x = x_train[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
        train_y = y_train[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
        y_list.append(train_y)
        op = my_net.forward_pass(train_x)
        y_pred_list.append(op)
        # Calcualte the loss
        # print(f"The loss at try {i}", cross_entropy(y_pred = op, y_label = y_train[i*BATCH_SIZE: i*BATCH_SIZE + BATCH_SIZE]))

        l2_reg = np.concat(my_net.weight_l2).sum() + np.concat(my_net.bias_l2).sum()

        if config["loss_fn"] == ["cross entropy"]:
            main_loss = cross_entropy(y_pred=op, y_label=train_y)
        elif config["loss_fn"] == "mse":
            main_loss = mse(y_pred=op, y_label=train_y)
        training_loss_list.append(
            main_loss + (config["L2_regularisation"] / 2) * l2_reg
        )
        my_net.backpropagation(x_train=train_x, y_label=train_y)
        if my_net.optimizer != "NAG":
            my_net.update(epoch=epoch)
        elif my_net.optimizer == "NAG":
            temp_net = copy.deepcopy(my_net)
            temp_net.NAG_look_weight_update()
            _ = temp_net.forward_pass(train_x)
            temp_net.backpropagation(x_train=train_x, y_label=train_y)
            my_net.NAG_leep_weight_update(temp_net)

            del temp_net
            ## Do a Forward pass and backpropagation to get the gradients
    train_accuracy = accuracy(y_list, y_pred_list)

    y_pred_list = []
    y_list = []
    for i in tqdm.tqdm(range(x_test.shape[0] // 4)):
        test_x = x_test[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
        test_y = y_test[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
        y_list.append(test_y)
        op = my_net.forward_pass(test_x)
        y_pred_list.append(op)
        # Calcualte the loss
        # print(f"The loss at try {i}", cross_entropy(y_pred = op, y_label = y_train[i*BATCH_SIZE: i*BATCH_SIZE + BATCH_SIZE]))
        temp_1 = np.concat(my_net.weight_l2)
        temp_2 = np.concat(my_net.bias_l2)
        l2_reg = temp_1.sum() + temp_2.sum()

        if config["loss_fn"] == "cross entropy":
            main_loss = cross_entropy(y_pred=op, y_label=test_y)
        elif config["loss_fn"] == "mse":
            main_loss = mse(y_pred=op, y_label=test_y)
        validation_loss_list.append(
            main_loss + (config["L2_regularisation"] / 2) * l2_reg
        )
    val_accuracy = accuracy(y_list, y_pred_list)
    ##

    # print(
    #     f"Training loss at epoch {epoch}",
    #     np.array(training_loss_list).reshape(-1).mean(),
    # )
    # print(
    #     f"Validation loss at epoch {epoch}",
    #     np.array(validation_loss_list).reshape(-1).mean(),
    # )

    wandb.log(
        {
            "Training loss": np.array(training_loss_list).reshape(-1).mean(),
            "Validation loss": np.array(validation_loss_list).reshape(-1).mean(),
            "Train accucary": train_accuracy,
            "Validation accucary": val_accuracy,
        }
    )
