import os
import wandb
from API_key import WANDB_API
import numpy as np
from keras.datasets import fashion_mnist
import tqdm

wandb.require("core")
wandb.login(key=WANDB_API)


## Setup the configuration
config = {
    "epochs": 10,
    "num_hidden_layers": 3,
    "neurons_per_hidden_layer": [32, 32, 32],
    "num_of_output_neuron": 10,
    "learning_rate": 0.000001,
    "batch_size": 4,
    "hidden_activation": "relu",
}

wandb.init(
    # Set the project where this run will be logged
    project="Fashion_MNIST",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_v1",
    # Track hyperparameters and run metadata
    config=config,
)

## TODO:  split training data for validation 10%
(x_train, y_train_int), (x_test, y_test_int) = fashion_mnist.load_data()
## Normalize the x
x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()
# print("Train", x_train.min(), x_train.max())
## One hot encode the Y
y_train = np.zeros((y_train_int.size, y_train_int.max() + 1))
y_train[np.arange(y_train_int.size), y_train_int] = 1

y_test = np.zeros((y_test_int.size, y_test_int.max() + 1))
y_test[np.arange(y_test_int.size), y_test_int] = 1


x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)
print("Dataset summary")
print("Train", x_train.shape, y_train.shape)
print("Test", x_test.shape, y_test.shape)


def sigmoid(x: np.array) -> np.array:
    x = np.clip(x, None, 709)  # Clip values at 709 to avoid overflow
    return 1 / (1 + np.exp(-x))


def softmax(x: np.array) -> np.array:
    x = np.clip(x, None, 709)  # Clip values at 709 to avoid overflow
    e_x = np.exp(x)
    return e_x / e_x.sum()


def tanh(x: np.array) -> np.array:
    x = np.clip(x, None, 709)  # Clip values at 709 to avoid overflow
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x: np.array) -> np.array:
    return np.maximum(0, x)


def cross_entropy(y_pred: np.array, y_label: np.array) -> np.array:
    epsilon = 1e-10
    dot_ = y_label * np.log(y_pred + epsilon)
    # dot_ = np.dot(y_label,np.log(y_pred))
    return -dot_  # .sum()


class HiddenLayer:
    def __init__(
        self,
        num_of_nodes: int,
        num_of_nodes_prev_layer: int,
        activation: str = "sigmoid",
    ):
        self.num_of_nodes = num_of_nodes
        self.num_of_nodes_prev_layer = num_of_nodes_prev_layer
        self.activation = activation
        self.weight = np.random.normal(
            0, 1, size=(self.num_of_nodes, self.num_of_nodes_prev_layer)
        )
        self.bias = np.random.normal(0, 1, size=(self.num_of_nodes, 1))

    def forward(self, input):
        temp = np.matmul(self.weight, input.T).T
        self.a = temp + self.bias.T  ## Need to check the input shape?
        if self.activation == "sigmoid":
            self.h = sigmoid(self.a)
        if self.activation == "tanh":
            self.h = tanh(self.a)
        if self.activation == "relu":
            self.h = relu(self.a)

    def backpropagation(
        self,
        next_layer_w: np.array,
        next_layer_L_theta_by_a: np.array,
        prev_layer_h: np.array,
    ):
        self.L_theta_by_h = np.matmul(
            next_layer_w.T, next_layer_L_theta_by_a.T
        ).T  # (4,100)
        ## get the g hat function for the sigmoid function
        if self.activation == "sigmoid":
            ## Here is some problem with getting the g_hat
            self.g_hat = np.multiply(self.h, (1 - self.h))
        elif self.activation == "tanh":
            self.g_hat = 1 - np.multiply(self.h, self.h)
        elif self.activation == "relu":
            self.g_hat = np.where(self.h > 0, 1, 0)

        ## calculate the L_theta_by_a

        self.L_theta_by_a = np.multiply(self.L_theta_by_h, self.g_hat)
        self.L_theta_by_w = np.matmul(self.L_theta_by_a.T, prev_layer_h)
        self.L_theta_by_b = np.expand_dims(self.L_theta_by_a.sum(axis=0), axis=-1)


class OutputLayer:
    def __init__(
        self,
        num_of_output_neuron: int,
        num_of_nodes_prev_layer: int,
        activation: str = "softmax",
    ):
        self.num_of_output_neuron = num_of_output_neuron
        self.num_of_nodes_prev_layer = num_of_nodes_prev_layer
        self.activation = activation
        self.weight = np.random.normal(
            0, 1, size=(self.num_of_output_neuron, self.num_of_nodes_prev_layer)
        )
        self.bias = np.random.normal(0, 1, size=(self.num_of_output_neuron, 1))

    def forward(self, input: np.array):
        temp = np.matmul(self.weight, input.T).T
        self.a = temp + self.bias.T  ## Need to check the input shape?
        if self.activation == "softmax":
            self.h = softmax(self.a)

    def backpropagation(self, y_label: np.array, prev_layer_h: np.array):
        # self.L_theta_by_y_hat = np.dot(-1/self.h, y_label)
        self.L_theta_by_a = -1 * (y_label - self.h)
        self.L_theta_by_w = np.matmul(self.L_theta_by_a.T, prev_layer_h)
        self.L_theta_by_b = np.expand_dims(self.L_theta_by_a.sum(axis=0), axis=-1)
        # print("Hi")


class NeuralNetwork:
    def __init__(
        self,
        input_neuron: int = 784,
        num_hidden_layers: int = 3,
        neurons_per_hidden_layer: list = [32, 32, 32],
        num_of_output_neuron: int = 10,
        learning_rate: float = 0.0001,
        hidden_activation: str = "sigmoid",
    ):
        self.input_neuron = input_neuron
        self.num_hidden_layers = num_hidden_layers
        assert self.num_hidden_layers == len(neurons_per_hidden_layer)
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.num_of_output_neuron = num_of_output_neuron
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
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

        for layer in self.nn_dict.values():
            ## forward pass
            layer["layer"].forward(x)
            layer["a"] = layer["layer"].a
            layer["h"] = layer["layer"].h
            x = layer["h"]

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

    def update(self):
        for count, (layer_name, layer) in enumerate(list(self.nn_dict.items())):
            layer["layer"].weight -= np.clip(
                layer["layer"].L_theta_by_w * self.learning_rate, a_min=-1e5, a_max=1e5
            )
            layer["layer"].bias -= np.clip(
                layer["layer"].L_theta_by_b * self.learning_rate, a_min=-1e5, a_max=1e5
            )
            ## do the update


"""
my_net = NeuralNetwork()

for i in range(4):
    op = my_net.forward_pass(np.expand_dims(x_train[0], axis = -1))
    # Calcualte the loss 
    print(f"The loss at try {i}", cross_entropy(y_pred = op, y_label = y_train[0]))


    my_net.backpropagation(x_train = np.expand_dims(x_train[0], axis = -1),
                        y_label = np.expand_dims(y_train[0], axis = -1))
    my_net.update()
"""
my_net = NeuralNetwork(
    num_hidden_layers=config["num_hidden_layers"],
    neurons_per_hidden_layer=config["neurons_per_hidden_layer"],
    num_of_output_neuron=config["num_of_output_neuron"],
    learning_rate=config["learning_rate"],
    hidden_activation=config["hidden_activation"],
)
BATCH_SIZE = config["batch_size"]
for epoch in range(config["epochs"]):
    training_loss_list = []
    validation_loss_list = []
    for i in tqdm.tqdm(range(x_train.shape[0] // 4)):
        train_x = x_train[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
        train_y = y_train[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
        op = my_net.forward_pass(train_x)
        # Calcualte the loss
        # print(f"The loss at try {i}", cross_entropy(y_pred = op, y_label = y_train[i*BATCH_SIZE: i*BATCH_SIZE + BATCH_SIZE]))
        training_loss_list.append(cross_entropy(y_pred=op, y_label=train_y))
        my_net.backpropagation(x_train=train_x, y_label=train_y)

        my_net.update()

    for i in tqdm.tqdm(range(x_test.shape[0] // 4)):
        test_x = x_test[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
        test_y = y_test[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
        op = my_net.forward_pass(test_x)
        # Calcualte the loss
        # print(f"The loss at try {i}", cross_entropy(y_pred = op, y_label = y_train[i*BATCH_SIZE: i*BATCH_SIZE + BATCH_SIZE]))
        validation_loss_list.append(cross_entropy(y_pred=op, y_label=test_y))
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
        }
    )
