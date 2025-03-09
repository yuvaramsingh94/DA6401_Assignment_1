import numpy as np
from layers import OutputLayer, HiddenLayer
from configuration import config
from optimizers import SGD, momentum, Adam, Nadam, RMSprop


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
        config: dict = {},
    ):
        self.input_neuron = input_neuron
        self.num_hidden_layers = num_hidden_layers
        assert self.num_hidden_layers == len(neurons_per_hidden_layer)
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.num_of_output_neuron = num_of_output_neuron
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.optimizer = optimizer
        self.config = config
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
                        config=self.config,
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
                        config=self.config,
                    )
                }
        ## Add the output layer
        self.nn_dict["Output_layer"] = {
            "layer": OutputLayer(
                self.num_of_output_neuron,
                self.neurons_per_hidden_layer[-1],
                activation="softmax",
                config=self.config,
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

        for count, (_, layer) in enumerate(list(self.nn_dict.items())):

            if self.optimizer == "sgd":
                SGD(layer, self.learning_rate, self.config)

            elif self.optimizer == "momentum":
                momentum(layer, self.learning_rate, self.config)
            elif self.optimizer == "rmsprop":
                RMSprop(layer, self.learning_rate, self.config)

            elif self.optimizer == "adam":
                Adam(layer, self.learning_rate, epoch, self.config)
            elif self.optimizer == "nadam":
                Nadam(layer, self.learning_rate, epoch, self.config)

    def NAG_look_weight_update(self):
        for count, (_, layer) in enumerate(list(self.nn_dict.items())):
            layer["layer"].weight -= np.clip(
                self.config["momentum_beta"] * layer["layer"].u_w, a_min=-1e5, a_max=1e5
            )
            layer["layer"].bias -= np.clip(
                self.config["momentum_beta"] * layer["layer"].u_b, a_min=-1e5, a_max=1e5
            )

    def NAG_leep_weight_update(self, temp_net: "NeuralNetwork"):
        for count, ((_, layer_actual), (_, layer_copy)) in enumerate(
            zip(list(self.nn_dict.items()), list(temp_net.nn_dict.items()))
        ):
            layer_actual["layer"].u_w = (
                self.config["momentum_beta"] * layer_actual["layer"].u_w
                + layer_copy["layer"].L_theta_by_w
            )
            layer_actual["layer"].u_b = (
                self.config["momentum_beta"] * layer_actual["layer"].u_b
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
