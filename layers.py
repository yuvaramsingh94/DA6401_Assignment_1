import numpy as np
from activation import sigmoid, relu, tanh, softmax
from configuration import config


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
        if config["weight_initialisation"] == "random":
            self.weight = np.random.normal(
                0, 1, size=(self.num_of_nodes, self.num_of_nodes_prev_layer)
            )
            self.bias = np.random.normal(0, 1, size=(self.num_of_nodes, 1))
        elif config["weight_initialisation"] == "xavier":
            xavier = np.sqrt(6 / (self.num_of_nodes + self.num_of_nodes_prev_layer))
            self.weight = np.random.uniform(
                -xavier,
                xavier,
                size=(self.num_of_nodes, self.num_of_nodes_prev_layer),
            )
            self.bias = np.random.uniform(
                -xavier,
                xavier,
                size=(self.num_of_nodes, 1),
            )
        ## Set this as zero for now
        self.u_w = np.zeros_like(self.weight)
        self.u_b = np.zeros_like(self.bias)

        ## Set this as zero for now
        self.v_w = np.zeros_like(self.weight)
        self.v_b = np.zeros_like(self.bias)

        ## Set this as zero for now
        self.m_w = np.zeros_like(self.weight)
        self.m_b = np.zeros_like(self.bias)

    def forward(self, input):
        temp = np.matmul(self.weight, input.T).T
        self.a = temp + self.bias.T  ## Need to check the input shape?
        if self.activation == "sigmoid":
            self.h = sigmoid(self.a)
        if self.activation == "tanh":
            self.h = tanh(self.a)
        if self.activation == "relu":
            self.h = relu(self.a)

        self.weight_l2 = np.multiply(self.weight, self.weight).reshape(-1)
        self.bias_l2 = np.multiply(self.bias, self.bias).reshape(-1)

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
        self.L_theta_by_w = (
            np.matmul(self.L_theta_by_a.T, prev_layer_h)
            + config["L2_regularisation"] * self.weight
        )
        self.L_theta_by_b = (
            np.expand_dims(self.L_theta_by_a.sum(axis=0), axis=-1)
            + config["L2_regularisation"] * self.bias
        )


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
        if config["weight_initialisation"] == "random":
            self.weight = np.random.normal(
                0, 1, size=(self.num_of_output_neuron, self.num_of_nodes_prev_layer)
            )
            self.bias = np.random.normal(0, 1, size=(self.num_of_output_neuron, 1))
        elif config["weight_initialisation"] == "xavier":
            self.weight = np.random.uniform(
                -(
                    np.sqrt(
                        6 / (self.num_of_output_neuron + self.num_of_nodes_prev_layer)
                    )
                ),
                (
                    np.sqrt(
                        6 / (self.num_of_output_neuron + self.num_of_nodes_prev_layer)
                    )
                ),
                size=(self.num_of_output_neuron, self.num_of_nodes_prev_layer),
            )
            self.bias = np.random.uniform(
                -(
                    np.sqrt(
                        6 / (self.num_of_output_neuron + self.num_of_nodes_prev_layer)
                    )
                ),
                (
                    np.sqrt(
                        6 / (self.num_of_output_neuron + self.num_of_nodes_prev_layer)
                    )
                ),
                size=(self.num_of_output_neuron, 1),
            )

        ## Set this as zero for now
        self.u_w = np.zeros_like(self.weight)
        self.u_b = np.zeros_like(self.bias)

        ## Set this as zero for now
        self.v_w = np.zeros_like(self.weight)
        self.v_b = np.zeros_like(self.bias)

        ## Set this as zero for now
        self.m_w = np.zeros_like(self.weight)
        self.m_b = np.zeros_like(self.bias)

    def forward(self, input: np.array):
        temp = np.matmul(self.weight, input.T).T
        self.a = temp + self.bias.T  ## Need to check the input shape?
        if self.activation == "softmax":
            self.h = softmax(self.a)

        self.weight_l2 = np.multiply(self.weight, self.weight).reshape(-1)
        self.bias_l2 = np.multiply(self.bias, self.bias).reshape(-1)

    def backpropagation(self, y_label: np.array, prev_layer_h: np.array):
        # self.L_theta_by_y_hat = np.dot(-1/self.h, y_label)
        self.L_theta_by_a = -1 * (y_label - self.h)
        self.L_theta_by_w = (
            np.matmul(self.L_theta_by_a.T, prev_layer_h)
            + config["L2_regularisation"] * self.weight
        )
        self.L_theta_by_b = (
            np.expand_dims(self.L_theta_by_a.sum(axis=0), axis=-1)
            + config["L2_regularisation"] * self.bias
        )
        # print("Hi")
