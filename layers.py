import numpy as np
from activation import sigmoid, relu, tanh, softmax
import copy

# from configuration import config


class HiddenLayer:
    """
    This is the hidden layer implementation of a fully connected network.
    """

    def __init__(
        self,
        num_of_nodes: int,
        num_of_nodes_prev_layer: int,
        config: dict,
        activation: str = "sigmoid",
    ):
        """

        Args:
            num_of_nodes (int): Number of nodes in this layer
            num_of_nodes_prev_layer (int): Number of nodes in the previous layer
            activation (str, optional): Layer activation function ["sigmoid", "tanh", "ReLU", "identity"]. Defaults to "sigmoid".
            config (dict): a dictionary of configuration.
        """
        self.num_of_nodes = num_of_nodes
        self.num_of_nodes_prev_layer = num_of_nodes_prev_layer
        self.activation = activation
        self.config = config
        if self.config["weight_initialisation"] == "random":
            self.weight = np.random.normal(
                0, 1, size=(self.num_of_nodes, self.num_of_nodes_prev_layer)
            )
            self.bias = np.random.normal(0, 1, size=(self.num_of_nodes, 1))
        elif self.config["weight_initialisation"] == "Xavier":
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

    def forward(self, input: np.array):
        """
        Perform the forward pass. activation(Wx+b)

        Args:
            input (np.array): The input to this layer
        """
        temp = np.matmul(self.weight, input.T).T
        self.a = temp + self.bias.T  ## Need to check the input shape?
        if self.activation == "sigmoid":
            self.h = sigmoid(self.a)
        if self.activation == "tanh":
            self.h = tanh(self.a)
        if self.activation == "ReLU":
            self.h = relu(self.a)
        if self.activation == "identity":
            self.h = copy.deepcopy(self.a)

        self.weight_l2 = np.multiply(self.weight, self.weight).reshape(-1)
        self.bias_l2 = np.multiply(self.bias, self.bias).reshape(-1)

    def backpropagation(
        self,
        next_layer_w: np.array,
        next_layer_L_theta_by_a: np.array,
        prev_layer_h: np.array,
    ):
        """
        Backpropogation of loss through the parameters to calculation
        the gradient w.r.t the loss.

        Args:
            next_layer_w (np.array): weights of the next layer
            next_layer_L_theta_by_a (np.array): Pre-activation of the next layer
            prev_layer_h (np.array): Post-activation of the previous layers
        """
        self.L_theta_by_h = np.matmul(next_layer_w.T, next_layer_L_theta_by_a.T).T
        ## get the g hat function for the sigmoid function
        if self.activation == "sigmoid":
            ## Here is some problem with getting the g_hat
            self.g_hat = np.multiply(self.h, (1 - self.h))
        elif self.activation == "tanh":
            self.g_hat = 1 - np.multiply(self.h, self.h)
        elif self.activation == "ReLU":
            self.g_hat = np.where(self.a > 0, 1, 0)
        elif self.activation == "identity":
            self.g_hat = np.ones_like(self.a)

        self.L_theta_by_a = np.multiply(self.L_theta_by_h, self.g_hat)
        self.L_theta_by_w = (
            np.matmul(self.L_theta_by_a.T, prev_layer_h)
            + self.config["L2_regularisation"] * self.weight
        )
        self.L_theta_by_b = (
            np.expand_dims(self.L_theta_by_a.sum(axis=0), axis=-1)
            # + self.config["L2_regularisation"] * self.bias ## removing L2 req using bias
        )


class OutputLayer:
    """
    This is the ouput layer implementation of a fully connected network.
    """

    def __init__(
        self,
        num_of_output_neuron: int,
        num_of_nodes_prev_layer: int,
        config: dict,
        activation: str = "softmax",
    ):
        """
        Ouput layer implementation

        Args:
            num_of_output_neuron (int): Number of the output nodes in this layer.
            num_of_nodes_prev_layer (int): Number of nodes in the previous layers
            activation (str, optional): Final activation layer. Defaults to "softmax".
            config (dict, optional): A dictionary of configuration.
        """
        self.num_of_output_neuron = num_of_output_neuron
        self.num_of_nodes_prev_layer = num_of_nodes_prev_layer
        self.activation = activation
        self.config = config
        if self.config["weight_initialisation"] == "random":
            self.weight = np.random.normal(
                0, 1, size=(self.num_of_output_neuron, self.num_of_nodes_prev_layer)
            )
            self.bias = np.random.normal(0, 1, size=(self.num_of_output_neuron, 1))
        elif self.config["weight_initialisation"] == "Xavier":
            xavier = np.sqrt(
                6 / (self.num_of_output_neuron + self.num_of_nodes_prev_layer)
            )
            self.weight = np.random.uniform(
                -xavier,
                xavier,
                size=(self.num_of_output_neuron, self.num_of_nodes_prev_layer),
            )
            self.bias = np.random.uniform(
                -xavier,
                xavier,
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
        """
        Perform the forward pass. activation(Wx+b)

        Args:
            input (np.array): The input to this layer
        """
        temp = np.matmul(self.weight, input.T).T
        self.a = temp + self.bias.T  ## Need to check the input shape?
        if self.activation == "softmax":
            self.h = softmax(self.a)

        self.weight_l2 = np.multiply(self.weight, self.weight).reshape(-1)
        self.bias_l2 = np.multiply(self.bias, self.bias).reshape(-1)

    def mse_softmax_gradient(self, y_label: np.array) -> np.array:
        """
        Compute the gradient of MSE loss with respect to logits when using softmax activation.
        Args:
            y_label (np.array): The label
        Returns:
            np.array: Gradient of MSE loss with respect to logits
        """

        diff = self.h - y_label

        n_samples, n_classes = self.h.shape
        gradient = np.zeros_like(self.h)

        # Vectorized implementation of the gradient formula
        for j in range(n_classes):
            # Create a mask for the Kronecker delta (1_{i=j})
            kronecker_delta = np.zeros(n_classes)
            kronecker_delta[j] = 1

            # For each sample, calculate gradient for class j
            for sample_idx in range(n_samples):
                sample_pred = self.h[sample_idx]
                sample_diff = diff[sample_idx]

                # Implement the formula: 2 * sum_i [(y_pred_i - y_true_i) * y_pred_i * (1_{i=j} - y_pred_j)]
                gradient[sample_idx, j] = 2 * np.sum(
                    sample_diff * sample_pred * (kronecker_delta - sample_pred[j])
                )

        return gradient

    def backpropagation(self, y_label: np.array, prev_layer_h: np.array):
        """
        Backpropogation of loss through the parameters to calculation
        the gradient w.r.t the loss.

        Args:
            y_label (np.array): Actual label
            prev_layer_h (np.array): Post-activation from previous layer
        """

        # self.L_theta_by_y_hat = np.dot(-1/self.h, y_label)
        if self.config["loss_fn"] == "cross_entropy":
            self.L_theta_by_a = -1 * (y_label - self.h)
        elif self.config["loss_fn"] == "mean_squared_error":
            self.L_theta_by_a = self.mse_softmax_gradient(y_label)
        self.L_theta_by_w = (
            np.matmul(self.L_theta_by_a.T, prev_layer_h)
            + self.config["L2_regularisation"] * self.weight
        )
        self.L_theta_by_b = (
            np.expand_dims(self.L_theta_by_a.sum(axis=0), axis=-1)
            # + self.config["L2_regularisation"] * self.bias ## removing L2 reg on bias
        )
