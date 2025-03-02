import numpy as np
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)
print("Dataset summary")
print("Train",x_train.shape,y_train.shape)
print("Test",x_test.shape,y_test.shape)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    e_x = np.exp(x)
    return e_x/e_x.sum()

class HiddenLayer:
    def __init__(self, num_of_nodes: int, num_of_nodes_prev_layer: int, activation: str = "sigmoid"):
        self.num_of_nodes = num_of_nodes
        self.num_of_nodes_prev_layer = num_of_nodes_prev_layer
        self.activation = activation
        self.weight = np.random.normal(0, 1, size=(self.num_of_nodes, self.num_of_nodes_prev_layer))
        self.bias = np.random.normal(0, 1, size=(self.num_of_nodes,1))

    def forward(self, input):
        mul_o = np.matmul(self.weight, input)
        self.a = mul_o + self.bias ## Need to check the input shape?
        if self.activation == "sigmoid":
            self.h = sigmoid(self.a)


class OutputLayer:
    def __init__(self, num_of_output_neuron: int, num_of_nodes_prev_layer: int, activation: str = "softmax"):
        self.num_of_output_neuron = num_of_output_neuron
        self.num_of_nodes_prev_layer = num_of_nodes_prev_layer
        self.activation = activation
        self.weight = np.random.normal(0, 1, size=(self.num_of_output_neuron, self.num_of_nodes_prev_layer))
        self.bias = np.random.normal(0, 1, size=(self.num_of_output_neuron))

    def forward(self, input):
        self.a = np.matmul(self.weight, input) + self.bias ## Need to check the input shape?
        if self.activation == "softmax":
            self.h = softmax(self.a)



class NeuralNetwork:
    def __init__(self, input_neuron: int= 784, num_hidden_layers: int = 3, neurons_per_hidden_layer: list = [100,100,100], num_of_output_neuron: int = 10):
        self.input_neuron = input_neuron
        self.num_hidden_layers = num_hidden_layers
        assert self.num_hidden_layers == len(neurons_per_hidden_layer)
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.num_of_output_neuron = num_of_output_neuron
        ## Build the NN
        self.build_nn()

    # wx+b
    def build_nn(self):
        self.nn_dict = {}
        for layer_i, neurons_i in enumerate(self.neurons_per_hidden_layer):
            if layer_i == 0: # first layer 
                self.nn_dict[f"Layer_{layer_i+1}"] = {"layer":HiddenLayer(num_of_nodes = neurons_i, num_of_nodes_prev_layer = self.input_neuron, activation = "sigmoid")}
            else:
                self.nn_dict[f"Layer_{layer_i+1}"] = {"layer":HiddenLayer(num_of_nodes = neurons_i, num_of_nodes_prev_layer = self.neurons_per_hidden_layer[layer_i - 1], activation = "sigmoid")}
        ## Add the output layer
        self.nn_dict["Output_layer"] = {"layer":OutputLayer(self.num_of_output_neuron, self.neurons_per_hidden_layer[-1], activation = "softmax")}

    def forward_pass(self, x):
        
        for layer in self.nn_dict.values():
            ## forward pass
            layer["layer"].forward(x)
            layer["a"] = layer["layer"].a
            layer["h"] = layer["layer"].h
            x = layer["h"]

        return x


my_net = NeuralNetwork()
my_net.forward_pass(np.expand_dims(x_train[0], axis = -1))