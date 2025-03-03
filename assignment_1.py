import numpy as np
from keras.datasets import fashion_mnist

(x_train, y_train_int), (x_test, y_test_int) = fashion_mnist.load_data()

## One hot encode the Y
y_train = np.zeros((y_train_int.size, y_train_int.max() + 1))
y_train[np.arange(y_train_int.size), y_train_int] = 1

y_test = np.zeros((y_test_int.size, y_test_int.max() + 1))
y_test[np.arange(y_test_int.size), y_test_int] = 1


x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)
print("Dataset summary")
print("Train",x_train.shape,y_train.shape)
print("Test",x_test.shape,y_test.shape)


def sigmoid(x: np.array) -> np.array:
    return 1/(1+np.exp(-x))

def softmax(x: np.array) -> np.array:
    e_x = np.exp(x)
    return e_x/e_x.sum()

def cross_entropy(y_pred: np.array, y_label: np.array) -> np.array:
    dot_ = np.dot(y_label,np.log(y_pred))
    return -dot_.sum()


class HiddenLayer:
    def __init__(self, num_of_nodes: int, num_of_nodes_prev_layer: int, activation: str = "sigmoid"):
        self.num_of_nodes = num_of_nodes
        self.num_of_nodes_prev_layer = num_of_nodes_prev_layer
        self.activation = activation
        self.weight = np.random.normal(0, 1, size=(self.num_of_nodes, self.num_of_nodes_prev_layer))
        self.bias = np.random.normal(0, 1, size=(self.num_of_nodes,1))

    def forward(self, input):
        self.a = np.matmul(self.weight, input) + self.bias ## Need to check the input shape?
        if self.activation == "sigmoid":
            self.h = sigmoid(self.a)

    def backpropagation(self, next_layer_w: np.array, next_layer_L_theta_by_a: np.array, prev_layer_h: np.array):
        self.L_theta_by_h = np.matmul(next_layer_w.T, next_layer_L_theta_by_a)
        ## get the g hat function for the sigmoid function
        if self.activation == "sigmoid":
            ## Here is some problem with getting the g_hat
            self.g_hat = np.dot(self.h.squeeze(), (1- self.h.squeeze()))
        ## calculate the L_theta_by_a
        
        self.L_theta_by_a = np.dot(self.L_theta_by_h, self.g_hat)## dummy placeholder
        self.L_theta_by_w = np.matmul(self.L_theta_by_a, prev_layer_h.T)
        self.L_theta_by_b = self.L_theta_by_a 
class OutputLayer:
    def __init__(self, num_of_output_neuron: int, num_of_nodes_prev_layer: int, activation: str = "softmax"):
        self.num_of_output_neuron = num_of_output_neuron
        self.num_of_nodes_prev_layer = num_of_nodes_prev_layer
        self.activation = activation
        self.weight = np.random.normal(0, 1, size=(self.num_of_output_neuron, self.num_of_nodes_prev_layer))
        self.bias = np.random.normal(0, 1, size=(self.num_of_output_neuron,1))

    def forward(self, input: np.array):
        self.a = np.matmul(self.weight, input) + self.bias ## Need to check the input shape?
        if self.activation == "softmax":
            self.h = softmax(self.a)

    def backpropagation(self, y_label: np.array, prev_layer_h: np.array):
        #self.L_theta_by_y_hat = np.dot(-1/self.h, y_label)
        self.L_theta_by_a = -1 * (y_label - self.h)
        self.L_theta_by_w = np.matmul(self.L_theta_by_a, prev_layer_h.T)
        self.L_theta_by_b = self.L_theta_by_a 

class NeuralNetwork:
    def __init__(self, input_neuron: int= 784, num_hidden_layers: int = 3, neurons_per_hidden_layer: list = [100,100,100], num_of_output_neuron: int = 10, learning_rate = 0.0001):
        self.input_neuron = input_neuron
        self.num_hidden_layers = num_hidden_layers
        assert self.num_hidden_layers == len(neurons_per_hidden_layer)
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.num_of_output_neuron = num_of_output_neuron
        self.learning_rate = learning_rate
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
    
    def backpropagation(self, x_train:np.array, y_label: np.array):
        reverse_layers = list(self.nn_dict.items())[::-1]
        for count, (layer_name, layer) in enumerate(reverse_layers):   # reverse the  
            
            if layer_name == "Output_layer":
                layer["layer"].backpropagation(y_label = y_label, prev_layer_h = reverse_layers[count + 1][-1]["layer"].h)
            elif layer_name == "Layer_1":
                layer["layer"].backpropagation(next_layer_w = reverse_layers[count - 1][-1]["layer"].weight, 
                                               next_layer_L_theta_by_a = reverse_layers[count - 1][-1]["layer"].L_theta_by_a, 
                                               prev_layer_h = x_train,)
            else:
                layer["layer"].backpropagation(next_layer_w = reverse_layers[count - 1][-1]["layer"].weight, 
                                               next_layer_L_theta_by_a = reverse_layers[count - 1][-1]["layer"].L_theta_by_a, 
                                               prev_layer_h = reverse_layers[count + 1][-1]["layer"].h,)
                
    def update(self):
        for count, (layer_name, layer) in enumerate(list(self.nn_dict.items())):   
            layer["layer"].weight -= layer["layer"].L_theta_by_w * self.learning_rate
            layer["layer"].bias -= layer["layer"].L_theta_by_b * self.learning_rate
            ## do the update

my_net = NeuralNetwork()

for i in range(4):
    op = my_net.forward_pass(np.expand_dims(x_train[0], axis = -1))
    # Calcualte the loss 
    print(f"The loss at try {i}", cross_entropy(y_pred = op, y_label = y_train[0]))


    my_net.backpropagation(x_train = np.expand_dims(x_train[0], axis = -1),
                        y_label = np.expand_dims(y_train[0], axis = -1))
    my_net.update()