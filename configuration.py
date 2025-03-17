## Setup the configuration
"""
About:
This file holds the configuarations that will be used to 
train the network. This will have the best hyperparameter 
to start with.
"""
config = {
    "wandb_project": "Fashion_MNIST",
    "wandb_entity": "test2",
    "dataset": "fashion_mnist",
    "epochs": 10,
    "num_hidden_layers": 3,  # Number of hidden layer
    "neurons_per_hidden_layer": [128],
    "num_of_output_neuron": 10,
    "learning_rate": 0.006857163485944494,
    "batch_size": 16,
    "hidden_activation": "ReLU",
    "optimizer": "adam",  # momentum
    "momentum_beta": 0.5,
    "RMS_epsilon": 1e-5,
    "RMSprop_beta": 0.5,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "validation_split": 0.1,
    "weight_initialisation": "Xavier",
    "L2_regularisation": 0.0005,
    "loss_fn": "cross_entropy",  # mse
    ### To be set
}
