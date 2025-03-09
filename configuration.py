## Setup the configuration
config = {
    "wandb_project": "Fashion_MNIST",
    "wandb_entity": "experiment_v1",
    "dataset": "fashion_mnist",
    "epochs": 3,
    "num_hidden_layers": 3,  # Number of hidden layer
    "neurons_per_hidden_layer": [20],
    "num_of_output_neuron": 10,
    "learning_rate": 0.000001,
    "batch_size": 4,
    "hidden_activation": "sigmoid",
    "optimizer": "nag",  # momentum
    "momentum_beta": 0.5,
    "RMS_epsilon": 1e-5,
    "RMSprop_beta": 0.5,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "validation_split": 0.1,
    "weight_initialisation": "random",
    "L2_regularisation": 0.0001,
    "loss_fn": "mean_squared_error",  # mse
    ### To be set
}
