## Setup the configuration
config = {
    "epochs": 10,
    "num_hidden_layers": 3,
    "neurons_per_hidden_layer": [32, 32, 32],
    "num_of_output_neuron": 10,
    "learning_rate": 0.0001,
    "batch_size": 4,
    "hidden_activation": "sigmoid",
    "optimizer": "Nadam",  # momentum
    "momentum_beta": 0.5,
    "RMS_epsilon": 1e-5,
    "RMSprop_beta": 0.5,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "validation_split": 0.1,
    "weight_initialisation": "random",
    "L2_regularisation": 0.0001,
    "loss_fn": "mse",  # mse
}
