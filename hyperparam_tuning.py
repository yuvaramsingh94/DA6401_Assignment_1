import os
import wandb
from API_key import WANDB_API
import numpy as np


from loss_function import cross_entropy, mse
import tqdm
import copy
from configuration import config
from NeuralNetwork import NeuralNetwork
from utils import accuracy, data_loader

wandb.require("core")
wandb.login(key=WANDB_API)

## TODO: Check the MSE implementation


x_train, y_train, x_val, y_val, x_test, y_test = data_loader(
    dataset="Fashion_MNIST", config=config
)
print("Dataset summary")
print("Train", x_train.shape, y_train.shape)
print("Val", x_val.shape, y_val.shape)
print("Test", x_test.shape, y_test.shape)

# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_accuracy"},
    "parameters": {
        "learning_rate": {"max": 0.0001, "min": 0.00001},
        "optimizer": {"values": ["SGD", "momentum", "RMSprop", "Adam", "Nadam"]},
    },
}


def main():

    wandb.init(
        # Set the project where this run will be logged
        project="Fashion-MNIST-sweep",
        # Track hyperparameters and run metadata
        # config=config,
    )
    ## Update the config dict with the hpt from sweep
    config["learning_rate"] = wandb.config.learning_rate
    config["optimizer"] = wandb.config.optimizer

    my_net = NeuralNetwork(
        num_hidden_layers=config["num_hidden_layers"],
        neurons_per_hidden_layer=config["neurons_per_hidden_layer"],
        num_of_output_neuron=config["num_of_output_neuron"],
        learning_rate=config["learning_rate"],
        hidden_activation=config["hidden_activation"],
        optimizer=config["optimizer"],
        config=config,
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
        for i in tqdm.tqdm(range(x_val.shape[0] // 4)):
            val_x = x_val[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
            val_y = y_val[i * BATCH_SIZE : i * BATCH_SIZE + BATCH_SIZE]
            y_list.append(val_y)
            op = my_net.forward_pass(val_x)
            y_pred_list.append(op)
            # Calcualte the loss
            # print(f"The loss at try {i}", cross_entropy(y_pred = op, y_label = y_train[i*BATCH_SIZE: i*BATCH_SIZE + BATCH_SIZE]))
            temp_1 = np.concat(my_net.weight_l2)
            temp_2 = np.concat(my_net.bias_l2)
            l2_reg = temp_1.sum() + temp_2.sum()

            if config["loss_fn"] == "cross entropy":
                main_loss = cross_entropy(y_pred=op, y_label=val_y)
            elif config["loss_fn"] == "mse":
                main_loss = mse(y_pred=op, y_label=val_y)
            validation_loss_list.append(
                main_loss + (config["L2_regularisation"] / 2) * l2_reg
            )
        val_accuracy = accuracy(y_list, y_pred_list)
        wandb.log({"val_accuracy": val_accuracy})


## initialize the HPT
sweep_id = wandb.sweep(sweep=sweep_configuration, project="Fashion-MNIST-sweep")

wandb.agent(sweep_id, function=main, count=3)
