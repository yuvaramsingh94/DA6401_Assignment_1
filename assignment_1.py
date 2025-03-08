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


wandb.init(
    # Set the project where this run will be logged
    project="Fashion_MNIST",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_v1",
    # Track hyperparameters and run metadata
    config=config,
)
x_train, y_train, x_val, y_val, x_test, y_test = data_loader(
    dataset="Fashion_MNIST", config=config
)
print("Dataset summary")
print("Train", x_train.shape, y_train.shape)
print("Val", x_val.shape, y_val.shape)
print("Test", x_test.shape, y_test.shape)


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
            "Train accucary": train_accuracy,
            "Validation accucary": val_accuracy,
        }
    )
