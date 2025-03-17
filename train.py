import os
import argparse
import wandb
import numpy as np
from loss_function import cross_entropy, mse
import tqdm
import copy
from configuration import config
from NeuralNetwork import NeuralNetwork
from utils import accuracy, data_loader, parse_arguments, update_configuration
import matplotlib.pyplot as plt


## TODO: Check the NAG implementation with the learning rate update
## from the pseudocode

wandb.require("core")
if "WANDB_API_KEY" in dict(os.environ).keys():
    wandb.init()
else:
    print(
        "WANDB_API_KEY environment variable is not set. Please set it or make a python file called "
        "'API_key.py' add a single line WANDB_API = '<Your KEY>'"
    )
    from API_key import WANDB_API

    wandb.login(key=WANDB_API)

## Get the commandline argument
args = parse_arguments(default_config=config)
config = update_configuration(args, default_config=config)

if "WANDB_API_KEY" in dict(os.environ).keys():
    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_entity"],
        config=config,
    )
else:
    print(
        "WANDB_API_KEY environment variable is not set. Please set it or make a python file called "
        "'API_key.py' add a single line WANDB_API = '<Your KEY>'"
    )
    from API_key import WANDB_API

    wandb.login(key=WANDB_API)
    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_entity"],
        config=config,
    )


x_train, y_train, x_val, y_val, x_test, y_test = data_loader(
    dataset=config["dataset"], config=config
)
print("Dataset summary")
print("Train", x_train.shape, y_train.shape)
print("Val", x_val.shape, y_val.shape)
print("Test", x_test.shape, y_test.shape)


## Display the image

fig, axes = plt.subplots(2, 5, figsize=(50, 45))
axes = axes.ravel()  # Flatten the axes array for easy indexing
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
# Iterate through the classes and plot one image per class
for i in range(10):
    # Find the first image index for the current class
    idx = np.where(np.argmax(y_train, axis=1) == i)[0][0]

    # Plot the image on the corresponding subplot
    axes[i].imshow(
        x_train[idx].reshape(28, 28), cmap="gray"
    )  # Use 'gray' colormap for Fashion MNIST
    axes[i].set_title(class_names[i], fontsize=106)
    axes[i].axis("off")  # Hide axes ticks and labels

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()

# Log the plot to wandb
wandb.log({"Fashion MNIST Classes": fig})


def main():
    my_net = NeuralNetwork(
        config=config,
    )
    BATCH_SIZE = config["batch_size"]
    for epoch in range(1, config["epochs"] + 1):
        training_loss_list = []
        validation_loss_list = []
        y_pred_list = []
        y_list = []
        for i in tqdm.tqdm(range(x_train.shape[0] // BATCH_SIZE)):
            train_x = x_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            train_y = y_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            y_list.append(train_y)
            op = my_net.forward_pass(train_x)
            y_pred_list.append(op)
            # Calcualte the loss
            # print(f"The loss at try {i}", cross_entropy(y_pred = op, y_label = y_train[i*BATCH_SIZE: i*BATCH_SIZE + BATCH_SIZE]))

            l2_reg = np.sum(np.concatenate(my_net.weight_l2))  # + np.sum(
            #    np.concatenate(my_net.bias_l2)
            # )

            if config["loss_fn"] == "cross_entropy":
                main_loss = cross_entropy(y_pred=op, y_label=train_y)
            elif config["loss_fn"] == "mean_squared_error":
                main_loss = mse(y_pred=op, y_label=train_y)
            training_loss_list.append(
                main_loss + (config["L2_regularisation"] / 2) * l2_reg
            )
            my_net.backpropagation(x_train=train_x, y_label=train_y)
            if my_net.optimizer != "nag":
                my_net.update(epoch=epoch)
            elif my_net.optimizer == "nag":
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
        for i in tqdm.tqdm(range(x_val.shape[0] // BATCH_SIZE)):
            val_x = x_val[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            val_y = y_val[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            y_list.append(val_y)
            op = my_net.forward_pass(val_x)
            y_pred_list.append(op)
            # Calcualte the loss
            # print(f"The loss at try {i}", cross_entropy(y_pred = op, y_label = y_train[i*BATCH_SIZE: i*BATCH_SIZE + BATCH_SIZE]))
            l2_reg = np.sum(np.concatenate(my_net.weight_l2))  # + np.sum(
            #    np.concatenate(my_net.bias_l2)
            # )

            if config["loss_fn"] == "cross_entropy":
                main_loss = cross_entropy(y_pred=op, y_label=val_y)
            elif config["loss_fn"] == "mean_squared_error":
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
                "Training loss": np.mean(training_loss_list),
                "Validation loss": np.mean(validation_loss_list),
                "Train accucary": train_accuracy,
                "Validation accucary": val_accuracy,
            }
        )
    ## Predict on the test dataset
    y_pred_list_test = []
    y_list_test = []
    for i in tqdm.tqdm(range(x_test.shape[0] // BATCH_SIZE)):
        test_x = x_test[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        test_y = y_test[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        y_list_test.append(test_y)
        op = my_net.forward_pass(test_x)
        y_pred_list_test.append(op)
        # Calcualte the loss
        # print(f"The loss at try {i}", cross_entropy(y_pred = op, y_label = y_train[i*BATCH_SIZE: i*BATCH_SIZE + BATCH_SIZE]))
        # l2_reg = np.sum(np.concatenate(my_net.weight_l2))  # + np.sum(
        #    np.concatenate(my_net.bias_l2)
        # )

        if config["loss_fn"] == "cross_entropy":
            main_loss = cross_entropy(y_pred=op, y_label=test_y)
        elif config["loss_fn"] == "mean_squared_error":
            main_loss = mse(y_pred=op, y_label=test_y)

    test_accuracy = accuracy(y_list_test, y_pred_list_test)

    ## Convert the test pred to plot
    y_actual = np.concat(y_list_test, axis=0)
    y_pred = np.concat(y_pred_list_test, axis=0)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_actual, axis=1)
    wandb.log(
        {
            "Test_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                preds=y_pred_classes,
                y_true=y_true_classes,
                class_names=class_names,
            )
        }
    )
    wandb.log({"Test accuracy": test_accuracy})
    wandb.finish()


if __name__ == "__main__":
    main()
