# DA6401 Assignment 1

Author : V M Vijaya Yuvaram Singh (DA24S015)

## Errors to handle 
RuntimeWarning: overflow encountered in reduce

## TODO
Wandb key as environment variable


## a README file with clear instructions on training and evaluating the model (the 10 marks will be based on this)

## Folder organization

DA6401_Assignment_1/
├── .gitignore
├── activation.py
├── API_key.py
├── configuration.py
├── hyperparam_tuning.py
├── layers.py
├── loss_function.py
├── NeuralNetwork.py
├── optimizers.py
├── README.md
├── requirments.txt
├── train.py
├── utils.py

## File explaination
- `activation.py`: Implementation of the activation functions. `sigmoid`,`tanh`,`ReLU`, and `identity`.
- `API_key.py`: This file has the Wandb key. But, this file is not uploaded to github. added to .gitignore.
- `configuration.py`: This has the configuration for building the model. This is the best performing configuration.
- `hyperparam_tuning.py`: This is the `wandb sweep` for performing hyperparameter tuning.
- `layers.py`: Hidden and output layer implementation with forward and gradient calculations.
- `loss_function.py`: Implemented `mean squared error` and `cross entropy`.
- `NeuralNetwork.py`: Class to build multi layer neural network from the configuration provided.
- `optimizers.py`: Implemented `sgd`,`momuntum`,`nag`,`rmsprop`,`adam`, and `nadam`.
- `train.py`: The main training python file. It can take command line arguments and also from `configuration.py` file.
- `utils.py`: Implemented helper functions such as `accuracy`, `argparser`, `dataloader`
- `requirments.txt`: The python pkg. mainly `numpy` and `keras`, `tensorflow` for dataset.

## How to run
### Running hyperparameter tuning
```python hyperparam_tuning.py```
### Running training code
```python train.py```
<br>
you can pass the following commandline arguments.

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |
