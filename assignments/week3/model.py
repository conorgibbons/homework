import torch
from typing import Callable
from torch import nn


class MLP(torch.nn.Module):
    """
    A simple multi-layer perceptron.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """

        super(MLP, self).__init__()

        self.actv = activation()

        # Initialize layers of MLP
        self.layers = nn.ModuleList()

        # Loop over layers and create each one
        for i in range(hidden_count):
            next_num_inputs = hidden_size

            self.layers += [nn.Linear(input_size, next_num_inputs)]

            input_size = next_num_inputs

        # Create final layer
        self.out = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = x.view(x.shape[0], -1)

        # Get activations of each layer
        for layer in self.layers:
            x = self.actv(layer(x))

        # Get outputs
        x = self.out(x)

        return x
