import torch
from torch import nn


class Model(torch.nn.Module):
    """
    A convolutional neural network model for image classification.

    Args:
        num_channels (int): The number of input channels in the image.
        num_classes (int): The number of output classes for classification.

    Attributes:
        conv1 (torch.nn.Conv2d): The first convolutional layer.
        conv2 (torch.nn.Conv2d): The second convolutional layer.
        conv3 (torch.nn.Conv2d): The third convolutional layer.
        pool (torch.nn.MaxPool2d): The pooling layer.
        fc1 (torch.nn.Linear): The first fully connected layer.
        fc2 (torch.nn.Linear): The second fully connected layer.

    Methods:
        forward(x): Performs the forward pass of the model on a given input tensor.

    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initializes the layers of the model.

        Args:
            num_channels (int): The number of input channels in the image.
            num_classes (int): The number of output classes for classification.

        Returns:
            None

        """
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, kernel_size=4, padding=1)

        self.pool = nn.MaxPool2d(3, 3)
        self.batch_norm = nn.BatchNorm2d(3)

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(300, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model on a given input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).

        """
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.batch_norm(x)

        x = self.flatten(x)

        x = self.lin1(x)

        return x
