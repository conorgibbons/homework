import torch


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

        self.conv1 = torch.nn.Conv2d(num_channels, 16, kernel_size=8, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, kernel_size=5, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 4, kernel_size=5, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(288, num_classes)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, num_classes)

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

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = torch.relu(x)



        return x
