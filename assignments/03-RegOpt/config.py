from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip


class CONFIG:
    batch_size = 1024
    num_epochs = 20
    initial_learning_rate = 0.01
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "gamma": 0.9,
        "step_size": 799,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            RandomHorizontalFlip(0.5),
            ToTensor(),
            Normalize(0.5,0.5)
        ]
    )
