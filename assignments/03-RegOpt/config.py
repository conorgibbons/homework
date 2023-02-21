from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip


class CONFIG:
    batch_size = 32
    num_epochs = 8
    initial_learning_rate = 0.002
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "gamma": 0.99,
        "step_size": 420,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
        amsgrad=True,
    )

    transforms = Compose(
        [
            ToTensor(),
            RandomHorizontalFlip(),
            Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ]
    )
