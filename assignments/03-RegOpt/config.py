from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 64
    num_epochs = 200
    initial_learning_rate = 0.005
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "start_factor": 0.33,
        "end_factor": 1,
        "total_iters": 10,
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
            ToTensor(),
        ]
    )
