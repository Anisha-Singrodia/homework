from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 32
    num_epochs = 5
    initial_learning_rate = 0.001
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        # "T_max": batch_size + num_epochs,
        "T_0": 200,
        "T_mult": 1,
        "eta_min": 0,
        "last_epoch": -1,
        "verbose": False,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [ToTensor(), Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))]
    )
