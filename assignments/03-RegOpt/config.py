from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor

# Resize, RandomRotation, RandomHorizontalFlip, ColorJitter, RandomAdjustSharpness, RandomErasing


class CONFIG:
    batch_size = 64
    num_epochs = 5
    initial_learning_rate = 0.001
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        # "T_max": (batch_size + num_epochs),
        # "T_0": 8,
        # "T_mult": 1,
        # "eta_min": 1e-4,
        # "last_epoch": -1,
        # "verbose": False,
        "base_lr": 0.001,
        "max_lr": 0.01,
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
            Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # Resize((32, 32)), # Resize the image in a 32X32 shape
            # RandomRotation(10), # Randomly rotate some images by 20 degrees
            # RandomHorizontalFlip(0.1), # Randomly horizontal flip the images
            # ColorJitter(brightness = 0.1, contrast = 0.1,
            #                     saturation = 0.1), #reduces
            # RandomAdjustSharpness(sharpness_factor = 2, p = 0.1),
            # RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
        ]
    )
