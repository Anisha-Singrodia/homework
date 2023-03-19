from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    batch_size = 32
    num_epochs = 1

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=0.001)

    transforms = Compose(
        [
            ToTensor(),
            # Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # RandomRotation(1),
            # ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1),
            # Normalize(mean=[0.485, 0.456, 0.4], std=[0.229, 0.224, 0.2])
        ]
    )
