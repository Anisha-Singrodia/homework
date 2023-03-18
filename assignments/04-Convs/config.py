from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize


class CONFIG:
    batch_size = 64
    num_epochs = 3

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=0.005)

    transforms = Compose(
        [
            ToTensor(),
            # Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # Normalize(mean=[0.485, 0.456, 0.4], std=[0.229, 0.224, 0.2])
        ]
    )
