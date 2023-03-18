import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class Model(nn.Module):
    """
    A simple CNN with 2 convolutional layers and 2 fully-connected layers.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:

        super().__init__()
        self.model = torch.nn.Sequential(
            # Input = 3 x 32 x 32, Output = 32 x 32 x 32
            torch.nn.Conv2d(
                in_channels=num_channels, out_channels=32, kernel_size=3, padding=1
            ),
            torch.nn.ReLU(),
            # Input = 32 x 32 x 32, Output = 32 x 16 x 16
            torch.nn.MaxPool2d(kernel_size=2),
            # Input = 32 x 16 x 16, Output = 64 x 16 x 16
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Input = 64 x 16 x 16, Output = 64 x 8 x 8
            torch.nn.MaxPool2d(kernel_size=2),
            # #Input = 64 x 8 x 8, Output = 64 x 8 x 8
            # torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            # torch.nn.ReLU(),
            # #Input = 64 x 8 x 8, Output = 64 x 4 x 4
            # torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward method
        """
        return self.model(x)
