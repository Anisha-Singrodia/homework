import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A simple CNN with 2 convolutional layers and 2 fully-connected layers.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        # Input = 3 x 32 x 32, Output = 32 x 30 x 30
        self.conv_layer1 = nn.Conv2d(
            in_channels=num_channels, out_channels=16, kernel_size=3, stride=2
        )
        # Input = 32 x 30 x 30, Output = 32 x 28 x 28
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        # Input = 32 x 28 x 28, Output = 32 x 14 x 14 =
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Input = 32 x 14 x 14, Output = 64 x 12 x 12
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # Input = 64 x 12 x 12, Output = 64 x 10 x 10
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        # Input = 64 x 10 x 10, Output = 64 x 5 x 5 = 1600
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward method
        """
        out = self.conv_layer1(x)
        # out = self.relu1(out)
        # out = self.conv_layer2(out)
        out = self.max_pool1(out)
        # out = self.conv_layer3(out)
        # out = self.conv_layer4(out)
        # out = self.max_pool2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
