import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import NUM_CLASSES, IMAGE_SIZE, CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, HIDDEN_SIZE


class ArabicCNN(nn.Module):
    """
    Convolutional Neural Network for Arabic Letter Recognition.
    """

    def __init__(self):
        super(ArabicCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=CONV1_OUT_CHANNELS,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=CONV1_OUT_CHANNELS,
            out_channels=CONV2_OUT_CHANNELS,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        feature_map_size = IMAGE_SIZE // 4
        fc_input_size = CONV2_OUT_CHANNELS * feature_map_size * feature_map_size
        self.fc1 = nn.Linear(fc_input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = ArabicCNN()
    # fake data for test only Nabil ola Omar fach ghatraini model hahowa example dyal data:
    dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
