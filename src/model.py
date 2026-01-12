Admiral
"""
================================================================================
PILLAR 2: THE ARCHITECTURE (model.py)
================================================================================

🎯 PURPOSE:
    Define the "brain" structure - the Convolutional Neural Network (CNN)

📚 CONCEPTS YOU NEED TO LEARN FIRST:
    1. What is a Neural Network? (Layers of connected nodes)
    2. What is a Convolution? (A filter that slides over the image)
    3. What is Pooling? (Reducing image size while keeping important info)
    4. What is an Activation Function? (Introduces non-linearity)
    5. What is a Fully Connected Layer? (Every neuron connected to every input)

🔑 KEY PYTORCH CLASSES:
    - nn.Module: Base class for all neural networks
    - nn.Conv2d: 2D Convolution layer
    - nn.MaxPool2d: Max pooling layer
    - nn.Linear: Fully connected layer
    - nn.ReLU: Activation function (Rectified Linear Unit)
    - nn.Flatten: Flatten multi-dim tensor to 1D

📖 RESOURCES TO STUDY (DO THIS FIRST!):
    - 3Blue1Brown "What is a Neural Network": https://www.youtube.com/watch?v=aircAruvnKk
    - 3Blue1Brown "What is Convolution": https://www.youtube.com/watch?v=KuXjwB4LzSA
    - PyTorch nn.Module: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

💡 THE BIG PICTURE - How a CNN processes an image:

    INPUT IMAGE (32x32x1)
          ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │  CONV LAYER 1: Detect simple features (edges, corners)              │
    │  [Conv2d] → [ReLU] → [MaxPool2d]                                    │
    │  32x32x1 → 32x32x32 → 32x32x32 → 16x16x32                           │
    └─────────────────────────────────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │  CONV LAYER 2: Detect complex features (curves, parts of letters)   │
    │  [Conv2d] → [ReLU] → [MaxPool2d]                                    │
    │  16x16x32 → 16x16x64 → 16x16x64 → 8x8x64                            │
    └─────────────────────────────────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │  FLATTEN: Convert 2D feature maps to 1D vector                      │
    │  8x8x64 → 4096                                                      │
    └─────────────────────────────────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │  FULLY CONNECTED: Learn combinations of features                    │
    │  [Linear] → [ReLU] → [Linear]                                       │
    │  4096 → 128 → 28 (number of classes)                                │
    └─────────────────────────────────────────────────────────────────────┘
          ↓
    OUTPUT: 28 numbers (one score for each Arabic letter)
            The highest score = the prediction!

================================================================================
"""

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
