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

# Import config
import sys
sys.path.append('..')
from config import NUM_CLASSES, IMAGE_SIZE, CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, HIDDEN_SIZE


class ArabicCNN(nn.Module):
    """
    Convolutional Neural Network for Arabic Letter Recognition.

    This class defines the structure of your neural network.

    You MUST implement these methods:
    - __init__: Define all layers
    - forward: Define how data flows through the layers

    -------------------------------------------------------------------------
    TODO: Complete this class!

    ARCHITECTURE SUGGESTION:

    Layer 1: Conv Block
        - Conv2d: 1 input channel, 32 output channels, 3x3 kernel, padding=1
        - ReLU activation
        - MaxPool2d: 2x2 kernel

    Layer 2: Conv Block
        - Conv2d: 32 input channels, 64 output channels, 3x3 kernel, padding=1
        - ReLU activation
        - MaxPool2d: 2x2 kernel

    Layer 3: Fully Connected
        - Flatten
        - Linear: (calculate input size) → 128
        - ReLU
        - Linear: 128 → NUM_CLASSES (28)
    -------------------------------------------------------------------------
    """

    def __init__(self):
        """
        Initialize the network layers.

        IMPORTANT: You must call super().__init__() first!

        TODO:
        1. Define conv layers using nn.Conv2d
        2. Define pooling layers using nn.MaxPool2d
        3. Define fully connected layers using nn.Linear

        HINTS:
        - nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        - nn.MaxPool2d(kernel_size)
        - nn.Linear(in_features, out_features)

        CALCULATING LINEAR INPUT SIZE:
        After 2 conv+pool layers with 2x2 pooling:
        32x32 → 16x16 → 8x8
        With 64 channels: 8 * 8 * 64 = 4096
        """
        super(ArabicCNN, self).__init__()  # Always call this first!

        # TODO: Define your layers here

        # Convolutional layers
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling layer (can reuse the same one)
        # self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # Calculate: after 2 pools of 2x2, image is 32/4 = 8x8
        # self.fc1 = nn.Linear(64 * 8 * 8, 128)
        # self.fc2 = nn.Linear(128, NUM_CLASSES)

        pass  # Remove this when you implement

    def forward(self, x):
        """
        Define the forward pass - how data flows through the network.

        This method is called automatically when you do: model(input)

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
                       Example: (32, 1, 32, 32) for batch of 32 grayscale 32x32 images

        Returns:
            Tensor: Output scores of shape (batch_size, NUM_CLASSES)
                   Example: (32, 28) - 28 scores for each of 32 images

        TODO:
        1. Pass x through conv1 → relu → pool
        2. Pass through conv2 → relu → pool
        3. Flatten the tensor
        4. Pass through fc1 → relu
        5. Pass through fc2
        6. Return the output

        HINTS:
        - F.relu(x) applies ReLU activation
        - x.view(x.size(0), -1) flattens while keeping batch dimension
        - Or use: x = torch.flatten(x, 1)
        """
        # TODO: Implement the forward pass

        # Example structure:
        # x = self.pool(F.relu(self.conv1(x)))  # Conv block 1
        # x = self.pool(F.relu(self.conv2(x)))  # Conv block 2
        # x = x.view(x.size(0), -1)             # Flatten
        # x = F.relu(self.fc1(x))               # FC block 1
        # x = self.fc2(x)                       # Output layer
        # return x

        pass  # Remove this when you implement


def count_parameters(model):
    """
    Count the total number of trainable parameters in the model.

    This helps you understand the complexity of your model.
    More parameters = more learning capacity, but also more prone to overfitting.

    Args:
        model: The PyTorch model

    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """
    Print a summary of the model architecture.

    Args:
        model: The PyTorch model
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 60)


# =============================================================================
# TESTING YOUR MODEL
# =============================================================================
if __name__ == "__main__":
    """
    Test your model implementation here.
    """
    print("Testing Model Implementation...")

    # Create the model
    # TODO: Uncomment when ready to test
    # model = ArabicCNN()
    # print_model_summary(model)

    # Test with dummy input
    # dummy_input = torch.randn(1, 1, 32, 32)  # (batch, channels, height, width)
    # output = model(dummy_input)
    # print(f"\nInput shape: {dummy_input.shape}")
    # print(f"Output shape: {output.shape}")
    # print(f"Expected output shape: (1, {NUM_CLASSES})")

    print("\nModel test complete!")
