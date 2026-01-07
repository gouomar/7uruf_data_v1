"""
Utils - Helper functions for the project
========================================

This file contains utility functions that don't fit elsewhere.
Things like: setting random seeds, visualization helpers, etc.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# Import config
import sys
sys.path.append('..')
from config import SEED


def set_seed(seed=SEED):
    """
    Set random seed for reproducibility.

    This ensures you get the same results when running the code multiple times.
    Very important for debugging and comparing experiments!

    Args:
        seed: Random seed (default from config)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def show_sample_images(dataset, num_images=9, labels=None):
    """
    Display a grid of sample images from the dataset.

    Use this to visually verify your data loading is working correctly.

    Args:
        dataset: The PyTorch dataset
        num_images: Number of images to show (should be a perfect square)
        labels: List of label names
    """
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if i < num_images and i < len(dataset):
            image, label = dataset[i]

            # Handle different image formats
            if image.dim() == 3:
                image = image.squeeze(0)  # Remove channel dim for grayscale

            ax.imshow(image.numpy(), cmap='gray')

            if labels:
                ax.set_title(f'Label: {labels[label]}')
            else:
                ax.set_title(f'Label: {label}')

        ax.axis('off')

    plt.tight_layout()
    plt.show()


def check_gpu():
    """
    Check if GPU is available and print info.

    Run this to verify your CUDA setup is working.
    """
    print("=" * 50)
    print("GPU INFORMATION")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("✗ CUDA is NOT available. Training will use CPU (slower).")
        print("  If you're on Google Colab, go to Runtime > Change runtime type > GPU")

    print(f"  PyTorch Version: {torch.__version__}")
    print("=" * 50)


def count_samples_per_class(dataset):
    """
    Count how many samples are in each class.

    This helps identify imbalanced datasets (some classes have more samples).

    Args:
        dataset: The PyTorch dataset

    Returns:
        dict: {class_label: count}
    """
    class_counts = {}

    for _, label in dataset:
        label = label.item() if torch.is_tensor(label) else label
        class_counts[label] = class_counts.get(label, 0) + 1

    return dict(sorted(class_counts.items()))


if __name__ == "__main__":
    check_gpu()
    set_seed()
