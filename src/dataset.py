"""
================================================================================
PILLAR 1: DATA INGESTION (dataset.py)
================================================================================

🎯 PURPOSE:
    Load images from disk and convert them into a format PyTorch can use.

📚 CONCEPTS YOU NEED TO LEARN FIRST:
    1. What is a Tensor? (Think: a multi-dimensional array/matrix)
    2. What is a Dataset? (A collection of data samples)
    3. What is a DataLoader? (A tool to feed data in batches)
    4. What is Normalization? (Scaling values to a standard range)

🔑 KEY PYTORCH CLASSES:
    - torch.utils.data.Dataset: Base class for all datasets
    - torch.utils.data.DataLoader: Loads data in batches
    - torchvision.transforms: Image transformations

📖 RESOURCES TO STUDY:
    - PyTorch Dataset Tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    - Transforms: https://pytorch.org/vision/stable/transforms.html

💡 THE BIG PICTURE:

    [Image Files on Disk]
            ↓
    [Load with PIL/OpenCV]
            ↓
    [Convert to Tensor (numbers)]
            ↓
    [Normalize (scale to 0-1)]
            ↓
    [Create Dataset object]
            ↓
    [Wrap in DataLoader]
            ↓
    [Feed to Model in batches]

================================================================================
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

# Import our config
import sys
sys.path.append('..')
from config import IMAGE_SIZE, BATCH_SIZE, DATA_DIR


class ArabicLetterDataset(Dataset):
    """
    Custom Dataset for Arabic Handwritten Letters.

    This class tells PyTorch HOW to:
    1. Find the data
    2. Load a single sample
    3. Transform it properly

    You MUST implement these methods:
    - __init__: Initialize the dataset (load CSV, setup transforms)
    - __len__: Return total number of samples
    - __getitem__: Return ONE sample (image, label) given an index

    -------------------------------------------------------------------------
    TODO: Complete this class!

    HINTS:
    - The CSV file probably has columns like: image_path, label
    - Use PIL.Image to load images
    - Use transforms to resize and convert to tensor
    -------------------------------------------------------------------------
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Initialize the dataset.

        Args:
            csv_file (str): Path to CSV file with image paths and labels
            root_dir (str): Directory with all the images
            transform: Optional transform to apply to images

        TODO:
        1. Load the CSV file using pandas
        2. Store the root directory
        3. Store/create the transform pipeline
        """
        # TODO: Load the CSV file
        # self.data_frame = pd.read_csv(???)

        # TODO: Store root directory
        # self.root_dir = ???

        # TODO: Store transform (or create default)
        # self.transform = ???

        pass  # Remove this when you implement

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        TODO:
        Return the number of rows in your dataframe

        Example:
            return len(self.data_frame)
        """
        # TODO: Implement this
        pass

    def __getitem__(self, idx):
        """
        Get ONE sample from the dataset.

        This method is called when you do: dataset[0], dataset[1], etc.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (image_tensor, label)

        TODO:
        1. Get the image path from the CSV
        2. Load the image using PIL
        3. Get the label from the CSV
        4. Apply transforms to the image
        5. Return (transformed_image, label)

        HINTS:
        - img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        - image = Image.open(img_path)
        - label = self.data_frame.iloc[idx, 1]
        """
        # TODO: Implement this
        pass


def get_transforms():
    """
    Create the transformation pipeline for images.

    Transforms do things like:
    - Resize images to consistent size
    - Convert to Tensor (0-255 → 0-1)
    - Normalize values

    TODO:
    Create a transforms.Compose() with:
    1. transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    2. transforms.Grayscale(num_output_channels=1)  # If images are grayscale
    3. transforms.ToTensor()  # Converts to tensor and scales to [0,1]
    4. transforms.Normalize((0.5,), (0.5,))  # Scale to [-1,1]

    Returns:
        transforms.Compose: The transformation pipeline
    """
    # TODO: Implement this
    # transform = transforms.Compose([
    #     ???
    # ])
    # return transform
    pass


def create_data_loaders(train_dataset, val_dataset):
    """
    Wrap datasets in DataLoaders.

    DataLoaders help by:
    - Loading data in batches (not all at once)
    - Shuffling data (for training)
    - Loading data in parallel (faster)

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset

    Returns:
        tuple: (train_loader, val_loader)

    TODO:
    Use torch.utils.data.DataLoader with:
    - batch_size=BATCH_SIZE
    - shuffle=True for training, False for validation
    - num_workers=2 (parallel loading)
    """
    # TODO: Implement this
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # return train_loader, val_loader
    pass


# =============================================================================
# TESTING YOUR IMPLEMENTATION
# =============================================================================
if __name__ == "__main__":
    """
    Test your dataset implementation here.

    Uncomment and run to verify everything works:
    """
    print("Testing Dataset Implementation...")

    # TODO: Uncomment when ready to test
    # transform = get_transforms()
    # dataset = ArabicLetterDataset(
    #     csv_file="path/to/your/labels.csv",
    #     root_dir="path/to/your/images",
    #     transform=transform
    # )
    #
    # print(f"Dataset size: {len(dataset)}")
    #
    # # Get one sample
    # image, label = dataset[0]
    # print(f"Image shape: {image.shape}")
    # print(f"Label: {label}")

    print("Dataset test complete!")
