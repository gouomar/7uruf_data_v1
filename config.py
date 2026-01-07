"""
Configuration File - All hyperparameters in one place
=====================================================

WHY THIS FILE EXISTS:
Instead of having magic numbers scattered throughout your code,
we put all settings here. This makes it easy to experiment!

WHAT ARE HYPERPARAMETERS?
These are the "knobs" you can turn to change how your model learns.
They are NOT learned by the model - YOU choose them.
"""

import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
DATA_DIR = "data/raw"           # Where your images and CSV are
PROCESSED_DIR = "data/processed" # Where processed data goes
MODEL_DIR = "models"            # Where to save trained models
OUTPUT_DIR = "outputs"          # Where to save graphs/results

# =============================================================================
# DATA PARAMETERS
# =============================================================================
IMAGE_SIZE = 32                 # Resize all images to 32x32 pixels
                                # Why 32? Good balance between detail and speed
                                # Arabic letters don't need huge resolution

NUM_CLASSES = 28                # How many Arabic letters? (ا to ي)
                                # TODO: Verify this matches your dataset!

TRAIN_SPLIT = 0.8               # 80% for training, 20% for validation
                                # This is a common split ratio

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 32                 # How many images to process at once
                                # Larger = faster but needs more memory
                                # Start with 32, increase if you have good GPU

LEARNING_RATE = 0.001           # How big of a step to take when learning
                                # Too high = unstable, too low = slow
                                # 0.001 is a safe starting point for Adam

NUM_EPOCHS = 20                 # How many times to go through ALL the data
                                # More epochs = more learning (up to a point)
                                # Watch for overfitting!

# =============================================================================
# MODEL ARCHITECTURE (You'll understand these after Phase B)
# =============================================================================
# These define the structure of your CNN
CONV1_OUT_CHANNELS = 32         # First conv layer output channels
CONV2_OUT_CHANNELS = 64         # Second conv layer output channels
HIDDEN_SIZE = 128               # Fully connected hidden layer size

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
# This automatically uses GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# RANDOM SEED (for reproducibility)
# =============================================================================
SEED = 42                       # The answer to life, universe, and everything
                                # Setting this makes your results reproducible


# =============================================================================
# PRINT CONFIGURATION (helpful for debugging)
# =============================================================================
def print_config():
    """Print all configuration values."""
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
