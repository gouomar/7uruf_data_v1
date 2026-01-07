"""
src/__init__.py
===============
This file makes the 'src' folder a Python package.
You can leave it empty or import your modules here for easier access.
"""

from .dataset import ArabicLetterDataset
from .model import ArabicCNN
from .train import train_model
from .evaluate import evaluate_model
