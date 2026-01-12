"""
================================================================================
PILLAR 3: THE TRAINING LOOP (train.py)
================================================================================
Omar's part - Training loop, loss, backward, optimizer step
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import config
import sys

sys.path.append("..")
from config import DEVICE, NUM_EPOCHS, LEARNING_RATE, MODEL_DIR


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for ONE epoch.

    Args:
        model: The neural network
        train_loader: DataLoader with training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device: 'cuda' or 'cpu'

    Returns:
        tuple: (average_loss, accuracy) for this epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Evaluate the model on validation data (NO training).

    Args:
        model: The neural network
        val_loader: DataLoader with validation data
        criterion: Loss function
        device: 'cuda' or 'cpu'

    Returns:
        tuple: (average_loss, accuracy) on validation set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total

    return val_loss, val_acc


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
):
    """
    Full training loop - trains the model for multiple epochs.

    Args:
        model: The neural network
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: 'cuda' or 'cpu'

    Returns:
        dict: Training history with losses and accuracies
    """
    print(f"Training on {device}")
    print(f"Epochs: {num_epochs}, Learning Rate: {learning_rate}")
    print("=" * 60)

    # Move model to device
    model = model.to(device)

    # Loss function: CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Best validation accuracy (for saving best model)
    best_val_acc = 0.0

    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Store history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

    print("\n" + "=" * 60)
    print(f"Training complete! Best Val Accuracy: {best_val_acc:.2f}%")

    return history


def save_model(model, path):
    """
    Save the model weights to a file.

    Args:
        model: The trained model
        path: File path to save to (e.g., 'models/my_model.pth')
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device=DEVICE):
    """
    Load model weights from a file.

    Args:
        model: The model architecture (must match saved model)
        path: File path to load from
        device: Device to load to

    Returns:
        model: Model with loaded weights
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
