"""
================================================================================
PILLAR 3: THE TRAINING LOOP (train.py)
================================================================================

🎯 PURPOSE:
    This is where the LEARNING happens! The training loop shows images to
    the model, checks if it guessed correctly, and updates the brain.

📚 CONCEPTS YOU NEED TO LEARN FIRST:
    1. Forward Pass: Input goes through the network to get predictions
    2. Loss Function: Measures how wrong the predictions are
    3. Backpropagation: Calculates how to fix the mistakes
    4. Optimizer: Actually updates the network weights
    5. Epoch: One complete pass through all training data
    6. Batch: A small group of samples processed together

🔑 KEY PYTORCH CLASSES:
    - nn.CrossEntropyLoss: Loss function for classification
    - optim.Adam: Optimizer (adaptive learning rate)
    - model.zero_grad(): Clear previous gradients
    - loss.backward(): Backpropagation
    - optimizer.step(): Update weights

📖 RESOURCES TO STUDY:
    - PyTorch Training Loop: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    - Cross Entropy Loss Explained: https://www.youtube.com/watch?v=6ArSys5qHAU

💡 THE BIG PICTURE - The Training Loop:

    FOR each epoch (pass through all data):
        │
        FOR each batch of images:
        │    │
        │    ├─→ 1. FORWARD PASS
        │    │      predictions = model(images)
        │    │      "Show the image to the brain and get a guess"
        │    │
        │    ├─→ 2. CALCULATE LOSS
        │    │      loss = criterion(predictions, true_labels)
        │    │      "How wrong was the guess?"
        │    │
        │    ├─→ 3. BACKWARD PASS (Backpropagation)
        │    │      loss.backward()
        │    │      "Figure out which weights caused the error"
        │    │
        │    └─→ 4. UPDATE WEIGHTS
        │           optimizer.step()
        │           "Adjust the brain to make fewer mistakes"
        │
        └─→ Print progress, validate on test set

================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Progress bar

# Import config
import sys
sys.path.append('..')
from config import DEVICE, NUM_EPOCHS, LEARNING_RATE


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for ONE epoch (one pass through all training data).

    Args:
        model: The neural network
        train_loader: DataLoader with training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device: 'cuda' or 'cpu'

    Returns:
        tuple: (average_loss, accuracy) for this epoch

    TODO:
    1. Set model to training mode: model.train()
    2. Loop through batches in train_loader
    3. For each batch:
       a. Move data to device (GPU/CPU)
       b. Zero the gradients: optimizer.zero_grad()
       c. Forward pass: outputs = model(images)
       d. Calculate loss: loss = criterion(outputs, labels)
       e. Backward pass: loss.backward()
       f. Update weights: optimizer.step()
    4. Track total loss and correct predictions
    5. Return average loss and accuracy
    """
    # TODO: Implement this function

    # Set model to training mode
    # model.train()

    # Initialize tracking variables
    # running_loss = 0.0
    # correct = 0
    # total = 0

    # Loop through batches
    # for images, labels in tqdm(train_loader, desc="Training"):
    #     # Move to device
    #     images = images.to(device)
    #     labels = labels.to(device)
    #
    #     # Zero gradients
    #     optimizer.zero_grad()
    #
    #     # Forward pass
    #     outputs = model(images)
    #
    #     # Calculate loss
    #     loss = criterion(outputs, labels)
    #
    #     # Backward pass
    #     loss.backward()
    #
    #     # Update weights
    #     optimizer.step()
    #
    #     # Track statistics
    #     running_loss += loss.item()
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()

    # Calculate averages
    # epoch_loss = running_loss / len(train_loader)
    # epoch_acc = 100 * correct / total

    # return epoch_loss, epoch_acc

    pass  # Remove this when you implement


def validate(model, val_loader, criterion, device):
    """
    Evaluate the model on validation data (NO training, just testing).

    Args:
        model: The neural network
        val_loader: DataLoader with validation data
        criterion: Loss function
        device: 'cuda' or 'cpu'

    Returns:
        tuple: (average_loss, accuracy) on validation set

    TODO:
    Similar to train_one_epoch BUT:
    1. Set model to evaluation mode: model.eval()
    2. Use torch.no_grad() context (no gradient calculation needed)
    3. Do NOT call optimizer.zero_grad(), loss.backward(), or optimizer.step()
    """
    # TODO: Implement this function

    # model.eval()  # Evaluation mode
    # running_loss = 0.0
    # correct = 0
    # total = 0
    #
    # with torch.no_grad():  # No gradients needed for validation
    #     for images, labels in tqdm(val_loader, desc="Validating"):
    #         images = images.to(device)
    #         labels = labels.to(device)
    #
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #
    #         running_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # val_loss = running_loss / len(val_loader)
    # val_acc = 100 * correct / total
    #
    # return val_loss, val_acc

    pass  # Remove this when you implement


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE, device=DEVICE):
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

    TODO:
    1. Move model to device
    2. Create criterion (CrossEntropyLoss)
    3. Create optimizer (Adam)
    4. Loop for num_epochs:
       a. Train one epoch
       b. Validate
       c. Print progress
       d. Save best model
    5. Return training history
    """
    print(f"Training on {device}")
    print(f"Epochs: {num_epochs}, Learning Rate: {learning_rate}")
    print("=" * 60)

    # TODO: Implement this function

    # Move model to device
    # model = model.to(device)

    # Loss function: CrossEntropyLoss for multi-class classification
    # criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam is a good default choice
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track training history
    # history = {
    #     'train_loss': [],
    #     'train_acc': [],
    #     'val_loss': [],
    #     'val_acc': []
    # }

    # Best validation accuracy (for saving best model)
    # best_val_acc = 0.0

    # Training loop
    # for epoch in range(num_epochs):
    #     print(f"\nEpoch {epoch+1}/{num_epochs}")
    #     print("-" * 30)
    #
    #     # Train
    #     train_loss, train_acc = train_one_epoch(
    #         model, train_loader, criterion, optimizer, device
    #     )
    #
    #     # Validate
    #     val_loss, val_acc = validate(model, val_loader, criterion, device)
    #
    #     # Store history
    #     history['train_loss'].append(train_loss)
    #     history['train_acc'].append(train_acc)
    #     history['val_loss'].append(val_loss)
    #     history['val_acc'].append(val_acc)
    #
    #     # Print progress
    #     print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    #     print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    #
    #     # Save best model
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), 'models/best_model.pth')
    #         print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

    # print("\n" + "=" * 60)
    # print(f"Training complete! Best Val Accuracy: {best_val_acc:.2f}%")
    #
    # return history

    pass  # Remove this when you implement


def save_model(model, path):
    """
    Save the model weights to a file.

    Args:
        model: The trained model
        path: File path to save to (e.g., 'models/my_model.pth')
    """
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


# =============================================================================
# TESTING
# =============================================================================
if __name__ == "__main__":
    print("Training module loaded successfully!")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
