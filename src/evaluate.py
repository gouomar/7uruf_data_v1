"""
================================================================================
PILLAR 4: VALIDATION & EVALUATION (evaluate.py)
================================================================================
Omar's part - Metrics, confusion matrix, classification report
================================================================================
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

# Import config
import sys

sys.path.append("..")
from config import DEVICE, NUM_CLASSES, OUTPUT_DIR


# Arabic letter labels
ARABIC_LETTERS = [
    "ا",
    "ب",
    "ت",
    "ث",
    "ج",
    "ح",
    "خ",
    "د",
    "ذ",
    "ر",
    "ز",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ع",
    "غ",
    "ف",
    "ق",
    "ك",
    "ل",
    "م",
    "ن",
    "ه",
    "و",
    "ي",
]


def evaluate_model(model, test_loader, device=DEVICE):
    """
    Evaluate the model on test data and return predictions.

    Args:
        model: The trained neural network
        test_loader: DataLoader with test data
        device: 'cuda' or 'cpu'

    Returns:
        tuple: (all_predictions, all_labels, accuracy)
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_predictions) * 100

    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    return all_predictions, all_labels, accuracy


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Create and display a confusion matrix visualization.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        labels: List of class names (Arabic letters)
        save_path: Path to save the figure (optional)
    """
    if labels is None:
        labels = ARABIC_LETTERS[:NUM_CLASSES]

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix - Arabic Letter Recognition")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Path to save the figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    ax1.plot(history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(history["val_loss"], label="Validation Loss", marker="o")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    ax2.plot(history["train_acc"], label="Train Accuracy", marker="o")
    ax2.plot(history["val_acc"], label="Validation Accuracy", marker="o")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")

    plt.show()


def get_classification_report(y_true, y_pred, labels=None, save_path=None):
    """
    Print and optionally save a detailed classification report.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        labels: List of class names
        save_path: Path to save the report (optional)
    """
    if labels is None:
        labels = ARABIC_LETTERS[:NUM_CLASSES]

    report = classification_report(y_true, y_pred, target_names=labels)
    print("\nClassification Report:")
    print("=" * 60)
    print(report)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("Classification Report\n")
            f.write("=" * 60 + "\n")
            f.write(report)
        print(f"Classification report saved to {save_path}")

    return report


def predict_single_image(model, image, device=DEVICE):
    """
    Make a prediction for a single image.

    Args:
        model: The trained model
        image: A preprocessed image tensor
        device: 'cuda' or 'cpu'

    Returns:
        tuple: (predicted_class, confidence_score, all_probabilities)
    """
    model.eval()

    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = predicted.item()
    confidence_score = confidence.item() * 100

    print(f"Predicted: {ARABIC_LETTERS[predicted_class]}")
    print(f"Confidence: {confidence_score:.2f}%")

    return predicted_class, confidence_score, probabilities.cpu().numpy()


def visualize_predictions(
    model, test_loader, num_images=16, device=DEVICE, save_path=None
):
    """
    Visualize model predictions on a grid of test images.

    Args:
        model: The trained model
        test_loader: DataLoader with test data
        num_images: Number of images to display
        device: 'cuda' or 'cpu'
        save_path: Path to save the figure (optional)
    """
    model.eval()

    # Get a batch of images
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = torch.max(outputs, 1)

    predictions = predictions.cpu()

    # Plot
    rows = int(np.ceil(np.sqrt(num_images)))
    cols = rows
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].squeeze().numpy()
            true_label = ARABIC_LETTERS[labels[i]]
            pred_label = ARABIC_LETTERS[predictions[i]]

            ax.imshow(img, cmap="gray")

            color = "green" if labels[i] == predictions[i] else "red"
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)

        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Predictions visualization saved to {save_path}")

    plt.show()


def get_per_class_accuracy(y_true, y_pred, labels=None):
    """
    Calculate accuracy for each class.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        labels: List of class names

    Returns:
        dict: Per-class accuracy
    """
    if labels is None:
        labels = ARABIC_LETTERS[:NUM_CLASSES]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    per_class_acc = {}
    for i, label in enumerate(labels):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean() * 100
            per_class_acc[label] = acc
        else:
            per_class_acc[label] = 0.0

    print("\nPer-Class Accuracy:")
    print("-" * 30)
    for label, acc in per_class_acc.items():
        print(f"  {label}: {acc:.2f}%")

    return per_class_acc


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Arabic letters: {ARABIC_LETTERS}")
