"""
================================================================================
PILLAR 4: VALIDATION & EVALUATION (evaluate.py)
================================================================================

🎯 PURPOSE:
    Prove your model works on data it has NEVER seen before.
    A model that memorizes training data is useless!

📚 CONCEPTS YOU NEED TO LEARN FIRST:
    1. Accuracy: % of correct predictions
    2. Confusion Matrix: Shows what mistakes the model makes
    3. Overfitting: Model memorizes training data, fails on new data
    4. Train/Test Split: Separate data for training vs evaluation

🔑 KEY CONCEPTS:
    - Accuracy = (Correct Predictions) / (Total Predictions) × 100
    - Confusion Matrix: A table showing predicted vs actual labels
    - Per-class accuracy: How well does it recognize each letter?

📖 RESOURCES TO STUDY:
    - Confusion Matrix: https://www.youtube.com/watch?v=Kdsp6soqA7o
    - scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

💡 THE BIG PICTURE:

    CONFUSION MATRIX EXAMPLE (3 classes):

                    Predicted
                    أ    ب    ت
    Actual    أ   [45]   3    2     ← 45 correct, 5 mistakes
              ب    2   [48]   0     ← 48 correct, 2 mistakes
              ت    1    4   [45]    ← 45 correct, 5 mistakes

    Diagonal = Correct predictions
    Off-diagonal = Mistakes (which letters get confused?)

================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

# Import config
import sys
sys.path.append('..')
from config import DEVICE, NUM_CLASSES, OUTPUT_DIR


# Arabic letter labels (you may need to adjust based on your dataset)
ARABIC_LETTERS = [
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر',
    'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
    'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'
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

    TODO:
    1. Set model to evaluation mode
    2. Loop through test data
    3. Collect all predictions and true labels
    4. Calculate accuracy
    """
    # TODO: Implement this function

    # model.eval()
    # all_predictions = []
    # all_labels = []
    #
    # with torch.no_grad():
    #     for images, labels in tqdm(test_loader, desc="Evaluating"):
    #         images = images.to(device)
    #
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs, 1)
    #
    #         all_predictions.extend(predicted.cpu().numpy())
    #         all_labels.extend(labels.numpy())
    #
    # all_predictions = np.array(all_predictions)
    # all_labels = np.array(all_labels)
    # accuracy = accuracy_score(all_labels, all_predictions) * 100
    #
    # print(f"\nOverall Accuracy: {accuracy:.2f}%")
    #
    # return all_predictions, all_labels, accuracy

    pass  # Remove this when you implement


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Create and display a confusion matrix visualization.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        labels: List of class names (Arabic letters)
        save_path: Path to save the figure (optional)

    TODO:
    1. Calculate confusion matrix using sklearn
    2. Create a heatmap using seaborn
    3. Add labels and title
    4. Save or display
    """
    if labels is None:
        labels = ARABIC_LETTERS[:NUM_CLASSES]

    # TODO: Implement this function

    # # Calculate confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    #
    # # Create figure
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=labels, yticklabels=labels)
    # plt.title('Confusion Matrix - Arabic Letter Recognition')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path)
    #     print(f"Confusion matrix saved to {save_path}")
    #
    # plt.show()

    pass  # Remove this when you implement


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    These curves help you understand:
    - Is the model learning? (loss going down)
    - Is it overfitting? (train acc >> val acc)
    - When to stop training? (val loss starts going up)

    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Path to save the figure (optional)
    """
    # TODO: Implement this function

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    #
    # # Plot Loss
    # ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    # ax1.plot(history['val_loss'], label='Validation Loss', marker='o')
    # ax1.set_title('Loss over Epochs')
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss')
    # ax1.legend()
    # ax1.grid(True)
    #
    # # Plot Accuracy
    # ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    # ax2.plot(history['val_acc'], label='Validation Accuracy', marker='o')
    # ax2.set_title('Accuracy over Epochs')
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Accuracy (%)')
    # ax2.legend()
    # ax2.grid(True)
    #
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path)
    #     print(f"Training history saved to {save_path}")
    #
    # plt.show()

    pass  # Remove this when you implement


def get_classification_report(y_true, y_pred, labels=None):
    """
    Print a detailed classification report with precision, recall, F1-score.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        labels: List of class names
    """
    if labels is None:
        labels = ARABIC_LETTERS[:NUM_CLASSES]

    # TODO: Implement this function

    # report = classification_report(y_true, y_pred, target_names=labels)
    # print("\nClassification Report:")
    # print("=" * 60)
    # print(report)

    pass  # Remove this when you implement


def predict_single_image(model, image, device=DEVICE):
    """
    Make a prediction for a single image.

    Args:
        model: The trained model
        image: A preprocessed image tensor
        device: 'cuda' or 'cpu'

    Returns:
        tuple: (predicted_class, confidence_score, all_probabilities)

    This is useful for:
    - Testing with your own handwritten letters
    - Building a demo/app
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


def visualize_predictions(model, test_loader, num_images=16, device=DEVICE):
    """
    Visualize model predictions on a grid of test images.

    Shows the image, true label, and predicted label.
    Correct predictions in green, wrong in red.

    Args:
        model: The trained model
        test_loader: DataLoader with test data
        num_images: Number of images to display
        device: 'cuda' or 'cpu'
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
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].squeeze().numpy()
            true_label = ARABIC_LETTERS[labels[i]]
            pred_label = ARABIC_LETTERS[predictions[i]]

            ax.imshow(img, cmap='gray')

            color = 'green' if labels[i] == predictions[i] else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/prediction_samples.png')
    plt.show()


# =============================================================================
# TESTING
# =============================================================================
if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Arabic letters: {ARABIC_LETTERS}")
