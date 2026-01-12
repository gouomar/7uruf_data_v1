"""
================================================================================
MAIN.PY - Entry point for the Arabic Letter Recognition Project
================================================================================
Omar's part - Main orchestration so python main.py works end-to-end
================================================================================

Run this file to:
1. Load the data
2. Create the model
3. Train the model
4. Evaluate performance

Usage:
    python main.py
================================================================================
"""

import os
import torch
from pathlib import Path

# Import our modules
from config import (
    DEVICE,
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    NUM_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_CLASSES,
    IMAGE_SIZE,
    print_config,
)
from src.dataset import ArabicLetterDataset, get_transforms, create_data_loaders
from src.model import ArabicCNN
from src.train import train_model, save_model
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    get_classification_report,
    get_per_class_accuracy,
    visualize_predictions,
)
from src.utils import set_seed, check_gpu


def main():
    """
    Main function - orchestrates the entire pipeline.
    """

    print("=" * 60)
    print("   🔤 7URUF VISION - Arabic Letter Recognition 🔤")
    print("=" * 60)

    # =========================================================================
    # STEP 0: SETUP
    # =========================================================================
    print("\n📋 STEP 0: Setup")
    print("-" * 40)

    set_seed()
    check_gpu()
    print_config()

    # Create directories if they don't exist
    Path(MODEL_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n📁 STEP 1: Loading Data")
    print("-" * 40)

    # Get transforms
    transform = get_transforms()

    # Load full dataset
    # NOTE: Adjust csv_file and root_dir based on your actual data structure
    # Nabil should have set this up in dataset.py
    csv_path = os.path.join(DATA_DIR, "labels.csv")
    images_dir = os.path.join(DATA_DIR, "images")

    # Check if data exists
    if not os.path.exists(csv_path):
        print(f"⚠️  Warning: CSV file not found at {csv_path}")
        print("   Please make sure your data is in the correct location.")
        print("   Expected: data/raw/labels.csv and data/raw/images/")
        print("\n   For testing, creating dummy data...")

        # Create dummy data for testing the pipeline
        train_loader, val_loader = create_dummy_data()
    else:
        full_dataset = ArabicLetterDataset(
            csv_file=csv_path, root_dir=images_dir, transform=transform
        )

        # Split into train and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        # Create DataLoaders
        train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

        print(f"✓ Training samples: {len(train_dataset)}")
        print(f"✓ Validation samples: {len(val_dataset)}")

    # =========================================================================
    # STEP 2: CREATE MODEL
    # =========================================================================
    print("\n🧠 STEP 2: Creating Model")
    print("-" * 40)

    model = ArabicCNN()
    model = model.to(DEVICE)

    # Print model summary
    print(f"Model: ArabicCNN")
    print(
        f"Input: ({1}, {IMAGE_SIZE}, {IMAGE_SIZE}) - Grayscale {IMAGE_SIZE}x{IMAGE_SIZE}"
    )
    print(f"Output: {NUM_CLASSES} classes (Arabic letters)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # STEP 3: TRAIN MODEL
    # =========================================================================
    print("\n🏋️ STEP 3: Training Model")
    print("-" * 40)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
    )

    # Save the final model
    final_model_path = os.path.join(MODEL_DIR, "final_model.pth")
    save_model(model, final_model_path)

    # Plot training history
    history_plot_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plot_training_history(history, save_path=history_plot_path)

    # =========================================================================
    # STEP 4: EVALUATE MODEL
    # =========================================================================
    print("\n📊 STEP 4: Evaluating Model")
    print("-" * 40)

    # Evaluate on validation data
    predictions, labels, accuracy = evaluate_model(model, val_loader, DEVICE)

    # Per-class accuracy
    get_per_class_accuracy(labels, predictions)

    # Print classification report
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    get_classification_report(labels, predictions, save_path=report_path)

    # Plot confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(labels, predictions, save_path=cm_path)

    # Visualize some predictions
    pred_viz_path = os.path.join(OUTPUT_DIR, "prediction_samples.png")
    visualize_predictions(
        model, val_loader, num_images=16, device=DEVICE, save_path=pred_viz_path
    )

    # =========================================================================
    # DONE!
    # =========================================================================
    print("\n" + "=" * 60)
    print("   ✅ Pipeline Complete!")
    print("=" * 60)
    print(f"\n📊 Final Results:")
    print(f"   - Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"   - Final Validation Accuracy: {accuracy:.2f}%")
    print(f"\n📁 Outputs saved to:")
    print(f"   - Model: {MODEL_DIR}/best_model.pth")
    print(f"   - Confusion Matrix: {OUTPUT_DIR}/confusion_matrix.png")
    print(f"   - Training History: {OUTPUT_DIR}/training_history.png")
    print(f"   - Classification Report: {OUTPUT_DIR}/classification_report.txt")
    print("\n🎉 Great job, team!")


def create_dummy_data():
    """
    Create dummy data for testing the pipeline when real data is not available.
    This helps verify the code works end-to-end.
    """
    from torch.utils.data import TensorDataset, DataLoader

    print("Creating dummy data for pipeline testing...")

    # Create random images and labels
    num_train = 500
    num_val = 100

    # Random grayscale images (1 channel, IMAGE_SIZE x IMAGE_SIZE)
    train_images = torch.randn(num_train, 1, IMAGE_SIZE, IMAGE_SIZE)
    train_labels = torch.randint(0, NUM_CLASSES, (num_train,))

    val_images = torch.randn(num_val, 1, IMAGE_SIZE, IMAGE_SIZE)
    val_labels = torch.randint(0, NUM_CLASSES, (num_val,))

    # Create datasets
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"✓ Created {num_train} dummy training samples")
    print(f"✓ Created {num_val} dummy validation samples")
    print("⚠️  Note: These are random images, accuracy will be ~random chance")

    return train_loader, val_loader


if __name__ == "__main__":
    main()
