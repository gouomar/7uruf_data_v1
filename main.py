"""
================================================================================
MAIN.PY - Entry point for the Arabic Letter Recognition Project
================================================================================

This is where everything comes together!

Run this file to:
1. Load the data
2. Create the model
3. Train the model
4. Evaluate performance

Usage:
    python main.py

================================================================================
"""

import torch
from pathlib import Path

# Import our modules
from config import (
    DEVICE, DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    print_config
)
from src.dataset import ArabicLetterDataset, get_transforms, create_data_loaders
from src.model import ArabicCNN, print_model_summary
from src.train import train_model, save_model
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    get_classification_report
)
from src.utils import set_seed, check_gpu


def main():
    """
    Main function - orchestrates the entire pipeline.

    TODO: Complete this function by following the steps below.
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

    # TODO: Implement data loading
    #
    # transform = get_transforms()
    #
    # # Load full dataset
    # full_dataset = ArabicLetterDataset(
    #     csv_file=f"{DATA_DIR}/labels.csv",  # Adjust filename
    #     root_dir=f"{DATA_DIR}/images",       # Adjust folder
    #     transform=transform
    # )
    #
    # # Split into train and validation
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     full_dataset, [train_size, val_size]
    # )
    #
    # # Create DataLoaders
    # train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)
    #
    # print(f"Training samples: {len(train_dataset)}")
    # print(f"Validation samples: {len(val_dataset)}")

    print("⚠️  TODO: Implement data loading in this section")

    # =========================================================================
    # STEP 2: CREATE MODEL
    # =========================================================================
    print("\n🧠 STEP 2: Creating Model")
    print("-" * 40)

    # TODO: Create and initialize model
    #
    # model = ArabicCNN()
    # model = model.to(DEVICE)
    # print_model_summary(model)

    print("⚠️  TODO: Implement model creation in this section")

    # =========================================================================
    # STEP 3: TRAIN MODEL
    # =========================================================================
    print("\n🏋️ STEP 3: Training Model")
    print("-" * 40)

    # TODO: Train the model
    #
    # history = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=NUM_EPOCHS,
    #     learning_rate=LEARNING_RATE,
    #     device=DEVICE
    # )
    #
    # # Save the final model
    # save_model(model, f"{MODEL_DIR}/final_model.pth")
    #
    # # Plot training history
    # plot_training_history(history, f"{OUTPUT_DIR}/training_history.png")

    print("⚠️  TODO: Implement training in this section")

    # =========================================================================
    # STEP 4: EVALUATE MODEL
    # =========================================================================
    print("\n📊 STEP 4: Evaluating Model")
    print("-" * 40)

    # TODO: Evaluate on test data
    #
    # predictions, labels, accuracy = evaluate_model(model, val_loader, DEVICE)
    #
    # # Print classification report
    # get_classification_report(labels, predictions)
    #
    # # Plot confusion matrix
    # plot_confusion_matrix(
    #     labels, predictions,
    #     save_path=f"{OUTPUT_DIR}/confusion_matrix.png"
    # )

    print("⚠️  TODO: Implement evaluation in this section")

    # =========================================================================
    # DONE!
    # =========================================================================
    print("\n" + "=" * 60)
    print("   ✅ Pipeline Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the outputs/ folder for graphs")
    print("2. Check the models/ folder for saved model")
    print("3. Try improving the model architecture!")


if __name__ == "__main__":
    main()
