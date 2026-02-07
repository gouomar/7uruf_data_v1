# 7uruf Vision - حروف
## Arabic Alphabet Recognition with Computer Vision

Build a deep learning model that recognizes handwritten Arabic letters using convolutional neural networks (CNN).

---

## 🎯 What This Project Does

This project trains an AI model to recognize and classify handwritten Arabic letters. You'll learn the complete machine learning pipeline: from raw image data to a trained neural network that can predict letters with high accuracy.

**Input:** 📸 Handwritten Arabic letter image  
**Output:** 🎯 Predicted letter (ا, ب, ت, ث, ج, etc.)

---

## 📊 Project Pipeline

```
┌─────────────┐        ┌──────────────┐        ┌──────────┐        ┌──────────┐
│  RAW DATA   │        │  PREPROCESS  │        │  TRAIN   │        │ EVALUATE │
│             │───────►│              │───────►│          │───────►│          │
│  Images     │        │  Resize      │        │  Model   │        │ Results  │
│  + Labels   │        │  Normalize   │        │ Learning │        │ & Metrics│
└─────────────┘        └──────────────┘        └──────────┘        └──────────┘
       △                      │                      │                   │
       │                      │                      │                   │
       └──────────────────────┴──────────────────────┴───────────────────┘
                         (Feedback & Optimization)
```

---

## 🔄 How It Works - 5 Steps

### Step 1: **Data Preparation**
- Load handwritten Arabic letter images
- Resize all images to a standard size (32×32 pixels)
- Normalize pixel values to [-1, 1] range
- Split into training and testing sets

### Step 2: **Build the CNN Model**
- Create a convolutional neural network architecture
- Extract features from images (edges, shapes, patterns)
- Final layer predicts the letter class

### Step 3: **Train the Model**
- Feed batches of images through the network
- Calculate prediction errors (loss)
- Adjust model weights using backpropagation
- Repeat for multiple epochs until model converges

### Step 4: **Validate & Evaluate**
- Test on unseen data
- Calculate accuracy, precision, recall
- Generate confusion matrix
- Visualize predictions

### Step 5: **Deploy & Use**
- Save trained model
- Use for inference on new handwritten letters

---

## 📁 Project Structure

```
src/
├── dataset.py      # Data loading and preprocessing
├── model.py        # CNN architecture definition
├── train.py        # Training loop & optimization
├── evaluate.py     # Validation & performance metrics
└── utils.py        # Helper functions

notebooks/
├── 01_data_exploration.ipynb      # Explore your dataset
├── 02_model_experiments.ipynb     # Test architectures
└── 03_final_training.ipynb        # Full training pipeline

config.py           # Hyperparameters & settings
main.py             # Run everything end-to-end
requirements.txt    # Python dependencies
outputs/            # Results, graphs, confusion matrices
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your dataset in `7uruf_data/` with the following structure:
```
7uruf_data/
├── images/          # All image files
└── labels.csv       # File mapping images to letter classes
```

### 3. Explore the Data (Optional)
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. Train the Model
```bash
python main.py
```

### 5. Review Results
Check the `outputs/` folder for:
- Training curves
- Confusion matrix
- Classification report
- Model accuracy

---

## 🧠 Key Concepts

| Concept | What It Does |
|---------|-------------|
| **CNN (Convolutional Neural Network)** | Automatically learns patterns in images |
| **Convolutional Layers** | Extract local features (edges, shapes) |
| **Pooling Layers** | Reduce dimensionality, keep important info |
| **Loss Function** | Measures how wrong predictions are |
| **Backpropagation** | Adjusts weights to minimize loss |
| **Batch Processing** | Train on multiple images at once |

---

## 📈 Model Architecture Overview

```
Input Image (32×32×1)
        ↓
   Conv Layer 1    (Learn edges & simple patterns)
        ↓
   Pool Layer 1    (Reduce size)
        ↓
   Conv Layer 2    (Learn complex patterns)
        ↓
   Pool Layer 2    (Reduce size)
        ↓
  Flatten Layer    (Convert to 1D vector)
        ↓
 Dense Layers      (Final classification)
        ↓
  Output Layer     (28 letter classes)
        ↓
  Prediction ✓     (Arabic letter)
```

---

## 💡 Tips for Success

- **Start small**: Use `notebooks/01_data_exploration.ipynb` to understand your data first
- **Experiment**: Try different architectures in `notebooks/02_model_experiments.ipynb`
- **Monitor**: Watch training/validation loss curves to spot overfitting
- **Tune**: Adjust hyperparameters in `config.py` (learning rate, batch size, epochs)
- **Evaluate**: Use confusion matrix to identify which letters are hardest to classify

---

## 📊 Expected Outcomes

After training, you should achieve:
- **Overall Accuracy**: 85-95%
- **Per-class Accuracy**: Varies by letter complexity
- **Fast Inference**: Predictions in milliseconds

---

## 🔗 Files Reference

| File | Purpose |
|------|---------|
| `main.py` | Entry point - runs full pipeline |
| `src/dataset.py` | Data loading & transformations |
| `src/model.py` | CNN architecture |
| `src/train.py` | Training loop |
| `src/evaluate.py` | Metrics & visualization |
| `config.py` | All hyperparameters |

---

## ❓ Troubleshooting

**Model not improving?**
- Check data quality and labels
- Adjust learning rate in `config.py`
- Try more epochs or different architecture

**Out of memory?**
- Reduce batch size in `config.py`
- Use a GPU if available

**Poor accuracy on specific letters?**
- Check if those letters have enough training samples
- Review confusion matrix to see which letters are confused

---

Happy training! 🎉
