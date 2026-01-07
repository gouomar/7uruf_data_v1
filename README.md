# 7uruf Vision - حروف
## Arabic Alphabet Recognition through Computer Vision

Welcome to your first AI project! This project will teach you to build a neural network that can recognize handwritten Arabic letters.

---

## 📁 Project Structure

```
7uruf/
│
├── data/
│   ├── raw/              # Original dataset (images + CSV) goes here
│   └── processed/        # Preprocessed/cleaned data
│
├── src/
│   ├── __init__.py
│   ├── dataset.py        # PILLAR 1: Data loading & preprocessing
│   ├── model.py          # PILLAR 2: CNN architecture definition
│   ├── train.py          # PILLAR 3: Training loop
│   ├── evaluate.py       # PILLAR 4: Validation & metrics
│   └── utils.py          # Helper functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Explore your data first!
│   ├── 02_model_experiments.ipynb   # Try different architectures
│   └── 03_final_training.ipynb      # Final training & evaluation
│
├── models/               # Saved trained models (.pth files)
├── outputs/              # Graphs, confusion matrices, results
│
├── config.py             # All hyperparameters in one place
├── main.py               # Main entry point to run everything
└── requirements.txt      # Python dependencies
```

---

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your data** in `data/raw/`

3. **Explore the data** using `notebooks/01_data_exploration.ipynb`

4. **Train your model:**
   ```bash
   python main.py
   ```

---

## 📚 The Four Pillars

| Pillar | File | What You'll Learn |
|--------|------|-------------------|
| 1. Data Ingestion | `src/dataset.py` | Loading images, creating tensors, DataLoader |
| 2. Architecture | `src/model.py` | CNN layers, feature extraction |
| 3. Training | `src/train.py` | Forward pass, loss, backpropagation |
| 4. Validation | `src/evaluate.py` | Accuracy, confusion matrix |

---

## 🎯 Your Goal

**Input:** An image of a handwritten Arabic letter (e.g., أ, ب, ت)
**Output:** The correct letter classification

---

Good luck! 🌟
