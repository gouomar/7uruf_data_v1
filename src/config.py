from pathlib import Path

TRAIN_DIR = Path("7uruf_data/train")   
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 64
IN_CHANNELS = 1
NUM_CLASSES = 28

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10
NUM_WORKERS = 2

VAL_RATIO = 0.2
SEED = 42
