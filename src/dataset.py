import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import config



def extract_label_from_filename(fname: str) -> int:
    label = int(fname.split("_label_")[1].replace(".png", ""))
    return label - 1  


def get_transforms(train: bool):
    base = [
        T.Grayscale(num_output_channels=config.IN_CHANNELS),
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    ]

    aug = []
    if train:
        aug = [
            T.RandomRotation(10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]

    tail = [
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ]

    return T.Compose(base + aug + tail)



class ArabicLetterDataset(Dataset):
    def __init__(self, img_dir: str, transform):
        self.img_dir = img_dir
        self.transform = transform

        self.files = [f for f in os.listdir(self.img_dir) if f.endswith(".png")]
        self.files.sort()

        if len(self.files) == 0:
            raise RuntimeError(f"No .png images found in {self.img_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = extract_label_from_filename(fname)

        return image, label


def create_datasets():
    train_dir = str(config.TRAIN_DIR)
    if not os.path.isabs(train_dir):
        train_dir = os.path.join(ROOT_DIR, train_dir)

    full_ds = ArabicLetterDataset(
        train_dir,
        transform=get_transforms(train=True)
    )

    val_len = int(config.VAL_RATIO * len(full_ds))
    train_len = len(full_ds) - val_len

    g = torch.Generator().manual_seed(config.SEED)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=g)

    val_ds.dataset.transform = get_transforms(train=False)

    return train_ds, val_ds


def create_data_loaders():
    train_ds, val_ds = create_datasets()

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader



if __name__ == "__main__":
    print("=== DATASET TEST ===")

    train_loader, val_loader = create_data_loaders()

    x, y = next(iter(train_loader))
    print("Train batch images:", x.shape)
    print("Train batch labels:", y[:10].tolist())

    xv, yv = next(iter(val_loader))
    print("Val batch images:", xv.shape)

    print("=== OK ===")
