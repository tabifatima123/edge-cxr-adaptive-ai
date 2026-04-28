"""
dataset.py
----------
Loads chest X-ray images from data/images/ and labels from data/labels.csv.
If the folder is empty or missing, generates synthetic demo images so the
pipeline still runs end-to-end (useful for grading / system testing).

Author: Edge CXR Benchmark Project
"""

import os
import csv
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"

# Common chest X-ray pathology labels (NIH ChestX-ray14 subset)
DEMO_CLASSES = [
    "No Finding", "Atelectasis", "Cardiomegaly",
    "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
]


# ---------------------------------------------------------------
# Synthetic demo image generator
# ---------------------------------------------------------------
def _generate_synthetic_xray(size: int = 224, seed: int = 0) -> Image.Image:
    """
    Generate a fake chest-X-ray-like grayscale image.
    Uses concentric ellipses + noise to roughly resemble lung fields.
    Purely for system testing — NOT for clinical use.
    """
    rng = np.random.RandomState(seed)
    # Base dark background
    img = np.zeros((size, size), dtype=np.float32)
    # Add bright "ribcage" ellipses
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2, size / 2
    for r in range(40, 110, 10):
        mask = ((xx - cx) ** 2 / (r ** 2) + (yy - cy) ** 2 / ((r + 20) ** 2)) < 1
        img[mask] += rng.uniform(0.05, 0.15)
    # Add lungs (two darker ellipses)
    for offset in [-40, 40]:
        mask = ((xx - (cx + offset)) ** 2 / 30 ** 2 + (yy - cy) ** 2 / 50 ** 2) < 1
        img[mask] *= 0.4
    # Noise
    img += rng.normal(0, 0.05, img.shape)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def ensure_demo_dataset(num_samples: int = 20) -> None:
    """
    If real dataset is missing or empty, create a synthetic demo dataset
    so the pipeline can run end-to-end. Writes images and labels.csv.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg"))
    if len(existing) >= 5 and LABELS_CSV.exists():
        print(f"[dataset] Real dataset detected: {len(existing)} images.")
        return

    print(f"[dataset] No real dataset found — generating {num_samples} synthetic demo images...")
    rows = [("filename", "label")]
    for i in range(num_samples):
        img = _generate_synthetic_xray(size=224, seed=i)
        fname = f"demo_{i:03d}.png"
        img.save(IMAGES_DIR / fname)
        label = DEMO_CLASSES[i % len(DEMO_CLASSES)]
        rows.append((fname, label))

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"[dataset] Wrote {num_samples} demo images + labels.csv")


# ---------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------
class ChestXrayDataset(Dataset):
    """Loads chest X-ray images + integer label indices."""

    def __init__(self, image_size: int = 224, transform=None):
        ensure_demo_dataset()
        self.image_size = image_size
        self.transform = transform
        self.samples: List[Tuple[Path, str]] = []

        # Read labels.csv
        if LABELS_CSV.exists():
            with open(LABELS_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fpath = IMAGES_DIR / row["filename"]
                    if fpath.exists():
                        self.samples.append((fpath, row["label"]))

        # Build label vocab
        self.classes = sorted(set(lbl for _, lbl in self.samples)) or DEMO_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        print(f"[dataset] Loaded {len(self.samples)} samples, {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize((self.image_size, self.image_size))
            arr = np.array(img).astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
            img = torch.from_numpy(arr).permute(2, 0, 1)
        return img, self.class_to_idx[label]


def get_default_transform(image_size: int = 224):
    """Default ImageNet-style transform."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


if __name__ == "__main__":
    ds = ChestXrayDataset(transform=get_default_transform())
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        x, y = ds[0]
        print(f"Sample tensor shape: {x.shape}, label: {y}")
