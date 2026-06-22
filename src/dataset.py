"""NIH ChestX-ray14 multi-label dataset.

Uses MedMNIST's ChestMNIST, which is the NIH ChestX-ray14 dataset standardized
to fixed resolutions (28/64/128/224). 14 binary pathology labels, multi-label.
Downloads automatically on first run, no credentialed access required.

Reference: Yang et al., "MedMNIST v2", Scientific Data 2023.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

# 14 NIH ChestX-ray14 pathology labels, in ChestMNIST channel order.
CHEST_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(CHEST_LABELS)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(size: int, train: bool) -> T.Compose:
    ops = []
    if train:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomRotation(7),
            T.RandomResizedCrop(size, scale=(0.9, 1.0), antialias=True),
        ]
    else:
        ops += [T.Resize((size, size), antialias=True)]
    # ChestMNIST is single-channel; replicate to 3 channels for ImageNet backbones.
    ops += [
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    return T.Compose(ops)


def _to_float_label(_img, label):
    # ChestMNIST returns label as an int array of shape (14,); BCE needs float.
    return torch.as_tensor(label, dtype=torch.float32)


class _LabelCastWrapper(torch.utils.data.Dataset):
    """Casts ChestMNIST's multi-label target to float32 for BCEWithLogitsLoss."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, torch.as_tensor(label, dtype=torch.float32).reshape(-1)


def get_dataloaders(size: int = 224, batch_size: int = 64, num_workers: int = 4,
                    data_dir: str | None = None):
    """Returns (train_loader, val_loader, test_loader) for ChestMNIST."""
    from medmnist import ChestMNIST

    common = dict(download=True, size=size)
    if data_dir:
        common["root"] = data_dir

    train = _LabelCastWrapper(ChestMNIST(split="train", transform=build_transforms(size, True), **common))
    val = _LabelCastWrapper(ChestMNIST(split="val", transform=build_transforms(size, False), **common))
    test = _LabelCastWrapper(ChestMNIST(split="test", transform=build_transforms(size, False), **common))

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True, drop_last=shuffle)

    return loader(train, True), loader(val, False), loader(test, False)


def pos_weight(train_loader) -> torch.Tensor:
    """Class imbalance handling: positive weight = (#neg / #pos) per label.

    Reads the raw label matrix directly (no image loading) for speed.
    """
    base = train_loader.dataset.base  # _LabelCastWrapper -> ChestMNIST
    labels = np.asarray(base.labels, dtype=np.float32).reshape(len(base), -1)
    pos = labels.sum(axis=0)
    neg = len(base) - pos
    w = np.clip(neg / np.clip(pos, 1, None), a_min=0.0, a_max=50.0)
    return torch.as_tensor(w, dtype=torch.float32)
