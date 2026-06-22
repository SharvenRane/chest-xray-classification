"""Model factory for multi-label chest X-ray classification."""
from __future__ import annotations

import timm
import torch.nn as nn

from dataset import NUM_CLASSES


def build_model(name: str = "densenet121", num_classes: int = NUM_CLASSES,
                pretrained: bool = True) -> nn.Module:
    """Create a timm backbone with a multi-label classification head.

    DenseNet-121 is the standard baseline for NIH ChestX-ray14 (CheXNet).
    Output logits are passed through BCEWithLogitsLoss (sigmoid per class).
    """
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
