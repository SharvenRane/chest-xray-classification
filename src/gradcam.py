"""Grad-CAM explainability for the chest X-ray classifier.

Saves an overlay heatmap for a given test image and target pathology, so a
human can see where the model is looking. Uses pytorch-grad-cam.

Usage:
    python src/gradcam.py --checkpoint outputs/best.pt --index 0 --label Cardiomegaly
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image

from dataset import get_dataloaders, CHEST_LABELS, _IMAGENET_MEAN, _IMAGENET_STD
from model import build_model


def _denormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    img = (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--label", default="Cardiomegaly")
    ap.add_argument("--out", default="outputs/gradcam.png")
    args = ap.parse_args()

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]
    model = build_model(cfg["model"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, _, test_loader = get_dataloaders(size=cfg["image_size"], batch_size=1,
                                        num_workers=0, data_dir=cfg.get("data_dir"))
    img_tensor = None
    for i, (x, _) in enumerate(test_loader):
        if i == args.index:
            img_tensor = x
            break

    target_layers = [list(model.modules())[-2] if not hasattr(model, "features")
                     else model.features[-1]]
    class_idx = CHEST_LABELS.index(args.label)

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor.to(device),
                        targets=[ClassifierOutputTarget(class_idx)])[0]
    rgb = _denormalize(img_tensor[0])
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Image.fromarray(overlay).save(args.out)
    print(f"Saved Grad-CAM for '{args.label}' (test index {args.index}) -> {args.out}")


if __name__ == "__main__":
    main()
