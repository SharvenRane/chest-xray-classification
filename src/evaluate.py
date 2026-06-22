"""Evaluate a trained checkpoint on the ChestMNIST test split and write a report.

Produces:
  - outputs/metrics.json        (machine-readable, real numbers)
  - docs/VALIDATION_REPORT.md   (human-readable, populated from real numbers)

Usage:
    python src/evaluate.py --checkpoint outputs/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

import numpy as np
import torch

from dataset import get_dataloaders, CHEST_LABELS
from model import build_model
from metrics import per_class_auc, mean_auc, sensitivity_specificity


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, gts = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        gts.append(y.numpy())
    return np.concatenate(gts), np.concatenate(probs)


def write_report(metrics: dict, path: str):
    auc = metrics["per_class_auc"]
    ss = metrics["sensitivity_specificity"]
    lines = [
        "# Model Validation Report — Chest X-ray Multi-Label Classifier",
        "",
        f"Generated: {date.today().isoformat()}  |  Model: {metrics['model']}  "
        f"|  Dataset: ChestMNIST (NIH ChestX-ray14)  |  Test n = {metrics['n_test']}",
        "",
        "All numbers below are produced by `src/evaluate.py` on the held-out test split.",
        "",
        f"**Mean ROC-AUC across 14 pathologies: {metrics['mean_auc']:.4f}**",
        "",
        "## Per-pathology performance (threshold = 0.5)",
        "",
        "| Pathology | ROC-AUC | Sensitivity | Specificity |",
        "|---|---|---|---|",
    ]
    for name in CHEST_LABELS:
        a = auc.get(name)
        s = ss.get(name, {})
        a_s = f"{a:.3f}" if a is not None else "n/a"
        lines.append(
            f"| {name} | {a_s} | {s.get('sensitivity', float('nan')):.3f} "
            f"| {s.get('specificity', float('nan')):.3f} |")
    lines += [
        "",
        "## Intended use",
        "Research and educational demonstration of a multi-label chest radiograph "
        "classifier with explicit class-imbalance handling and per-class validation. "
        "Not a medical device and not for clinical use.",
        "",
        "## Limitations",
        "- ChestMNIST is a standardized, down-sampled version of NIH ChestX-ray14; "
        "absolute AUCs are not directly comparable to full-resolution CheXNet results.",
        "- Labels are NLP-mined from radiology reports and carry known label noise.",
        "- No external/site-shift validation; performance may degrade on other scanners.",
        "",
        "See `docs/PCCP.md` for an illustrative change-control plan for this model.",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--report", default="docs/VALIDATION_REPORT.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]

    _, _, test_loader = get_dataloaders(
        size=cfg["image_size"], batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], data_dir=cfg.get("data_dir"))

    model = build_model(cfg["model"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    y_true, y_prob = predict(model, test_loader, device)
    auc = per_class_auc(y_true, y_prob, CHEST_LABELS)
    metrics = {
        "model": cfg["model"],
        "n_test": int(y_true.shape[0]),
        "mean_auc": mean_auc(auc),
        "per_class_auc": auc,
        "sensitivity_specificity": sensitivity_specificity(y_true, y_prob, CHEST_LABELS),
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    write_report(metrics, args.report)
    print(f"Mean test ROC-AUC: {metrics['mean_auc']:.4f}")
    print(f"Wrote outputs/metrics.json and {args.report}")


if __name__ == "__main__":
    main()
