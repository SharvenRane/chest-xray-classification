"""Train a multi-label chest X-ray classifier on ChestMNIST (NIH ChestX-ray14).

Real training loop: AMP mixed precision, class-imbalance-weighted BCE,
cosine LR schedule, MLflow logging, checkpoint on best validation mean AUC.

Usage:
    python src/train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from dataset import get_dataloaders, pos_weight, CHEST_LABELS
from model import build_model
from metrics import per_class_auc, mean_auc


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_auc(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    probs, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        gts.append(y.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(gts)
    return mean_auc(per_class_auc(y_true, y_prob, CHEST_LABELS)), y_true, y_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(f"Device: {device} | {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

    train_loader, val_loader, _ = get_dataloaders(
        size=cfg["image_size"], batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], data_dir=cfg.get("data_dir"))

    model = build_model(cfg["model"], pretrained=cfg["pretrained"]).to(device)

    print("Computing class imbalance weights...")
    pw = pos_weight(train_loader).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    try:
        import mlflow
        mlflow.set_experiment("chest-xray-classification")
        mlflow.start_run()
        mlflow.log_params({k: cfg[k] for k in ("model", "lr", "batch_size", "epochs", "image_size")})
        _mlflow = True
    except Exception as e:  # MLflow is optional, never block training on it
        print(f"MLflow disabled: {e}")
        _mlflow = False

    best_auc = -1.0
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg['epochs']}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()

        val_auc, _, _ = evaluate_auc(model, val_loader, device)
        train_loss = running / len(train_loader)
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f}  val_mean_auc={val_auc:.4f}")
        if _mlflow:
            mlflow.log_metrics({"train_loss": train_loss, "val_mean_auc": val_auc}, step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "cfg": cfg, "val_auc": val_auc},
                       os.path.join(cfg["output_dir"], "best.pt"))
            print(f"  -> saved new best (val_mean_auc={val_auc:.4f})")

    with open(os.path.join(cfg["output_dir"], "train_summary.json"), "w") as f:
        json.dump({"best_val_mean_auc": best_auc, "epochs": cfg["epochs"], "model": cfg["model"]}, f, indent=2)
    if _mlflow:
        mlflow.end_run()
    print(f"Done. Best val mean AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
