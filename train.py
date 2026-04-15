from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

from driving_dataset import DrivingDataset


TARGET_NAMES = ["steering", "throttle", "brake"]
DATASET_PATH = "dataset"


def set_seed(seed: int) -> None:
    print(f"[Setup] Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class DrivingModel(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        self.image_backbone = backbone.features
        self.image_pool = nn.AdaptiveAvgPool2d(1)

        image_feature_dim = 576
        numeric_feature_dim = 10

        self.numeric_mlp = nn.Sequential(
            nn.Linear(numeric_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(image_feature_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for images, features, targets in loader:
            images = images.to(device)
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(images, features)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def compute_mae_per_output(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    abs_sum = np.zeros(3, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for images, features, targets in loader:
            images = images.to(device)
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(images, features)
            abs_err = torch.abs(outputs - targets).sum(dim=0).cpu().numpy()

            abs_sum += abs_err
            count += images.size(0)

    mae = abs_sum / max(count, 1)
    return {name: float(mae[i]) for i, name in enumerate(TARGET_NAMES)}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    print(f"[Training] Epoch {epoch}/{total_epochs} - starting training pass")
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_idx, (images, features, targets) in enumerate(loader, start=1):
        images = images.to(device)
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images, features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        if batch_idx % 100 == 0:
            print(f"[Training] Epoch {epoch}/{total_epochs} - batch {batch_idx} processed")

    return total_loss / max(total_count, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento multimodal ETS2")
    parser.add_argument("--csv", type=str, default=f"{DATASET_PATH}/samples.csv")
    parser.add_argument("--images", type=str, default=f"{DATASET_PATH}/images")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    print("[Start] Training script started")
    args = parse_args()
    print(f"[Args] {vars(args)}")

    set_seed(args.seed)

    print("[Setup] Creating output directory...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"[Setup] Using device: {device}")

    print("[Setup] Building image transforms...")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    print("[Data] Loading dataset...")
    full_dataset = DrivingDataset(
        csv_path=args.csv,
        images_root=args.images,
        transform=transform,
        verify_images=True,
    )

    n = len(full_dataset)
    print(f"[Data] dataset loaded with {n} valid samples")

    if n < 100:
        raise ValueError(f"dataset demasiado pequeño: {n} muestras")

    print("[Data] Splitting dataset into train / validation / test...")
    indices = np.arange(n)
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=args.seed, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=args.seed, shuffle=True)

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)

    print(f"[Data] Train samples: {len(train_ds)}")
    print(f"[Data] Validation samples: {len(val_ds)}")
    print(f"[Data] Test samples: {len(test_ds)}")

    print("[Data] Building dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("[Model] Initializing model...")
    model = DrivingModel(pretrained=not args.no_pretrained).to(device)

    print("[Setup] Creating loss and optimizer...")
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_loss = float("inf")
    best_model_path = output_dir / "best_model.pt"
    last_model_path = output_dir / "last_model.pt"
    split_path = output_dir / "data_split.json"

    print("[Artifacts] Saving data split...")
    split_info = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }
    split_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    print("[Training] Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        print(f"[Training] Epoch {epoch}/{args.epochs}...")
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)

        print(f"[Validation] Epoch {epoch}/{args.epochs} - evaluating on validation set...")
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_mae = compute_mae_per_output(model, val_loader, device)

        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "epoch_seconds": elapsed,
        }
        history.append(row)

        print(
            f"[Epoch Summary] Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_mae(steer={val_mae['steering']:.4f}, "
            f"thr={val_mae['throttle']:.4f}, "
            f"brk={val_mae['brake']:.4f}) | "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("[Artifacts] New best model found, saving checkpoint...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "img_size": args.img_size,
                    "pretrained": not args.no_pretrained,
                    "model_name": "mobilenet_v3_small_multimodal",
                },
                best_model_path,
            )

    print("[Artifacts] Saving final model...")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "img_size": args.img_size,
            "pretrained": not args.no_pretrained,
            "model_name": "mobilenet_v3_small_multimodal",
        },
        last_model_path,
    )

    print("[Artifacts] Saving training history...")
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("[Evaluation] Running final test loss on last model...")
    test_loss = evaluate_loss(model, test_loader, criterion, device)

    summary = {
        "device": str(device),
        "dataset_size": n,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "best_val_loss": best_val_loss,
        "test_loss_last_model": test_loss,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "img_size": args.img_size,
        "lr": args.lr,
        "model_name": "mobilenet_v3_small_multimodal",
    }

    print("[Artifacts] Saving training summary...")
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[Done] Training finished")
    print(f"[Done] Best model: {best_model_path}")
    print(f"[Done] Last model: {last_model_path}")
    print(f"[Done] Split file: {split_path}")
    print(f"[Done] Final test loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
    